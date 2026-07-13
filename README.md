# DreamCube for Modly

Independent Modly integration for [DreamCube](https://github.com/Yukun-Huang/DreamCube), maintained by [DrHepa](https://github.com/DrHepa). It accepts a front RGB image, generates or uses a front depth map, runs DreamCube, and returns either an equirectangular panorama or a navigable scene manifest through Modly jobs.

This repository is the Modly extension, not the official DreamCube project. DreamCube source code and model weights remain third-party assets owned and licensed by their respective authors.

**Current extension version:** `0.1.1`

## Features

| Node | Output | Purpose |
| --- | --- | --- |
| `dreamcube/generate-panorama` | PNG image | Generate an equirectangular RGB panorama. |
| `dreamcube/generate-scene` | `modly.scene-manifest.v1` | Generate a navigable scene backed by GLB, with OBJ, 3DGS splat, and RGB-D sidecars. |

- Internal auto-depth by default using Depth Anything V2 Small.
- Optional manual depth input for users who already have a front-view depth map.
- Setup status and logs written under `.modly/setup/` for installation diagnostics.

## Requirements

- Modly with local model-extension support.
- Python 3 and `git` available to the setup environment.
- A CUDA/PyTorch-compatible NVIDIA GPU for practical DreamCube generation.
- Disk space for the extension-local virtual environment, the DreamCube upstream checkout, model weights, and runtime caches.

The currently validated development lane is Linux ARM64 with Python 3.12 and NVIDIA CUDA. Other operating-system, architecture, Python, or GPU lanes should be treated as unverified until they are tested; setup reports an actionable error instead of silently pretending that an unsupported dependency lane is ready.

## Install and setup

1. Open Modly's extension manager and choose **Install from GitHub**.
2. Use `https://github.com/DrHepa/modly-dreamcube-extension` as the repository URL.
3. Wait for the extension setup to finish and inspect the setup log if Modly reports a partial or failed installation.
4. Download the DreamCube weights from Modly's model UI before the first generation.

For local development, clone or copy the repository into Modly's extensions directory and reload the extension. Keep `manifest.json`, `setup.py`, `generator.py`, `dreamcube_mesh.py`, `README.md`, and `LICENSE` together.

`setup.py` prepares the extension runtime: it creates or reuses an extension-local environment, clones or updates DreamCube upstream source into `.modly/upstream/DreamCube`, installs runtime dependencies, records setup evidence, and validates core imports before marking the extension ready.

Setup evidence is written to:

- `.modly/setup/setup-status.json`
- `.modly/setup/logs/setup.log`

No Gradio server is launched; all execution happens through Modly jobs.

## Model weights

DreamCube model weights are managed by the Modly UI/downloader, not by `setup.py`.

- Hugging Face repository: <https://huggingface.co/KevinHuang/DreamCube>
- Local Modly owner path: `models/dreamcube/dreamcube`
- Download check: `model_index.json`

If readiness reports missing weights, use Modly's model download flow for DreamCube. Re-running setup will not download the DreamCube weights.

## Auto-depth behavior

By default, `depth_mode=auto` generates the front depth map inside this extension using:

- Repository: `depth-anything/Depth-Anything-V2-Small-hf`
- Variant: `vits`
- Cache directory: `.modly/auto-depth/cache`

Auto-depth weights are not downloaded during setup. They are cached lazily by Transformers the first time auto-depth generation is used. This extension has no dependency on a separate `modly-depth-anything` extension.

If `depth_image_path` is provided while `depth_mode=auto`, the supplied depth image is used instead of generating auto-depth. If `depth_mode=manual`, `depth_image_path` is required.

## Usage

Use the DreamCube nodes from Modly:

### Generate panorama

Run `dreamcube/generate-panorama` with a front RGB image, six directional prompts, and either auto-depth or a supplied depth image. The node returns an equirectangular RGB panorama PNG.

Important parameters:

- `depth_mode`: `auto` or `manual`
- `depth_image_path`: optional in auto mode, required in manual mode
- `save_input_depth`: saves generated auto-depth as `input_front_depth_auto.png`
- `num_inference_steps`, `guidance_scale`, `normalize_scale`
- `max_equi_size`, `max_cube_size`
- `seed`: set `-1` for pipeline default randomness

### Generate scene

Run `dreamcube/generate-scene` with the same image, depth, prompt, inference, size, and seed controls. The returned path is always the run's `scene-manifest.json`; there is no returned-format selector on this node.

The scene output contract is strict:

- `output_mesh.glb` is required and is the single visible `base-scene` asset in the manifest.
- `output_mesh.obj` and `output_3dgs.splat` remain sidecars in the same run directory.
- OBJ-to-GLB conversion errors, missing GLB output, and zero-byte GLB output fail generation instead of falling back to OBJ.
- The manifest asset's `workspacePath` is derived from the absolute, existing `WORKSPACE_DIR` supplied by Modly. Generation fails if that contract is unavailable, and assets outside that root are rejected.
- The initial view starts at the RGB-D camera origin and looks along +Z with +Y up.

#### Mesh reconstruction and filtering

OBJ meshes are reconstructed by the extension-owned `dreamcube_mesh.py`; the wrapper does not call DreamCube's upstream mesh converters. Cubemap meshes preserve six face grids plus seam and corner triangles, while equirectangular meshes preserve horizontal wrap. Invalid, non-finite, non-positive, or sub-centimeter depth samples are repaired deterministically from the nearest valid parameter-grid neighbor before topology is exported; exact 0.01 m and larger samples are valid.

`mesh_depth_jump_threshold` is scene-only and defaults to `0.20`. Triangles touching repaired or invalid depth are always removed; remaining triangles whose relative depth jump exceeds `(max_depth - min_depth) / max(min_depth, 1e-6)` are also removed. Set the threshold to `0` to disable only depth-jump removal. The advanced `mesh_footprint_ratio_threshold` (default `12`) rejects edges that are too long for their angular footprint, and `mesh_aspect_ratio_threshold` (default `10`) rejects needle/sliver triangles; set either geometric threshold to `0` to disable that filter. No absolute edge-length cutoff is used. `max_cube_size` is capped at 512 and `max_equi_size` at 2048 to control reconstruction cost.

The exported scene is an origin-centered RGB-D panorama shell intended for interior navigation. Its irregular radial silhouette when viewed from outside is inherent to the distance-times-ray representation and is not a conventional watertight object mesh; changing that silhouette would discard or reshape DreamCube’s generated depth and parallax.

Each successful scene run records reconstruction settings, coordinate-frame information, presentation metadata, and mesh statistics in `run_metadata.json`, including repaired samples, repair rounds, adaptive diagonals, removals by invalid/repaired depth, depth discontinuity, footprint ratio, aspect ratio, exported triangles, and retention. No support-plane geometry or private duplicate vertices are fabricated. If reconstruction or required GLB creation fails, metadata records the failure stage, error, available mesh statistics, and file diagnostics before the job fails. GLB is created from the already written OBJ with Trimesh using non-processing load when possible, preserving geometry/colors and adding normals plus a double-sided PBR material. The OBJ/GLB and 3DGS exports use the same canonical right-handed rays (X left, Y up, Z forward), bypassing DreamCube upstream's implicit X/Y flip when `rays` is omitted.

## Validation

Run these checks from this extension directory:

```bash
python3 validate_extension.py
python3 setup.py --validate-only
```

`validate_extension.py` checks the manifest/generator contract without loading DreamCube or heavy runtime dependencies. `setup.py --validate-only` validates setup configuration and manifest consistency without cloning upstream, installing dependencies, or downloading model weights.

## Troubleshooting

- **Setup fails**: inspect `.modly/setup/logs/setup.log` and `.modly/setup/setup-status.json`.
- **Weights are missing**: download `KevinHuang/DreamCube` through Modly's model downloader. Setup intentionally does not download these weights.
- **Auto-depth is slow on first run**: the first auto-depth generation may download and cache `depth-anything/Depth-Anything-V2-Small-hf` under `.modly/auto-depth/cache`.
- **Manual depth fails**: provide a valid front-view depth image in `depth_image_path`, or switch `depth_mode` back to `auto`.
- **No web UI appears**: expected. This extension does not launch Gradio.

## Roadmap

Version `0.2.0` is planned to add a manual RGB-D cubemap workflow that accepts six user-supplied RGB faces and six matching metric depth faces. That node is intentionally **not included** in version `0.1.1`; the current release contains only the validated panorama and generated-scene nodes documented above.

## License and third-party notices

The Modly wrapper code in this repository is copyright DrHepa and released under the MIT License. See [LICENSE](LICENSE). Third-party licensing is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

DreamCube upstream source and the `KevinHuang/DreamCube` model weights are separate third-party assets licensed under Apache-2.0 by their respective authors. See:

- <https://github.com/Yukun-Huang/DreamCube>
- <https://huggingface.co/KevinHuang/DreamCube>
