# DreamCube for Modly

Independent Modly integration for [DreamCube](https://github.com/Yukun-Huang/DreamCube), maintained by [DrHepa](https://github.com/DrHepa). It accepts a front RGB image, generates or uses a front depth map, estimates sparse cubemap depth, runs DreamCube when requested, and returns images or a GLB mesh through Modly jobs; scene manifests are retained as auxiliary sidecars.

This repository is the Modly extension, not the official DreamCube project. DreamCube source code and model weights remain separate third-party assets. The pinned upstream source does not declare a source-code license; the model repositories declare their own weight licenses.

**Current extension version:** `0.3.0`

## Features

| Node | Output | Purpose |
| --- | --- | --- |
| `dreamcube/generate-panorama` | PNG image | Generate an equirectangular RGB panorama. |
| `dreamcube/generate-scene` | GLB mesh | Generate a navigable mesh; `scene-manifest.json`, OBJ, 3DGS splat, and RGB-D files remain sidecars. |
| `dreamcube/estimate-cubemap-depths` | `front_depth.png` (primary) + sidecars | Estimate non-metric radial-depth from supplied cubemap RGB faces; output includes the front primary image and additional face/depth sidecars in the run folder. |
| `dreamcube/generate-scene-manual-cubemap` | GLB mesh | Generate a mesh from six supplied RGB faces and six paired 16-bit radial-depth PNG faces; retain `scene-manifest.json` as a sidecar. |

- Internal auto-depth by default using Depth Anything V2 Small.
- Optional manual depth input for users who already have a front-view depth map.
- Setup status and logs written under `.modly/setup/` for installation diagnostics.

## Requirements

- Modly with local model-extension support.
- Python 3 and `git` available to the setup environment.
- A CUDA/PyTorch-compatible NVIDIA GPU for practical DreamCube generation.
- Disk space for the extension-local virtual environment, the DreamCube upstream checkout, model weights, and runtime caches.

The currently validated development lane is Linux ARM64 with Python 3.12 and NVIDIA CUDA. Other operating-system, architecture, Python, or GPU lanes should be treated as unverified until they are tested; setup reports an actionable error instead of silently pretending that an unsupported dependency lane is ready.

## Installation

1. Open Modly's extension manager and choose **Install from GitHub**.
2. Use `https://github.com/DrHepa/modly-dreamcube-extension` as the repository URL.
3. Wait for the extension setup to finish and inspect the setup log if Modly reports a partial or failed installation.
4. Download the DreamCube weights from Modly's model UI before using panorama or scene nodes. The cubemap-depth node does not require them.

For local development, clone or copy the repository into Modly's extensions directory and reload the extension. Keep `manifest.json`, `setup.py`, `generator.py`, `dreamcube_mesh.py`, `dreamcube_manual_cubemap.py`, `dreamcube_cubemap_depth.py`, `README.md`, and `LICENSE` together.

`setup.py` prepares the extension runtime: it creates or reuses an extension-local environment, clones or repairs DreamCube upstream source into `.modly/upstream/DreamCube`, checks out pinned commit `aa04a53c6542581b5b0a6faa575865d2d57b5243` detached, installs runtime dependencies, records setup evidence, and validates core imports before marking the extension ready.

Setup evidence is written to:

- `.modly/setup/setup-status.json`
- `.modly/setup/logs/setup.log`

No Gradio server is launched; all execution happens through Modly jobs. Setup is dependency-only: it never downloads DreamCube or Depth Anything model weights.

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

The same lazy model and cache are reused by `dreamcube/estimate-cubemap-depths`. That node does not require DreamCube weights and never loads the DreamCube pipeline.

## Usage

Use the DreamCube nodes from Modly.

## Parameters

Key parameters are grouped by node below:

### Generate panorama

Run `dreamcube/generate-panorama` with a front RGB image, six directional prompts, and either auto-depth or a supplied depth image. The node returns an equirectangular RGB panorama PNG.

Important parameters for panorama generation:

- `depth_mode`: `auto` or `manual`
- `depth_image_path`: optional in auto mode, required in manual mode
- `save_input_depth`: saves generated auto-depth as `input_front_depth_auto.png`
- `num_inference_steps`, `guidance_scale`, `normalize_scale`
- `max_equi_size`, `max_cube_size`
- `seed`: set `-1` for pipeline default randomness

### Generate scene

Run `dreamcube/generate-scene` with the same image, depth, prompt, inference, size, and seed controls. The returned path is always the run's `output_mesh.glb`; there is no returned-format selector on this node. `scene-manifest.json` is still written as an auxiliary future-ready sidecar.

The scene output contract is strict:

- `output_mesh.glb` is required and is the primary Modly result; it is also the single visible `base-scene` asset in the auxiliary scene manifest.
- `output_mesh.obj` and `output_3dgs.splat` remain sidecars in the same run directory.
- OBJ-to-GLB conversion errors, missing GLB output, and zero-byte GLB output fail generation instead of falling back to OBJ.
- The manifest asset's `workspacePath` is derived from the absolute, existing `WORKSPACE_DIR` supplied by Modly. Generation fails if that contract is unavailable, and assets outside that root are rejected.
- The initial view starts at the RGB-D camera origin and looks along +Z with +Y up.


### Estimate cubemap depths

Run `dreamcube/estimate-cubemap-depths` with a required front RGB image input and optional cubemap face RGB pickers.

Input contract:

- `front` RGB input is required and always used as the first face.
- `rgb_right_path`, `rgb_back_path`, `rgb_left_path`, `rgb_top_path`, `rgb_bottom_path` are optional image pickers.
- Any front-containing subset is accepted (for example, `front`, `front+right`, `front+right+left`, etc. up to all six faces), and inference runs only for supplied faces.

Output contract:

- The single primary Modly result is `front_depth.png`.
- Every other supplied depth map, together with `depth-estimation.json`, is written as sidecars in the same run folder.
- Missing faces are never synthesized.

This contract preserves joint postprocessing and returns grayscale uint16 radial-depth images. It remains non-metric and includes the same quality limitations as before.

Depth Anything V2 Small predicts relative depth, not calibrated geometry. The extension replaces non-finite values, jointly normalizes all supplied predictions, fits positive affine alignment only within connected components of the observed cube-face adjacency graph, uses the front face as gauge when available, maps the global estimated z range to 1000..5000 mm, converts 90-degree z-distance to radial distance, and makes only observed shared borders and corners exactly equal. Files are grayscale uint16 PNG radial-millimetre encodings, but the values remain explicitly estimated and non-metric.

Quality is limited by monocular ambiguity, independent per-face inference, textureless or reflective regions, occlusion, and missing adjacency. Disconnected subsets such as `front+back` are marked degraded because no observed shared edge can align their components. No missing face is synthesized.

### Generate scene from a manual RGB-D cubemap

Run `dreamcube/generate-scene-manual-cubemap` when you have a complete cubemap. The manual workflow keeps one required front normal image input, five required RGB image pickers (`rgb_right_path`, `rgb_back_path`, `rgb_left_path`, `rgb_top_path`, `rgb_bottom_path`), and six required depth pickers (`depth_front_path` through `depth_bottom_path`). After the depth node runs, select `front_depth.png` and the optional face-depth sidecars from that run folder with the manual node's depth path pickers.

Generation requires all six RGB faces and all six depth faces. The front RGB remains the required primary image input; every other RGB face and all six depth faces are supplied through the path pickers.

> **Depth quality directly affects the final mesh.** The manual node requires six matched RGB/depth pairs: every depth map must be geometrically aligned with its RGB face, and all six pairs must share the same cubemap camera origin, orientation, and metric scale. More accurate, clean, and cross-face-consistent depth generally produces more coherent geometry. Noisy, blurred, incorrectly scaled, RGB-misaligned, or mutually inconsistent depth can warp surfaces, enlarge gaps at face joins, or cause unsafe triangles to be discarded even when the RGB images look correct. The node does not run auto-depth to repair or replace the supplied depth maps.

Manual cubemap contract:

- Face order is `front/right/back/left/top/bottom`; axes are `+Z/-X/-Z/+X/+Y/-Y`.
- All views must share one origin and be square 90-degree cubemap faces. No crop, rotation, or silent correction is applied.
- Depth files must be single-channel 16-bit PNG radial distance in millimetres. 8-bit or colorized depth, missing files, inconsistent dimensions, non-square faces, or more than 1% non-positive/invalid depth are rejected before inference.
- Relative depth mismatch is checked across the 12 matching cube edges. Preflight fails when edge p95 mismatch exceeds `0.50`; manual cubemap failures are written under `manual_cubemap` in `run_metadata.json` when practical.
- Inputs are resized only after validation: RGB with quality resampling, depth with nearest-neighbor resampling, using `max_cube_size` in the manifest-valid 256..512 range.
- Every supplied manual RGB-D pixel remains conditioned. The required upstream mask tensor is entirely false, so no border band or other pixels are marked for generation.
- The manual node calls the loaded DreamCube pipeline directly with all six RGB/depth/mask tensors. It does not use auto-depth, `app.inference`, or a fallback path.

When `save_all_outputs` is enabled, generated nodes save `output_faces/{face}_rgb.png` and `output_faces/{face}_depth_mm.png` so those faces can be fed into the manual cubemap node. The manual node always saves deterministic `input_faces/` and `output_faces/` sidecars in addition to existing equirectangular and dice outputs.

#### Mesh reconstruction and filtering

OBJ meshes are reconstructed by the extension-owned `dreamcube_mesh.py`; the wrapper does not call DreamCube's upstream mesh converters. Cubemap meshes preserve six face grids and use shared/welded border indices for safe cube joins; seam-strip and corner candidate triangles are suppressed, and unsafe or intentional depth-discontinuity holes may remain open and are diagnosed. Equirectangular meshes preserve horizontal wrap. Invalid, non-finite, non-positive, or sub-centimeter depth samples are repaired deterministically from the nearest valid parameter-grid neighbor before topology is exported; exact 0.01 m and larger samples are valid.

`mesh_depth_jump_threshold` is scene-only and defaults to `0.20`. Triangles touching repaired or invalid depth are always removed; remaining triangles whose relative depth jump exceeds `(max_depth - min_depth) / max(min_depth, 1e-6)` are also removed. Set the threshold to `0` to disable only depth-jump removal. The advanced `mesh_footprint_ratio_threshold` (default `12`) rejects edges that are too long for their angular footprint, and `mesh_aspect_ratio_threshold` (default `10`) rejects needle/sliver triangles; set either geometric threshold to `0` to disable that filter. No absolute edge-length cutoff is used. `max_cube_size` is capped at 512 and `max_equi_size` at 2048 to control reconstruction cost.

The exported scene is an origin-centered RGB-D panorama shell intended for interior navigation. Its irregular radial silhouette when viewed from outside is inherent to the distance-times-ray representation and is not a conventional watertight object mesh; changing that silhouette would discard or reshape DreamCube’s generated depth and parallax.

Each successful scene run records the GLB primary output, `scene-manifest.json` sidecar status/path, reconstruction settings, coordinate-frame information, presentation metadata, and mesh statistics in `run_metadata.json`, including repaired samples, repair rounds, adaptive diagonals, removals by invalid/repaired depth, depth discontinuity, footprint ratio, aspect ratio, exported triangles, and retention. No support-plane geometry or private duplicate vertices are fabricated. If reconstruction or required GLB creation fails, metadata records the failure stage, error, available mesh statistics, and file diagnostics before the job fails. GLB is created from the already written OBJ with Trimesh using non-processing load when possible, preserving geometry/colors and adding normals. Equirectangular scene GLBs use a double-sided PBR material; cubemap scene GLBs are single-sided and inward-facing so the room interior remains the intended visible surface. The OBJ/GLB and 3DGS exports use the same canonical right-handed rays (X left, Y up, Z forward), bypassing DreamCube upstream's implicit X/Y flip when `rays` is omitted.

## Outputs

- `dreamcube/generate-panorama` returns an equirectangular RGB PNG.
- `dreamcube/generate-scene` returns `output_mesh.glb` as the primary Modly mesh output. `scene-manifest.json`, OBJ, 3DGS splat, and RGB-D files remain sidecars in the same run directory.
- `dreamcube/estimate-cubemap-depths` returns `front_depth.png` as the primary Modly image output, with all other supplied face depths and `depth-estimation.json` as run-folder sidecars; outputs remain estimated relative radial-depth encodings and are not metric measurements.
- `dreamcube/generate-scene-manual-cubemap` returns `output_mesh.glb` from supplied RGB-D cubemap faces; deterministic input/output face files and `scene-manifest.json` remain sidecars/resources.

## Validation

Run these checks from this extension directory:

```bash
python3 validate_extension.py
python3 setup.py --validate-only
```

`validate_extension.py` checks the manifest/generator contract without loading DreamCube or heavy runtime dependencies. `setup.py --validate-only` validates setup configuration and manifest consistency without cloning upstream, installing dependencies, or downloading model weights.

## Limitations

- Practical generation requires a CUDA/PyTorch-compatible NVIDIA GPU; the currently validated development lane is Linux ARM64 with Python 3.12 and NVIDIA CUDA.
- Other operating-system, architecture, Python, or GPU lanes are unverified until tested.
- Setup never downloads DreamCube or Depth Anything model weights. Auto-depth weights are cached lazily on first use under `.modly/auto-depth/cache`.
- Estimated cubemap depth is relative and non-metric; sparse or disconnected face sets have weaker cross-face consistency even after observed-edge alignment.
- Manual RGB-D cubemap inputs must satisfy the strict face-order, square 90-degree view, and 16-bit radial-depth contracts described above.

## Troubleshooting

- **Setup fails**: inspect `.modly/setup/logs/setup.log` and `.modly/setup/setup-status.json`.
- **Weights are missing**: download `KevinHuang/DreamCube` through Modly's model downloader. Setup intentionally does not download these weights.
- **Auto-depth is slow on first run**: the first panorama, scene, or cubemap-depth request that needs auto-depth may download and cache `depth-anything/Depth-Anything-V2-Small-hf` under `.modly/auto-depth/cache`.
- **Manual depth fails**: for generated nodes, provide a valid front-view depth image in `depth_image_path`, or switch `depth_mode` back to `auto`. For the manual cubemap node, provide six single-channel 16-bit PNG radial-depth faces with matching square dimensions.
- **No web UI appears**: expected. This extension does not launch Gradio.

## Credits

- Extension/wrapper author: DrHepa.
- DreamCube upstream source: Yukun Huang and DreamCube contributors, <https://github.com/Yukun-Huang/DreamCube>, pinned to `aa04a53c6542581b5b0a6faa575865d2d57b5243`; source-code license not declared (`NOASSERTION`). The repository and pinned checkout contain no `LICENSE` file or other source-license grant as verified on 2026-07-16.
- DreamCube model weights: `KevinHuang/DreamCube`; its model repository declares Apache-2.0 for the weights.
- Depth Anything V2 Small model: `depth-anything/Depth-Anything-V2-Small-hf`, declared Apache-2.0 by its model repository.
- Host application/project: Modly by Lightning Pixel.

## License and third-party notices

The Modly wrapper code in this repository is copyright DrHepa and released under the MIT License. See [LICENSE](LICENSE). That MIT license applies only to the wrapper and does not relicense DreamCube source code or model weights. Third-party licensing is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The pinned DreamCube upstream source has no declared source-code license and is recorded as `NOASSERTION`. Separately, the `KevinHuang/DreamCube` and `depth-anything/Depth-Anything-V2-Small-hf` model repositories each declare Apache-2.0 for their weights. Those weight-license declarations do not establish a license for DreamCube source code. See:

- <https://github.com/Yukun-Huang/DreamCube>
- <https://huggingface.co/KevinHuang/DreamCube>
- <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>
