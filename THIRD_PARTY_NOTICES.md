# Third-Party Notices

## DreamCube

This repository contains an independent Modly wrapper for DreamCube. The MIT License in [LICENSE](LICENSE) applies only to the wrapper code maintained in this repository.

### Upstream source code

- Source: <https://github.com/Yukun-Huang/DreamCube>
- Pinned revision: `aa04a53c6542581b5b0a6faa575865d2d57b5243`
- Source-code license status: not declared (`NOASSERTION`)

As verified on 2026-07-16, the upstream repository and pinned checkout contain no `LICENSE` file or other source-license grant. This extension therefore makes no claim that DreamCube source code is Apache-2.0 licensed. The wrapper's MIT License does not apply to or relicense that upstream source.

### DreamCube model weights

- Model repository and weights: <https://huggingface.co/KevinHuang/DreamCube>
- License declared by the model repository for its weights: Apache-2.0
- Apache License 2.0 text: <https://www.apache.org/licenses/LICENSE-2.0>

The model repository's Apache-2.0 declaration applies to its weight artifacts; it does not supply a license for the separate DreamCube source checkout. All rights and attribution remain with the respective DreamCube authors and contributors. This repository does not relicense either third-party asset under MIT.

## Depth Anything V2 Small

The extension uses the separate `depth-anything/Depth-Anything-V2-Small-hf` checkpoint for lazy runtime relative-depth estimation:

- Model card and weights: <https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf>
- License declared by the model repository for its weights: Apache-2.0
- Upstream project: <https://github.com/DepthAnything/Depth-Anything-V2>

The checkpoint is downloaded only when auto-depth is first used. It is not included in this repository and is not downloaded by `setup.py`. Its Apache-2.0 weight-license declaration is separate from the undeclared DreamCube source-code license. All model, paper, and upstream attribution remains with the Depth Anything authors and contributors.
