"""Manual RGB-D cubemap preparation for DreamCube."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

FACE_ORDER = ("front", "right", "back", "left", "top", "bottom")
FACE_AXES = {
    "front": "+Z",
    "right": "-X",
    "back": "-Z",
    "left": "+X",
    "top": "+Y",
    "bottom": "-Y",
}
RGB_PATH_PARAMS = {
    "right": "rgb_right_path",
    "back": "rgb_back_path",
    "left": "rgb_left_path",
    "top": "rgb_top_path",
    "bottom": "rgb_bottom_path",
}
DEPTH_PATH_PARAMS = {face: f"depth_{face}_path" for face in FACE_ORDER}
EXTENSION_DIR = Path(__file__).resolve().parent
OFFICIAL_PROMPT_PREFIXES = (
    "This is one view of a scene. {prompt}",
    "This is one view of a scene. {prompt}",
    "This is one view of a scene. {prompt}",
    "This is one view of a scene. {prompt}",
    "This a upward view of a scene. {prompt}",
    "This a downward view of a scene. {prompt}",
)

# Image-space edge pairs after structural validation; inputs are never cropped or rotated.
# Tuple order is: a_face, a_edge, b_face, b_edge, reverse_b.
EDGE_PAIRS = (
    ("front", "right", "right", "left", False),
    ("front", "left", "left", "right", False),
    ("front", "top", "top", "bottom", False),
    ("front", "bottom", "bottom", "top", False),
    ("back", "right", "left", "left", False),
    ("back", "left", "right", "right", False),
    ("back", "top", "top", "top", True),
    ("back", "bottom", "bottom", "bottom", True),
    ("right", "top", "top", "right", True),
    ("right", "bottom", "bottom", "right", False),
    ("left", "top", "top", "left", False),
    ("left", "bottom", "bottom", "left", True),
)


@dataclass(frozen=True)
class ManualCubemapInputs:
    rgbs: dict[str, Image.Image]
    depths_mm: dict[str, np.ndarray]
    source_size: tuple[int, int]
    source_stats: dict[str, Any]
    seam_metrics: dict[str, Any]


def resolve_existing_path(value: Any, outputs_dir: Path, *, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"DreamCube manual cubemap requires {label}.")
    raw = Path(os.path.expandvars(str(value).strip())).expanduser()
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [EXTENSION_DIR / raw, Path(outputs_dir) / raw]
        workspace_dir = os.environ.get("WORKSPACE_DIR")
        if workspace_dir and workspace_dir.strip():
            candidates.append(Path(os.path.expandvars(workspace_dir.strip())).expanduser() / raw)
        candidates.extend([Path.cwd() / raw, raw])

    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"DreamCube manual cubemap file does not exist for {label}. Checked: {'; '.join(checked)}"
    )


def _edge(array: np.ndarray, edge: str) -> np.ndarray:
    if edge == "left":
        return array[:, 0]
    if edge == "right":
        return array[:, -1]
    if edge == "top":
        return array[0, :]
    if edge == "bottom":
        return array[-1, :]
    raise ValueError(f"Unknown edge {edge!r}")


def _png_depth_header(path: Path) -> tuple[int | None, int | None]:
    try:
        header = path.read_bytes()[:29]
    except OSError:
        return None, None
    return _png_depth_header_bytes(header)


def _png_depth_header_bytes(
    data: bytes | bytearray | memoryview,
) -> tuple[int | None, int | None]:
    header = bytes(data)[:29]
    if len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None, None
    return int(header[24]), int(header[25])


def _image_bit_depth(image: Image.Image, path: Path | None = None) -> int:
    if path is not None:
        bit_depth, _color_type = _png_depth_header(path)
        if bit_depth is not None:
            return bit_depth
    if image.mode in {"I;16", "I;16L", "I;16B", "I;16N"}:
        return 16
    if image.mode == "I":
        return 32
    return 8


def load_manual_cubemap_inputs(
    *,
    front_rgb_bytes: bytes | bytearray | memoryview,
    params: Mapping[str, Any],
    outputs_dir: Path,
) -> ManualCubemapInputs:
    if not isinstance(front_rgb_bytes, (bytes, bytearray, memoryview)) or not front_rgb_bytes:
        raise ValueError("DreamCube manual cubemap requires non-empty front RGB image bytes.")

    rgbs: dict[str, Image.Image] = {}
    depths: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {"faces": {}, "face_order": list(FACE_ORDER), "face_axes": dict(FACE_AXES)}

    with Image.open(io.BytesIO(bytes(front_rgb_bytes))) as image:
        rgbs["front"] = image.convert("RGB").copy()
        stats["faces"]["front"] = {
            "rgb_source": "modly_image_input",
            "rgb_mode": image.mode,
            "rgb_size": list(image.size),
        }

    for face, param_id in RGB_PATH_PARAMS.items():
        path = resolve_existing_path(params.get(param_id), outputs_dir, label=param_id)
        with Image.open(path) as image:
            rgbs[face] = image.convert("RGB").copy()
            stats["faces"][face] = {
                "rgb_source": str(path),
                "rgb_mode": image.mode,
                "rgb_size": list(image.size),
            }

    expected_size = rgbs["front"].size
    if expected_size[0] != expected_size[1]:
        raise ValueError(f"DreamCube manual cubemap requires square RGB faces; front is {expected_size}.")
    for face in FACE_ORDER:
        if rgbs[face].size != expected_size:
            raise ValueError(
                f"DreamCube manual cubemap requires all RGB faces to share size {expected_size}; "
                f"{face} is {rgbs[face].size}."
            )

    for face, param_id in DEPTH_PATH_PARAMS.items():
        depth_path = resolve_existing_path(params.get(param_id), outputs_dir, label=param_id)
        png_bit_depth, png_color_type = _png_depth_header(depth_path)
        with Image.open(depth_path) as image:
            bit_depth = _image_bit_depth(image, depth_path)
            bands = image.getbands()
            if png_bit_depth != 16 or png_color_type != 0 or len(bands) != 1:
                raise ValueError(
                    f"DreamCube manual cubemap requires single-channel 16-bit grayscale PNG depth for {face}; "
                    f"got mode={image.mode!r}, bands={bands!r}, bit_depth={bit_depth}, png_color_type={png_color_type}."
                )
            if image.size != expected_size:
                raise ValueError(f"DreamCube manual cubemap requires depth {face} size {expected_size}; got {image.size}.")
            arr = np.asarray(image, dtype=np.uint16)
        invalid = arr <= 0
        invalid_ratio = float(np.count_nonzero(invalid) / arr.size)
        if invalid_ratio > 0.01:
            raise ValueError(
                f"DreamCube manual cubemap rejects {face} depth: invalid/nonpositive ratio "
                f"{invalid_ratio:.4%} exceeds 1%."
            )
        depths[face] = arr
        stats["faces"].setdefault(face, {})
        stats["faces"][face].update(
            {
                "depth_source": str(depth_path),
                "depth_mode": "I;16",
                "depth_bit_depth": 16,
                "depth_size": list(image.size),
                "depth_min_mm": int(arr[arr > 0].min()) if np.any(arr > 0) else 0,
                "depth_max_mm": int(arr.max()),
                "depth_invalid_ratio": invalid_ratio,
            }
        )

    seam_metrics = validate_depth_seams(depths)
    failures = [item for item in seam_metrics["edges"] if item["p95_relative_mismatch"] > 0.50]
    if failures:
        worst = max(failures, key=lambda item: item["p95_relative_mismatch"])
        raise ValueError(
            "DreamCube manual cubemap depth seam preflight failed: "
            f"{len(failures)} edge(s) exceed p95 <= 0.50; worst "
            f"{worst['a_face']}.{worst['a_edge']} ↔ {worst['b_face']}.{worst['b_edge']} "
            f"p95={worst['p95_relative_mismatch']:.3f}."
        )

    return ManualCubemapInputs(
        rgbs=rgbs,
        depths_mm=depths,
        source_size=expected_size,
        source_stats=stats,
        seam_metrics=seam_metrics,
    )


def validate_depth_seams(depths_mm: Mapping[str, np.ndarray]) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    for a_face, a_edge, b_face, b_edge, reverse_b in EDGE_PAIRS:
        a = _edge(depths_mm[a_face].astype(np.float32), a_edge)
        b = _edge(depths_mm[b_face].astype(np.float32), b_edge)
        if reverse_b:
            b = b[::-1]
        denom = np.maximum(np.maximum(a, b), 1.0)
        mismatch = np.abs(a - b) / denom
        edges.append(
            {
                "a_face": a_face,
                "b_face": b_face,
                "a_edge": a_edge,
                "b_edge": b_edge,
                "p95_relative_mismatch": float(np.percentile(mismatch, 95)),
                "max_relative_mismatch": float(np.max(mismatch)),
            }
        )
    return {
        "threshold_p95": 0.50,
        "edges": edges,
        "max_p95_relative_mismatch": max(edge["p95_relative_mismatch"] for edge in edges),
    }


def resize_for_inference(inputs: ManualCubemapInputs, size: int = 512) -> tuple[list[Image.Image], list[np.ndarray]]:
    rgbs = [inputs.rgbs[face].resize((size, size), Image.Resampling.LANCZOS) for face in FACE_ORDER]
    depths = [
        np.asarray(
            Image.fromarray(np.asarray(inputs.depths_mm[face], dtype=np.uint16)).resize((size, size), Image.Resampling.NEAREST),
            dtype=np.uint16,
        )
        for face in FACE_ORDER
    ]
    return rgbs, depths


def prefixed_prompts(prompts: list[str]) -> list[str]:
    if len(prompts) != 6:
        raise ValueError("DreamCube manual cubemap requires exactly six prompts.")
    return [template.format(prompt=prompt) for template, prompt in zip(OFFICIAL_PROMPT_PREFIXES, prompts)]


def save_input_faces(run_dir: Path, inputs: ManualCubemapInputs) -> None:
    out = run_dir / "input_faces"
    out.mkdir(parents=True, exist_ok=True)
    for face in FACE_ORDER:
        inputs.rgbs[face].save(out / f"{face}_rgb.png")
        Image.fromarray(np.asarray(inputs.depths_mm[face], dtype=np.uint16)).save(out / f"{face}_depth_mm.png")
