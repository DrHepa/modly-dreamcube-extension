"""Modly generator entry point for the DreamCube model extension."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import uuid
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from collections.abc import Set as SetABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except ModuleNotFoundError:  # pragma: no cover - standalone tests run outside Modly
    class GenerationCancelled(Exception):
        """Fallback cancellation exception used outside the Modly runtime."""

    class BaseGenerator:  # type: ignore[override]
        MODEL_ID = ""
        DISPLAY_NAME = ""
        VRAM_GB = 0

        def __init__(self, model_dir: Path | str, outputs_dir: Path | str) -> None:
            self.model_dir = Path(model_dir)
            self.outputs_dir = Path(outputs_dir)
            self._model = None
            self.hf_repo = ""
            self.hf_skip_prefixes: list[str] = []
            self.download_check = ""
            self._params_schema: list[dict[str, Any]] = []

        def is_downloaded(self) -> bool:
            if self.download_check:
                return (self.model_dir / self.download_check).exists()
            return self.model_dir.exists() and any(self.model_dir.iterdir())

        def _check_cancelled(self, cancel_event: Any | None) -> None:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise GenerationCancelled()


EXTENSION_DIR = Path(__file__).resolve().parent
if str(EXTENSION_DIR) not in sys.path:
    sys.path.insert(0, str(EXTENSION_DIR))

import dreamcube_mesh
import dreamcube_manual_cubemap

MANIFEST_PATH = EXTENSION_DIR / "manifest.json"
SETUP_STATUS_PATH = EXTENSION_DIR / ".modly" / "setup" / "setup-status.json"
UPSTREAM_DIR = EXTENSION_DIR / ".modly" / "upstream" / "DreamCube"
UPSTREAM_APP_PATH = UPSTREAM_DIR / "app.py"

DOWNLOAD_CHECK = "model_index.json"
HF_REPO = "KevinHuang/DreamCube"
AUTO_DEPTH_DEFAULT_VARIANT = "vits"
AUTO_DEPTH_VARIANTS: dict[str, dict[str, str]] = {
    "vits": {
        "label": "Depth Anything V2 Small",
        "repo_id": "depth-anything/Depth-Anything-V2-Small-hf",
    },
}
AUTO_DEPTH_CACHE_DIR = EXTENSION_DIR / ".modly" / "auto-depth" / "cache"

PANORAMA_NODE_ID = "generate-panorama"
SCENE_NODE_ID = "generate-scene"
MANUAL_SCENE_NODE_ID = "generate-scene-manual-cubemap"
NODE_IDS = {PANORAMA_NODE_ID, SCENE_NODE_ID, MANUAL_SCENE_NODE_ID}

PANO_TO_3D_CUBEMAP = "3D from RGB-D Cubemap"
PANO_TO_3D_EQUIRECTANGULAR = "3D from RGB-D Equirectangular"
PANO_TO_3D_MODES = (PANO_TO_3D_CUBEMAP, PANO_TO_3D_EQUIRECTANGULAR)

SCENE_MANIFEST_SCHEMA = "modly.scene-manifest.v1"
SCENE_MANIFEST_FILE_NAME = "scene-manifest.json"


class SceneGenerationError(RuntimeError):
    """Scene output failed after preserving diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        stats: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.stats = dict(stats) if isinstance(stats, Mapping) else None
        self.diagnostics = dict(diagnostics or {})


PROMPT_FIELDS = (
    "prompt_front",
    "prompt_right",
    "prompt_back",
    "prompt_left",
    "prompt_top",
    "prompt_bottom",
)


def _log(message: str) -> None:
    print(f"[dreamcube] {message}", file=sys.stderr, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        value_type = type(value)
        return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _json_safe(value: Any) -> Any:
    """Recursively convert diagnostics to values accepted by strict JSON."""

    seen: set[int] = set()

    def normalise(current: Any, depth: int) -> Any:
        if depth > 64:
            return "<maximum-json-depth>"

        if current is None or isinstance(current, (str, bool, int)):
            return current
        if isinstance(current, float):
            if math.isfinite(current):
                return current
            if math.isnan(current):
                return "NaN"
            return "Infinity" if current > 0 else "-Infinity"
        if isinstance(current, datetime):
            return current.isoformat()
        if isinstance(current, os.PathLike):
            try:
                return str(os.fspath(current))
            except Exception:
                return _safe_repr(current)
        if isinstance(current, (bytes, bytearray, memoryview)):
            try:
                return bytes(current).decode("utf-8", errors="replace")
            except Exception:
                return _safe_repr(current)

        current_id = id(current)
        if current_id in seen:
            return "<recursive-reference>"

        value_type = type(current)
        module_name = value_type.__module__
        if module_name == "numpy" or module_name.startswith("numpy."):
            seen.add(current_id)
            try:
                if hasattr(current, "tolist"):
                    converted = current.tolist()
                elif hasattr(current, "item"):
                    converted = current.item()
                else:
                    return _safe_repr(current)
                if converted is current:
                    return _safe_repr(current)
                return normalise(converted, depth + 1)
            except Exception:
                return _safe_repr(current)
            finally:
                seen.discard(current_id)

        if module_name == "torch" or module_name.startswith("torch."):
            seen.add(current_id)
            try:
                converted = current.detach().cpu().tolist()
                if converted is current:
                    return _safe_repr(current)
                return normalise(converted, depth + 1)
            except Exception:
                return _safe_repr(current)
            finally:
                seen.discard(current_id)

        if isinstance(current, MappingABC):
            seen.add(current_id)
            try:
                result: dict[str, Any] = {}
                for key, item in current.items():
                    safe_key = normalise(key, depth + 1)
                    if isinstance(safe_key, str):
                        key_text = safe_key
                    elif safe_key is None or isinstance(safe_key, (bool, int, float)):
                        key_text = str(safe_key)
                    else:
                        key_text = _safe_repr(safe_key)

                    unique_key = key_text
                    suffix = 2
                    while unique_key in result:
                        unique_key = f"{key_text} [{suffix}]"
                        suffix += 1
                    result[unique_key] = normalise(item, depth + 1)
                return result
            except Exception:
                return _safe_repr(current)
            finally:
                seen.discard(current_id)

        if isinstance(current, SequenceABC):
            seen.add(current_id)
            try:
                return [normalise(item, depth + 1) for item in current]
            except Exception:
                return _safe_repr(current)
            finally:
                seen.discard(current_id)

        if isinstance(current, SetABC):
            seen.add(current_id)
            try:
                values = [normalise(item, depth + 1) for item in current]
                return sorted(values, key=_safe_repr)
            except Exception:
                return _safe_repr(current)
            finally:
                seen.discard(current_id)

        return _safe_repr(current)

    return normalise(value, 0)


def _initial_view_payload() -> dict[str, list[int]]:
    return {
        "position": [0, 0, 0],
        "target": [0, 0, 1],
        "up": [0, 1, 0],
    }


def _coordinate_frame_payload() -> dict[str, Any]:
    return {
        "handedness": "right-handed",
        "units": "meters",
        "origin": "camera",
        "axes": {
            "x": "left",
            "y": "up",
            "z": "forward",
        },
    }


def _presentation_payload() -> dict[str, Any]:
    return {
        "type": "navigable-panorama",
        "viewpoint": "interior",
        "initial_view": _initial_view_payload(),
    }


def _load_manifest() -> dict[str, Any]:
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _schema_for_node(node_id: str) -> list[dict[str, Any]]:
    manifest = _load_manifest()
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list):
        return []
    for node in nodes:
        if not isinstance(node, dict) or node.get("id") != node_id:
            continue
        schema = node.get("params_schema")
        return list(schema) if isinstance(schema, list) else []
    return []


def _schema_defaults(node_id: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for item in _schema_for_node(node_id):
        if isinstance(item, dict) and isinstance(item.get("id"), str) and "default" in item:
            defaults[item["id"]] = item["default"]
    return defaults


def _safe_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not an integer")
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a float")
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _safe_choice(value: Any, choices: tuple[str, ...] | list[str], default: str) -> str:
    text = str(value).strip() if value is not None else ""
    if text in choices:
        return text
    lowered = text.lower()
    for choice in choices:
        if choice.lower() == lowered:
            return choice
    return default


def _param(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: Any) -> Any:
    value = params.get(key)
    if value is not None:
        return value
    return defaults.get(key, fallback)


def _max_size(value: Any, default: int) -> int | None:
    parsed = _safe_int(value, default, minimum=0)
    return None if parsed <= 0 else parsed


def _normalise_node_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in NODE_IDS:
        return text
    lowered = text.lower()
    if lowered in NODE_IDS:
        return lowered
    if lowered.endswith("/" + SCENE_NODE_ID) or lowered.endswith(":" + SCENE_NODE_ID):
        return SCENE_NODE_ID
    if lowered.endswith("/" + PANORAMA_NODE_ID) or lowered.endswith(":" + PANORAMA_NODE_ID):
        return PANORAMA_NODE_ID
    if "manual-cubemap" in lowered or "rgbd-cubemap" in lowered:
        return MANUAL_SCENE_NODE_ID
    if "scene" in lowered or "mesh" in lowered:
        return SCENE_NODE_ID
    if "panorama" in lowered or "pano" in lowered or "equirect" in lowered:
        return PANORAMA_NODE_ID
    return None


def _progress(progress_cb: Callable[..., Any] | None, pct: int, phase: str) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(pct, phase)
    except TypeError:
        progress_cb({"progress": pct, "phase": phase})


def _raise_if_cancelled(cancel_event: Any | None) -> None:
    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
        raise GenerationCancelled()


def _expand_path_text(value: Any) -> Path:
    return Path(os.path.expandvars(str(value).strip())).expanduser()


def _resolve_depth_image_path(value: Any, outputs_dir: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("DreamCube requires params.depth_image_path pointing to the front depth image.")

    raw_path = _expand_path_text(value)
    if raw_path.is_absolute():
        if raw_path.is_file():
            return raw_path.resolve()
        raise FileNotFoundError(f"DreamCube depth image path does not exist: {raw_path}")

    candidates: list[Path] = [
        EXTENSION_DIR / raw_path,
        Path(outputs_dir) / raw_path,
    ]
    workspace_dir = os.environ.get("WORKSPACE_DIR")
    if workspace_dir and workspace_dir.strip():
        candidates.append(_expand_path_text(workspace_dir) / raw_path)

    candidates.append(Path.cwd() / raw_path)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "DreamCube depth image path does not exist. Checked: " + "; ".join(checked)
    )


def _has_path_value(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _resolve_prompts(params: Mapping[str, Any]) -> list[str]:
    prompts: list[str] = []
    missing: list[str] = []
    for field in PROMPT_FIELDS:
        value = params.get(field)
        if value is None or not str(value).strip():
            missing.append(field)
        else:
            prompts.append(str(value).strip())
    if missing:
        raise ValueError(
            "DreamCube requires six directional prompts before model loading; missing: "
            + ", ".join(missing)
        )
    return prompts


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _readiness_result(
    *,
    ok: bool,
    machine_code: str,
    label_hint: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "machine_code": machine_code,
        "label_hint": label_hint,
        "reason": reason,
        "checked_at": _utc_now(),
    }
    if details:
        payload["details"] = details
    return payload


def _center_square_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    return (left, top, left + side, top + side)


def _size_payload(size: tuple[int, int]) -> list[int]:
    return [int(size[0]), int(size[1])]


def _crop_payload(crop_box: tuple[int, int, int, int]) -> list[int]:
    return [int(value) for value in crop_box]


def _single_channel_depth_image(image: Any) -> Any:
    if len(image.getbands()) == 1 and image.mode != "P":
        return image
    return image.convert("L")


class DreamCubeGenerator(BaseGenerator):
    MODEL_ID = "dreamcube"
    DISPLAY_NAME = "DreamCube RGB-D Panorama"
    VRAM_GB = 16

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        """Return the default panorama schema when Modly has not injected one."""

        return _schema_for_node(PANORAMA_NODE_ID)

    @classmethod
    def capability_params_schema(cls, node_id: str) -> list[dict[str, Any]]:
        return _schema_for_node(node_id)

    def __init__(self, model_dir: Path | str, outputs_dir: Path | str) -> None:
        super().__init__(model_dir, outputs_dir)
        self.model_dir = Path(model_dir)
        self.outputs_dir = Path(outputs_dir)
        self.download_check = self.download_check or DOWNLOAD_CHECK
        self.hf_repo = self.hf_repo or HF_REPO
        self._pipe: Any | None = None
        self._app: Any | None = None
        self._auto_depth_processor: Any | None = None
        self._auto_depth_model: Any | None = None
        self._auto_depth_variant: str | None = None

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def is_downloaded(self) -> bool:
        return (self.model_dir / DOWNLOAD_CHECK).exists()

    def readiness_status(self) -> dict[str, Any]:
        status = _read_json_file(SETUP_STATUS_PATH)
        upstream_exists = UPSTREAM_APP_PATH.is_file()
        model_index = self.model_dir / DOWNLOAD_CHECK
        model_exists = model_index.is_file()
        details: dict[str, Any] = {
            "setup_status_path": str(SETUP_STATUS_PATH),
            "setup_status": status.get("status") if isinstance(status, dict) else "missing",
            "upstream_source": str(UPSTREAM_DIR),
            "upstream_app": str(UPSTREAM_APP_PATH),
            "upstream_source_exists": upstream_exists,
            "model_dir": str(self.model_dir),
            "download_check": DOWNLOAD_CHECK,
            "model_index": str(model_index),
            "model_index_exists": model_exists,
            "venv_dir": str(EXTENSION_DIR / "venv"),
            "readiness_source": "generator.py",
        }

        if isinstance(status, dict):
            if status.get("upstream_commit"):
                details["upstream_commit"] = status.get("upstream_commit")
            if status.get("torch_lane"):
                details["torch_lane"] = status.get("torch_lane")
            blockers = status.get("blockers")
            if isinstance(blockers, list):
                details["setup_blockers"] = blockers

        if status is None:
            return _readiness_result(
                ok=False,
                machine_code="setup_status_missing",
                label_hint="Setup required",
                reason="DreamCube setup has not produced .modly/setup/setup-status.json. Run extension setup first.",
                details=details,
            )

        setup_state = str(status.get("status", "unknown"))
        if setup_state != "ready":
            blockers = status.get("blockers")
            reason = f"DreamCube setup status is {setup_state!r}."
            if isinstance(blockers, list) and blockers:
                first = blockers[0]
                if isinstance(first, dict) and isinstance(first.get("message"), str):
                    reason = first["message"]
            return _readiness_result(
                ok=False,
                machine_code=f"setup_{setup_state}",
                label_hint="Setup incomplete",
                reason=reason,
                details=details,
            )

        if not upstream_exists:
            return _readiness_result(
                ok=False,
                machine_code="upstream_source_missing",
                label_hint="Repair setup",
                reason=f"DreamCube upstream app.py is missing at {UPSTREAM_APP_PATH}. Rerun extension setup.",
                details=details,
            )

        if not model_exists:
            return _readiness_result(
                ok=False,
                machine_code="model_weights_missing",
                label_hint="Download weights",
                reason=f"DreamCube model snapshot is missing {DOWNLOAD_CHECK} at {model_index}.",
                details=details,
            )

        return _readiness_result(
            ok=True,
            machine_code="ready",
            label_hint="Ready",
            reason="DreamCube setup, upstream source, and model snapshot are ready.",
            details=details,
        )

    def _validate_runtime_files(self) -> None:
        if not UPSTREAM_APP_PATH.is_file():
            raise RuntimeError(
                f"DreamCube upstream source is missing at {UPSTREAM_APP_PATH}. Run extension setup first."
            )
        model_index = self.model_dir / DOWNLOAD_CHECK
        if not model_index.is_file():
            raise RuntimeError(
                f"DreamCube model snapshot is missing {DOWNLOAD_CHECK} at {model_index}. "
                "Download or set up the model before generation."
            )

    def _import_upstream_app(self) -> Any:
        self._validate_runtime_files()
        upstream_text = str(UPSTREAM_DIR)
        if upstream_text not in sys.path:
            sys.path.insert(0, upstream_text)

        module_name = "_modly_dreamcube_upstream_app"
        cached = sys.modules.get(module_name)
        if cached is not None and Path(getattr(cached, "__file__", "")).resolve() == UPSTREAM_APP_PATH.resolve():
            return cached

        spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_APP_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load DreamCube upstream app module from {UPSTREAM_APP_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)

        required = (
            "build_pipeline",
            "inference",
            "postprocess_rgb",
            "postprocess_depth",
            "z_distance_to_depth",
            "depth_to_z_distance",
            "convert_rgbd_equi_to_3dgs",
            "convert_rgbd_cube_to_3dgs",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"DreamCube upstream app.py is missing required functions: {', '.join(missing)}")
        return module

    def load(self) -> None:
        if self._pipe is not None:
            return
        self._validate_runtime_files()
        _log(f"Loading DreamCube pipeline from {self.model_dir}")
        app = self._import_upstream_app()
        with contextlib.redirect_stdout(sys.stderr):
            pipe = app.build_pipeline(str(self.model_dir), local_files_only=True)
        self._app = app
        self._pipe = pipe
        self._model = pipe
        _log("DreamCube pipeline loaded")

    def unload(self) -> None:
        self._pipe = None
        self._app = None
        self._model = None
        self._auto_depth_processor = None
        self._auto_depth_model = None
        self._auto_depth_variant = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _resolve_node_id(self, params: Mapping[str, Any]) -> str:
        candidates = (
            getattr(self, "node_id", None),
            params.get("node_id"),
            params.get("nodeId"),
            params.get("operation"),
            params.get("capability_id"),
            getattr(self, "MODEL_ID", None),
            self.model_dir.name if isinstance(self.model_dir, Path) else None,
        )
        for candidate in candidates:
            node_id = _normalise_node_id(candidate)
            if node_id is not None:
                return node_id

        output_format = str(params.get("output_format", "equirect_rgb_png")).strip().lower()
        capability = str(params.get("capability_id", "")).strip().lower()
        if capability == "rgbd-cubemap-to-scene":
            return MANUAL_SCENE_NODE_ID
        if output_format in {"glb", "obj"}:
            return SCENE_NODE_ID
        return PANORAMA_NODE_ID

    def _validate_request(
        self,
        image_bytes: bytes | bytearray | memoryview,
        params: Mapping[str, Any],
    ) -> tuple[Path | None, list[str]]:
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or len(image_bytes) == 0:
            raise ValueError("DreamCube requires non-empty image_bytes for the front RGB image.")
        depth_mode = _safe_choice(params.get("depth_mode"), ("auto", "manual"), "auto")
        supplied_depth = params.get("depth_image_path")
        if depth_mode == "manual":
            depth_path = _resolve_depth_image_path(supplied_depth, self.outputs_dir)
        elif _has_path_value(supplied_depth):
            depth_path = _resolve_depth_image_path(supplied_depth, self.outputs_dir)
        else:
            depth_path = None
        prompts = _resolve_prompts(params)
        return depth_path, prompts

    def _save_input_rgb(
        self,
        image_bytes: bytes | bytearray | memoryview,
        run_dir: Path,
    ) -> tuple[Path, dict[str, Any]]:
        from PIL import Image

        image_path = run_dir / "input_front_rgb.png"
        with Image.open(io.BytesIO(bytes(image_bytes))) as image:
            original_size = (image.width, image.height)
            crop_box = _center_square_crop_box(image.width, image.height)
            canonical = image.convert("RGB")
            if crop_box != (0, 0, image.width, image.height):
                canonical = canonical.crop(crop_box)
            canonical.save(image_path)

        metadata = {
            "path": image_path.name,
            "original_size": _size_payload(original_size),
            "canonical_size": _size_payload((crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])),
            "crop_box": _crop_payload(crop_box),
        }
        return image_path, metadata

    def _save_manual_depth_image(
        self,
        depth_path: Path,
        run_dir: Path,
        rgb_original_size: tuple[int, int],
        rgb_crop_box: tuple[int, int, int, int],
    ) -> tuple[Path, dict[str, Any]]:
        from PIL import Image

        canonical_depth_path = run_dir / "input_front_depth_manual.png"
        with Image.open(depth_path) as depth_image:
            original_size = (depth_image.width, depth_image.height)
            if original_size == rgb_original_size:
                crop_box = rgb_crop_box
                crop_source = "rgb"
            else:
                crop_box = _center_square_crop_box(depth_image.width, depth_image.height)
                crop_source = "depth"

            canonical = depth_image.crop(crop_box)
            canonical = _single_channel_depth_image(canonical)
            canonical.save(canonical_depth_path)
            canonical_size = (canonical.width, canonical.height)
            canonical_mode = canonical.mode

        metadata = {
            "mode": "manual",
            "source": str(depth_path),
            "path": canonical_depth_path.name,
            "original_size": _size_payload(original_size),
            "canonical_size": _size_payload(canonical_size),
            "crop_box": _crop_payload(crop_box),
            "crop_source": crop_source,
            "image_mode": canonical_mode,
        }
        return canonical_depth_path, metadata

    def _write_run_metadata(self, run_dir: Path, metadata: Mapping[str, Any]) -> Path:
        metadata_path = run_dir / "run_metadata.json"
        safe_metadata = _json_safe(metadata)
        if not isinstance(safe_metadata, dict):
            safe_metadata = {"metadata": safe_metadata}
        metadata_path.write_text(
            json.dumps(
                safe_metadata,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return metadata_path

    def _record_failure_metadata(
        self,
        run_dir: Path,
        failure: Mapping[str, Any],
        *,
        section: str = "mesh_export",
    ) -> None:
        """Persist diagnostics without ever replacing the generation exception."""

        try:
            existing_metadata = _read_json_file(run_dir / "run_metadata.json") or {}
            existing_metadata[section] = failure
            self._write_run_metadata(run_dir, existing_metadata)
        except Exception as metadata_error:
            try:
                _log(
                    "Failed to persist run metadata while handling a generation error; "
                    f"the original error will be re-raised. metadata_error={_safe_repr(metadata_error)}"
                )
            except Exception:
                pass

    @staticmethod
    def _scene_file_diagnostics(obj_path: Path, glb_path: Path) -> dict[str, Any]:
        splat_path = obj_path.parent / "output_3dgs.splat"

        def file_status(path: Path) -> tuple[bool, int]:
            try:
                if not path.is_file():
                    return False, 0
                return True, int(path.stat().st_size)
            except OSError:
                return False, 0

        obj_exists, obj_size = file_status(obj_path)
        glb_exists, glb_size = file_status(glb_path)
        splat_exists, splat_size = file_status(splat_path)
        return {
            "obj_exists": obj_exists,
            "obj_size_bytes": obj_size,
            "glb_exists": glb_exists,
            "glb_size_bytes": glb_size,
            "splat_exists": splat_exists,
            "splat_size_bytes": splat_size,
        }

    def _official_workspace_root(self) -> Path:
        workspace_value = os.environ.get("WORKSPACE_DIR")
        if workspace_value is None or not workspace_value.strip():
            # Modly constructs generators with the workspace root, then generation_jobs
            # replaces outputs_dir with WORKSPACE_DIR / collection before generate().
            raise SceneGenerationError(
                "DreamCube WORKSPACE_DIR is required for scene output because Modly assigns "
                "outputs_dir to a collection directory before generation; refusing to "
                "guess the workspace root.",
                stage="workspace_path",
                diagnostics={
                    "workspace_root_source": "missing",
                    "outputs_dir": self.outputs_dir,
                },
            )

        workspace_path = _expand_path_text(workspace_value)
        if not workspace_path.is_absolute():
            raise SceneGenerationError(
                "DreamCube WORKSPACE_DIR must be an absolute path; refusing to resolve "
                "a workspace root relative to the extension process.",
                stage="workspace_path",
                diagnostics={
                    "workspace_root_source": "WORKSPACE_DIR",
                    "workspace_dir": workspace_path,
                    "workspace_dir_absolute": False,
                },
            )

        try:
            root = workspace_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SceneGenerationError(
                "DreamCube WORKSPACE_DIR could not be resolved to an existing workspace "
                "directory.",
                stage="workspace_path",
                diagnostics={
                    "workspace_root_source": "WORKSPACE_DIR",
                    "workspace_dir": workspace_path,
                    "workspace_dir_valid": False,
                    "resolution_error": exc,
                },
            ) from exc

        if not root.is_dir():
            raise SceneGenerationError(
                "DreamCube WORKSPACE_DIR must reference an existing directory.",
                stage="workspace_path",
                diagnostics={
                    "workspace_root_source": "WORKSPACE_DIR",
                    "workspace_dir": root,
                    "workspace_dir_valid": False,
                },
            )

        outputs_root = self.outputs_dir.resolve()
        if outputs_root != root and root not in outputs_root.parents:
            raise SceneGenerationError(
                "DreamCube outputs directory is outside the official workspace root; "
                "refusing to emit a scene manifest.",
                stage="workspace_path",
                diagnostics={
                    "workspace_root_source": "WORKSPACE_DIR",
                    "workspace_dir": root,
                    "outputs_dir": outputs_root,
                    "outputs_contained": False,
                },
            )
        return root

    def _workspace_relative_path(self, asset_path: Path) -> str:
        workspace_root = self._official_workspace_root()
        resolved_asset = Path(asset_path).resolve()
        try:
            relative = resolved_asset.relative_to(workspace_root)
        except ValueError as exc:
            raise SceneGenerationError(
                "DreamCube scene asset is outside the official workspace root; "
                "refusing to write an unsafe workspacePath.",
                stage="workspace_path",
                diagnostics={
                    "workspace_root_source": (
                        "WORKSPACE_DIR"
                        if os.environ.get("WORKSPACE_DIR", "").strip()
                        else "outputs_dir"
                    ),
                    "asset_contained": False,
                },
            ) from exc

        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise SceneGenerationError(
                "DreamCube scene asset did not resolve to a safe workspace-relative path.",
                stage="workspace_path",
                diagnostics={"asset_contained": False},
            )
        return relative.as_posix()

    def _write_scene_manifest(self, run_dir: Path, glb_path: Path) -> Path:
        run_dir = Path(run_dir)
        glb_path = Path(glb_path)
        diagnostics = self._scene_file_diagnostics(run_dir / "output_mesh.obj", glb_path)
        if not diagnostics["glb_exists"] or diagnostics["glb_size_bytes"] <= 0:
            raise SceneGenerationError(
                "DreamCube required GLB is missing or empty; scene manifest was not written.",
                stage="glb_validation",
                diagnostics=diagnostics,
            )

        workspace_path = self._workspace_relative_path(glb_path)
        manifest = {
            "schema": SCENE_MANIFEST_SCHEMA,
            "sceneRoot": ".",
            "generator": "modly.worlds",
            "version": 1,
            "createdAt": _utc_now(),
            "assets": [
                {
                    "id": "dreamcube-base-scene",
                    "name": glb_path.name,
                    "role": "base-scene",
                    "workspacePath": workspace_path,
                    "kind": "glb",
                    "visible": True,
                    "transform": {
                        "position": [0, 0, 0],
                        "rotation": [0, 0, 0],
                        "scale": [1, 1, 1],
                    },
                }
            ],
            "initialView": _initial_view_payload(),
        }
        manifest_path = run_dir / SCENE_MANIFEST_FILE_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    @staticmethod
    def _prepare_3dgs_inputs(
        *,
        rgb: Any,
        distance: Any,
        rays: Any,
        max_size: int | None,
        cubemap: bool,
    ) -> tuple[Any, Any, Any]:
        if max_size is None:
            return rgb, distance, rays

        height = int(distance.shape[-2])
        width = int(distance.shape[-1])
        largest = max(height, width)
        if largest <= int(max_size):
            return rgb, distance, rays

        import torch.nn.functional as functional

        scale = int(max_size) / largest
        if cubemap:
            rgb_resized = functional.interpolate(
                (rgb / 255.0).permute(0, 3, 1, 2),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ).permute(0, 2, 3, 1) * 255.0
            distance_resized = functional.interpolate(
                distance.unsqueeze(1),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ).squeeze(1)
            rays_resized = functional.interpolate(
                rays.permute(0, 3, 1, 2),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ).permute(0, 2, 3, 1)
        else:
            rgb_resized = functional.interpolate(
                (rgb / 255.0).permute(2, 0, 1).unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ).squeeze(0).permute(1, 2, 0) * 255.0
            distance_resized = functional.interpolate(
                distance.unsqueeze(0).unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ).squeeze(0).squeeze(0)
            rays_resized = functional.interpolate(
                rays.permute(2, 0, 1).unsqueeze(0),
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            ).squeeze(0).permute(1, 2, 0)

        rays_resized = rays_resized / (
            rays_resized.norm(dim=-1, keepdim=True) + 1e-8
        )
        return (
            rgb_resized.contiguous(),
            distance_resized.contiguous(),
            rays_resized.contiguous(),
        )

    def _convert_obj_to_glb(
        self,
        obj_path: Path,
        glb_path: Path,
        mesh_stats: Mapping[str, Any],
    ) -> Path:
        try:
            glb_path.unlink(missing_ok=True)
            import trimesh

            mesh = trimesh.load(
                str(obj_path),
                force="mesh",
                process=False,
                maintain_order=True,
            )
            _ = mesh.vertex_normals
            mesh.visual.material = trimesh.visual.material.PBRMaterial(
                doubleSided={
                    "cubemap": False,
                    "equirectangular": True,
                }[mesh_stats["mode"]]
            )
            mesh.export(str(glb_path))
        except Exception as exc:
            diagnostics = self._scene_file_diagnostics(obj_path, glb_path)
            _log(
                "Required GLB export failed "
                f"stage=glb_conversion obj_exists={diagnostics['obj_exists']} "
                f"obj_bytes={diagnostics['obj_size_bytes']} "
                f"glb_exists={diagnostics['glb_exists']} "
                f"glb_bytes={diagnostics['glb_size_bytes']} error={exc}"
            )
            raise SceneGenerationError(
                f"DreamCube OBJ-to-GLB conversion failed: {exc}",
                stage="glb_conversion",
                stats=mesh_stats,
                diagnostics=diagnostics,
            ) from exc

        diagnostics = self._scene_file_diagnostics(obj_path, glb_path)
        if not diagnostics["glb_exists"] or diagnostics["glb_size_bytes"] <= 0:
            _log(
                "Required GLB export failed "
                f"stage=glb_validation obj_exists={diagnostics['obj_exists']} "
                f"obj_bytes={diagnostics['obj_size_bytes']} "
                f"glb_exists={diagnostics['glb_exists']} "
                f"glb_bytes={diagnostics['glb_size_bytes']}"
            )
            raise SceneGenerationError(
                "DreamCube required GLB is missing or empty after OBJ-to-GLB conversion.",
                stage="glb_validation",
                stats=mesh_stats,
                diagnostics=diagnostics,
            )
        return glb_path

    def _build_seed_kwargs(self, seed: int) -> dict[str, Any]:
        if seed < 0:
            return {}
        if self._pipe is None:
            raise RuntimeError("DreamCube pipeline is not loaded; cannot build seeded generator.")
        import torch

        device = getattr(self._pipe, "device", None)
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
        except Exception as exc:
            raise RuntimeError(f"Unable to create torch.Generator on DreamCube device {device!r}: {exc}") from exc
        return {"generator": generator}

    def _load_auto_depth(self, variant: str) -> tuple[Any, Any]:
        variant = _safe_choice(variant, tuple(AUTO_DEPTH_VARIANTS), AUTO_DEPTH_DEFAULT_VARIANT)
        if (
            self._auto_depth_processor is not None
            and self._auto_depth_model is not None
            and self._auto_depth_variant == variant
        ):
            return self._auto_depth_processor, self._auto_depth_model

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            import torch
        except Exception as exc:  # pragma: no cover - exercised when runtime deps are absent
            raise RuntimeError(
                "DreamCube internal auto-depth requires transformers and torch installed in the DreamCube venv."
            ) from exc

        repo_id = AUTO_DEPTH_VARIANTS[variant]["repo_id"]
        AUTO_DEPTH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _log(f"Loading internal auto-depth model {repo_id} into {AUTO_DEPTH_CACHE_DIR}")
        processor = AutoImageProcessor.from_pretrained(repo_id, cache_dir=str(AUTO_DEPTH_CACHE_DIR))
        model = AutoModelForDepthEstimation.from_pretrained(repo_id, cache_dir=str(AUTO_DEPTH_CACHE_DIR))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        self._auto_depth_processor = processor
        self._auto_depth_model = model
        self._auto_depth_variant = variant
        return processor, model

    def _save_auto_depth_image(self, image_path: Path, run_dir: Path, variant: str) -> Path:
        """Estimate relative monocular depth and save DreamCube-compatible uint16 depth.

        Depth Anything V2 predicts relative inverse-depth-like values, not metric distance.
        DreamCube examples use positive 16-bit depth maps, so this wrapper robustly
        normalizes the prediction and maps it to an estimated 1000..5000 mm range.
        Higher prediction values are treated as nearer surfaces, therefore they map
        to smaller distance values. This is aligned to the already-saved
        input_front_rgb.png, not the user's original source path.
        """

        import numpy as np
        import torch
        from PIL import Image

        processor, model = self._load_auto_depth(variant)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            target_size = (image.height, image.width)
            inputs = processor(images=image, return_tensors="pt")

        device = next(model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            prediction = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.detach().float().cpu().numpy()
        finite = np.isfinite(depth)
        if not finite.any():
            normalized = np.zeros_like(depth, dtype=np.float32)
        else:
            valid = depth[finite]
            low, high = np.percentile(valid, [2.0, 98.0])
            if float(high - low) < 1e-6:
                normalized = np.full_like(depth, 0.5, dtype=np.float32)
            else:
                normalized = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)
                normalized[~finite] = 0.5

        depth_mm = np.rint(5000.0 - (normalized * 4000.0)).clip(1000, 5000).astype(np.uint16)
        depth_path = run_dir / "input_front_depth_auto.png"
        Image.fromarray(np.asarray(depth_mm, dtype=np.uint16)).save(depth_path)
        return depth_path


    def _run_manual_cubemap_inference(
        self,
        *,
        inputs: dreamcube_manual_cubemap.ManualCubemapInputs,
        prompts: list[str],
        num_inference_steps: int,
        guidance_scale: float,
        normalize_scale: float,
        max_cube_size: int,
        seed_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._pipe is None or self._app is None:
            raise RuntimeError("DreamCube pipeline is not loaded.")
        import numpy as np
        import torch

        target_size = min(512, max(256, int(max_cube_size)))
        rgbs, depths_mm = dreamcube_manual_cubemap.resize_for_inference(inputs, target_size)
        rgb_np = np.stack([np.asarray(image, dtype=np.float32) / 127.5 - 1.0 for image in rgbs], axis=0)
        radial_np = np.stack([depth.astype(np.float32) for depth in depths_mm], axis=0)[..., None]
        z_np = self._app.depth_to_z_distance(radial_np, fov_x=90.0, fov_y=90.0).astype(np.float32)
        device = getattr(self._pipe, "device", "cpu")
        cube_rgbs = torch.from_numpy(rgb_np).to(device).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        cube_depths = torch.from_numpy(z_np).to(device).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        cube_masks = torch.zeros_like(cube_depths, dtype=torch.bool)
        cube_prompts = dreamcube_manual_cubemap.prefixed_prompts(prompts)

        cuda_available = torch.cuda.is_available()
        autocast_context = torch.amp.autocast("cuda") if cuda_available else contextlib.nullcontext()
        try:
            with torch.inference_mode():
                with autocast_context:
                    with contextlib.redirect_stdout(sys.stderr):
                        prediction = self._pipe(
                            cube_rgbs=cube_rgbs,
                            cube_depths=cube_depths,
                            cube_masks=cube_masks,
                            prompt=cube_prompts,
                            height=target_size,
                            width=target_size,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            output_type="np",
                            normalize_scale=normalize_scale,
                            **seed_kwargs,
                        )
        except TypeError as exc:
            if seed_kwargs:
                raise RuntimeError(
                    "DreamCube upstream pipeline rejected the seeded torch.Generator. "
                    "Retry with seed=-1 or update the upstream pipeline to accept generator=."
                ) from exc
            raise

        images_pred = prediction.images
        depths_pred = prediction.depths
        if images_pred is None or depths_pred is None:
            raise RuntimeError("DreamCube manual cubemap pipeline returned incomplete predictions.")
        if hasattr(images_pred, "detach"):
            images_pred = images_pred.detach().cpu().numpy()
        if hasattr(depths_pred, "detach"):
            depths_pred = depths_pred.detach().cpu().numpy()
        images_pred = (images_pred * 255).round().astype("uint8")
        images_pred = images_pred.reshape((1, 6, target_size, target_size, images_pred.shape[-1]))
        depths_pred = depths_pred.reshape((1, 6, target_size, target_size, depths_pred.shape[-1]))
        diagnostics = {
            "conditioning_mode": "manual-rgbd-cubemap",
            "inference_size": [target_size, target_size],
            "cube_rgbs_shape": list(cube_rgbs.shape),
            "cube_depths_shape": list(cube_depths.shape),
            "cube_masks_shape": list(cube_masks.shape),
            "prompt_prefixes": list(dreamcube_manual_cubemap.OFFICIAL_PROMPT_PREFIXES),
        }
        return {"images": images_pred, "depths": depths_pred, "normals": None}, diagnostics

    def _save_output_faces(self, *, outputs: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
        from PIL import Image
        import numpy as np

        images = outputs.get("images_pred")
        depths = outputs.get("depths_distance")
        if images is None or depths is None:
            return {"status": "skipped", "reason": "missing cubemap predictions"}
        output_dir = run_dir / "output_faces"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for index, face in enumerate(dreamcube_manual_cubemap.FACE_ORDER):
            rgb_arr = np.asarray(images[0, index], dtype=np.uint8)
            depth_arr = np.rint(np.asarray(depths[0, index, ..., 0], dtype=np.float32)).clip(0, 65535).astype(np.uint16)
            rgb_path = output_dir / f"{face}_rgb.png"
            depth_path = output_dir / f"{face}_depth_mm.png"
            Image.fromarray(rgb_arr).save(rgb_path)
            Image.fromarray(np.asarray(depth_arr, dtype=np.uint16)).save(depth_path)
            saved.extend([str(rgb_path.relative_to(run_dir)), str(depth_path.relative_to(run_dir))])
        return {"status": "saved", "files": saved}

    def _generate_manual_cubemap(
        self,
        image_bytes: bytes,
        safe_params: dict[str, Any],
        defaults: Mapping[str, Any],
        progress_cb: Callable[..., Any] | None,
        cancel_event: Any | None,
    ) -> Path:
        _raise_if_cancelled(cancel_event)
        prompts = _resolve_prompts(safe_params)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.outputs_dir / f"dreamcube-{timestamp}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        num_inference_steps = _safe_int(_param(safe_params, defaults, "num_inference_steps", 50), 50, minimum=1, maximum=100)
        guidance_scale = _safe_float(_param(safe_params, defaults, "guidance_scale", 7.5), 7.5, minimum=0.0, maximum=20.0)
        normalize_scale = _safe_float(_param(safe_params, defaults, "normalize_scale", 0.6), 0.6, minimum=0.05, maximum=5.0)
        max_cube_size = _safe_int(_param(safe_params, defaults, "max_cube_size", 512), 512, minimum=256, maximum=512)
        mesh_depth_jump_threshold = _safe_float(_param(safe_params, defaults, "mesh_depth_jump_threshold", dreamcube_mesh.DEFAULT_DEPTH_JUMP_THRESHOLD), dreamcube_mesh.DEFAULT_DEPTH_JUMP_THRESHOLD, minimum=0.0, maximum=5.0)
        mesh_footprint_ratio_threshold = _safe_float(_param(safe_params, defaults, "mesh_footprint_ratio_threshold", dreamcube_mesh.DEFAULT_FOOTPRINT_RATIO_THRESHOLD), dreamcube_mesh.DEFAULT_FOOTPRINT_RATIO_THRESHOLD, minimum=0.0, maximum=100.0)
        mesh_aspect_ratio_threshold = _safe_float(_param(safe_params, defaults, "mesh_aspect_ratio_threshold", dreamcube_mesh.DEFAULT_ASPECT_RATIO_THRESHOLD), dreamcube_mesh.DEFAULT_ASPECT_RATIO_THRESHOLD, minimum=0.0, maximum=100.0)
        seed = _safe_int(_param(safe_params, defaults, "seed", -1), -1, minimum=-1, maximum=2_147_483_647)

        try:
            inputs = dreamcube_manual_cubemap.load_manual_cubemap_inputs(
                front_rgb_bytes=image_bytes,
                params=safe_params,
                outputs_dir=self.outputs_dir,
            )
            dreamcube_manual_cubemap.save_input_faces(run_dir, inputs)
            run_metadata: dict[str, Any] = {
                "created_at": _utc_now(),
                "node_id": MANUAL_SCENE_NODE_ID,
                "output_format": "glb",
                "conditioning_mode": "manual-rgbd-cubemap",
                "face_order": list(dreamcube_manual_cubemap.FACE_ORDER),
                "face_axes": dict(dreamcube_manual_cubemap.FACE_AXES),
                "input_faces": inputs.source_stats,
                "seam_metrics_preflight": inputs.seam_metrics,
                "prompts": dict(zip(PROMPT_FIELDS, prompts)),
                "reconstruction": {
                    "mode": PANO_TO_3D_CUBEMAP,
                    "max_equi_size": None,
                    "max_cube_size": max_cube_size,
                    "mesh_depth_jump_threshold": mesh_depth_jump_threshold,
                    "mesh_footprint_ratio_threshold": mesh_footprint_ratio_threshold,
                    "mesh_aspect_ratio_threshold": mesh_aspect_ratio_threshold,
                },
                "coordinate_frame": _coordinate_frame_payload(),
                "presentation": _presentation_payload(),
            }
            self._write_run_metadata(run_dir, run_metadata)

            _raise_if_cancelled(cancel_event)
            _progress(progress_cb, 10, "load")
            self.load()
            seed_kwargs = self._build_seed_kwargs(seed)

            _raise_if_cancelled(cancel_event)
            _progress(progress_cb, 25, "infer")
            predictions, manual_diagnostics = self._run_manual_cubemap_inference(
                inputs=inputs,
                prompts=prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                normalize_scale=normalize_scale,
                max_cube_size=max_cube_size,
                seed_kwargs=seed_kwargs,
            )

            _raise_if_cancelled(cancel_event)
            _progress(progress_cb, 75, "postprocess")
            outputs = self._save_postprocessed_outputs(predictions=predictions, run_dir=run_dir, save_all_outputs=True)
            output_faces = self._save_output_faces(outputs=outputs, run_dir=run_dir)
            post_seams = dreamcube_manual_cubemap.validate_depth_seams({
                face: outputs["depths_distance"][0, index, ..., 0]
                for index, face in enumerate(dreamcube_manual_cubemap.FACE_ORDER)
            })

            _raise_if_cancelled(cancel_event)
            _progress(progress_cb, 88, "export")
            mesh_stats: dict[str, Any] | None = None
            glb_path, mesh_stats = self._export_scene(
                outputs=outputs,
                run_dir=run_dir,
                mode=PANO_TO_3D_CUBEMAP,
                max_equi_size=None,
                max_cube_size=max_cube_size,
                mesh_depth_jump_threshold=mesh_depth_jump_threshold,
                mesh_footprint_ratio_threshold=mesh_footprint_ratio_threshold,
                mesh_aspect_ratio_threshold=mesh_aspect_ratio_threshold,
            )
            scene_manifest_path = self._write_scene_manifest(run_dir, glb_path)
            existing_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            existing_metadata.update({
                "manual_cubemap": manual_diagnostics,
                "output_faces": output_faces,
                "seam_metrics_postprocess": post_seams,
                "primary_output": {"status": "success", "format": "glb", "path": Path(glb_path).name},
                "mesh_export": {"status": "success", "format": "glb", "path": Path(glb_path).name, "stats": mesh_stats},
                "scene_manifest": {"status": "success", "path": Path(scene_manifest_path).name},
            })
            self._write_run_metadata(run_dir, existing_metadata)
            _progress(progress_cb, 100, "done")
            return Path(glb_path)
        except Exception as exc:
            _log(f"Manual cubemap generation failed stage=manual-cubemap error={exc}")
            self._record_failure_metadata(
                run_dir,
                {"status": "failed", "stage": "manual-cubemap", "error": str(exc)},
                section="manual_cubemap",
            )
            raise

    def _run_inference(
        self,
        *,
        image_path: Path,
        depth_path: Path,
        prompts: list[str],
        num_inference_steps: int,
        guidance_scale: float,
        normalize_scale: float,
        height: int,
        width: int,
        seed_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if self._pipe is None or self._app is None:
            raise RuntimeError("DreamCube pipeline is not loaded.")
        import torch

        cuda_available = torch.cuda.is_available()
        autocast_context = torch.amp.autocast("cuda") if cuda_available else contextlib.nullcontext()

        try:
            with torch.inference_mode():
                with autocast_context:
                    with contextlib.redirect_stdout(sys.stderr):
                        return self._app.inference(
                            self._pipe,
                            image=str(image_path),
                            depth=str(depth_path),
                            prompts=prompts,
                            height=height,
                            width=width,
                            output_type="np",
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            normalize_scale=normalize_scale,
                            **seed_kwargs,
                        )
        except TypeError as exc:
            if seed_kwargs:
                raise RuntimeError(
                    "DreamCube upstream inference rejected the seeded torch.Generator. "
                    "Retry with seed=-1 or update the upstream pipeline to accept generator=."
                ) from exc
            raise

    def _save_postprocessed_outputs(
        self,
        *,
        predictions: Mapping[str, Any],
        run_dir: Path,
        save_all_outputs: bool,
    ) -> dict[str, Any]:
        if self._app is None:
            raise RuntimeError("DreamCube upstream app is not loaded.")
        from PIL import Image

        images_pred = predictions.get("images")
        depths_pred = predictions.get("depths")
        if images_pred is None:
            raise RuntimeError("DreamCube upstream inference returned no RGB predictions.")
        if depths_pred is None:
            raise RuntimeError("DreamCube upstream inference returned no depth predictions.")
        if getattr(images_pred, "shape", [0])[0] != 1:
            raise RuntimeError("DreamCube wrapper expects batch size 1 for RGB predictions.")
        if getattr(depths_pred, "shape", [0])[0] != 1:
            raise RuntimeError("DreamCube wrapper expects batch size 1 for depth predictions.")

        depths_distance = self._app.z_distance_to_depth(depths_pred, fov_x=90.0, fov_y=90.0)

        post_rgb = self._app.postprocess_rgb(images_pred)
        equi_rgb = Image.fromarray(post_rgb["equi"][0])
        dice_rgb = Image.fromarray(post_rgb["dice"][0])
        equi_rgb_path = run_dir / "output_equi_rgb.png"
        equi_rgb.save(equi_rgb_path)

        post_depth = self._app.postprocess_depth(depths_distance)
        equi_depth_raw = Image.fromarray(post_depth["equi_depth_raw"][0])
        equi_depth_vis = Image.fromarray(post_depth["equi_depth_vis"][0])
        dice_depth_vis = Image.fromarray(post_depth["dice_depth_vis"][0])
        equi_depth_path = run_dir / "output_equi_depth.png"
        equi_depth_vis_path = run_dir / "output_equi_depth_vis.png"
        equi_depth_raw.save(equi_depth_path)
        equi_depth_vis.save(equi_depth_vis_path)

        outputs: dict[str, Any] = {
            "images_pred": images_pred,
            "depths_distance": depths_distance,
            "equi_rgb": equi_rgb,
            "equi_depth_raw": equi_depth_raw,
            "equi_rgb_path": equi_rgb_path,
            "equi_depth_path": equi_depth_path,
            "equi_depth_vis_path": equi_depth_vis_path,
        }

        if save_all_outputs:
            dice_rgb_path = run_dir / "output_dice_rgb.png"
            dice_depth_vis_path = run_dir / "output_dice_depth_vis.png"
            dice_rgb.save(dice_rgb_path)
            dice_depth_vis.save(dice_depth_vis_path)
            outputs["dice_rgb_path"] = dice_rgb_path
            outputs["dice_depth_vis_path"] = dice_depth_vis_path

        return outputs

    def _export_scene(
        self,
        *,
        outputs: Mapping[str, Any],
        run_dir: Path,
        mode: str,
        max_equi_size: int | None,
        max_cube_size: int | None,
        mesh_depth_jump_threshold: float,
        mesh_footprint_ratio_threshold: float = dreamcube_mesh.DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
        mesh_aspect_ratio_threshold: float = dreamcube_mesh.DEFAULT_ASPECT_RATIO_THRESHOLD,
    ) -> tuple[Path, dict[str, Any]]:
        if self._pipe is None or self._app is None:
            raise RuntimeError("DreamCube pipeline is not loaded.")
        import numpy as np
        import torch

        obj_path = run_dir / "output_mesh.obj"
        glb_path = run_dir / "output_mesh.glb"
        splat_path = run_dir / "output_3dgs.splat"
        device = getattr(self._pipe, "device", "cpu")

        with contextlib.redirect_stdout(sys.stderr):
            if mode == PANO_TO_3D_EQUIRECTANGULAR:
                rgb = torch.tensor(np.array(outputs["equi_rgb"]), device=device)
                distance = torch.tensor(
                    np.array(outputs["equi_depth_raw"]),
                    device=device,
                ) / 1000.0
                rays = dreamcube_mesh.equi_unit_rays(
                    int(distance.shape[0]),
                    int(distance.shape[1]),
                    device=device,
                )
                mesh_result = dreamcube_mesh.convert_rgbd_equi_to_mesh(
                    rgb=rgb,
                    distance=distance,
                    rays=rays,
                    max_size=max_equi_size,
                    save_path=str(obj_path),
                    depth_jump_threshold=mesh_depth_jump_threshold,
                    footprint_ratio_threshold=mesh_footprint_ratio_threshold,
                    aspect_ratio_threshold=mesh_aspect_ratio_threshold,
                )
                splat_rgb, splat_distance, splat_rays = self._prepare_3dgs_inputs(
                    rgb=rgb,
                    distance=distance,
                    rays=rays,
                    max_size=max_equi_size,
                    cubemap=False,
                )
                self._app.convert_rgbd_equi_to_3dgs(
                    rgb=splat_rgb,
                    distance=splat_distance,
                    rays=splat_rays,
                    max_size=max_equi_size,
                    save_path=str(splat_path),
                )
            else:
                rgb = torch.tensor(
                    outputs["images_pred"][0],
                    device=device,
                    dtype=torch.float32,
                )
                distance = torch.tensor(
                    outputs["depths_distance"][0, ..., 0],
                    device=device,
                    dtype=torch.float32,
                ) / 1000.0
                rays = dreamcube_mesh.cube_unit_rays(
                    int(distance.shape[1]),
                    device=device,
                )
                mesh_result = dreamcube_mesh.convert_rgbd_cube_to_mesh(
                    rgb=rgb,
                    distance=distance,
                    rays=rays,
                    max_size=max_cube_size,
                    save_path=str(obj_path),
                    depth_jump_threshold=mesh_depth_jump_threshold,
                    footprint_ratio_threshold=mesh_footprint_ratio_threshold,
                    aspect_ratio_threshold=mesh_aspect_ratio_threshold,
                )
                splat_rgb, splat_distance, splat_rays = self._prepare_3dgs_inputs(
                    rgb=rgb,
                    distance=distance,
                    rays=rays,
                    max_size=max_cube_size,
                    cubemap=True,
                )
                self._app.convert_rgbd_cube_to_3dgs(
                    rgb=splat_rgb,
                    distance=splat_distance,
                    rays=splat_rays,
                    max_size=max_cube_size,
                    save_path=str(splat_path),
                )

        mesh_stats = dict(mesh_result.stats)
        triangle_stats = mesh_stats["triangles"]
        vertex_stats = mesh_stats["vertices"]
        _log(
            "Mesh export "
            f"mode={mode} triangles={triangle_stats['exported']}/{triangle_stats['candidate']} "
            f"removed_invalid={triangle_stats.get('removed_invalid_or_repaired', 'unknown')} "
            f"removed_jump={triangle_stats.get('removed_depth_discontinuity', 'unknown')} "
            f"removed_footprint={triangle_stats.get('removed_footprint_ratio', 'unknown')} "
            f"removed_aspect={triangle_stats.get('removed_aspect_ratio', 'unknown')} "
            f"adaptive_diagonals={triangle_stats.get('adaptive_diagonals', 'unknown')} "
            f"vertices={vertex_stats['exported']}/{vertex_stats['candidate']} "
            f"invalid={vertex_stats.get('invalid', {}).get('total', 'unknown') if isinstance(vertex_stats.get('invalid'), Mapping) else 'unknown'} "
            f"repaired={vertex_stats.get('repaired', 'unknown')} repair_rounds={vertex_stats.get('repair_rounds', 'unknown')} "
            f"retention={triangle_stats.get('retention_percent', 'unknown')} "
            f"thresholds=jump:{mesh_depth_jump_threshold:g},footprint:{mesh_footprint_ratio_threshold:g},aspect:{mesh_aspect_ratio_threshold:g}"
        )
        return self._convert_obj_to_glb(obj_path, glb_path, mesh_stats), mesh_stats

    def generate(
        self,
        image_bytes: bytes,
        params: dict[str, Any] | None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> Path:
        safe_params: dict[str, Any] = dict(params or {})

        _raise_if_cancelled(cancel_event)
        _progress(progress_cb, 2, "validate")
        node_id = self._resolve_node_id(safe_params)
        defaults = _schema_defaults(node_id)
        if node_id == MANUAL_SCENE_NODE_ID:
            return self._generate_manual_cubemap(image_bytes, safe_params, defaults, progress_cb, cancel_event)

        depth_path, prompts = self._validate_request(image_bytes, safe_params)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.outputs_dir / f"dreamcube-{timestamp}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path, rgb_metadata = self._save_input_rgb(image_bytes, run_dir)
        rgb_original_size = (int(rgb_metadata["original_size"][0]), int(rgb_metadata["original_size"][1]))
        rgb_crop_box = tuple(int(value) for value in rgb_metadata["crop_box"])

        num_inference_steps = _safe_int(
            _param(safe_params, defaults, "num_inference_steps", 50),
            50,
            minimum=1,
            maximum=100,
        )
        guidance_scale = _safe_float(
            _param(safe_params, defaults, "guidance_scale", 7.5),
            7.5,
            minimum=0.0,
            maximum=20.0,
        )
        normalize_scale = _safe_float(
            _param(safe_params, defaults, "normalize_scale", 0.6),
            0.6,
            minimum=0.05,
            maximum=5.0,
        )
        pano_to_3d_mode = _safe_choice(
            _param(safe_params, defaults, "pano_to_3d_mode", PANO_TO_3D_CUBEMAP),
            PANO_TO_3D_MODES,
            PANO_TO_3D_CUBEMAP,
        )
        output_format = "glb" if node_id in {SCENE_NODE_ID, MANUAL_SCENE_NODE_ID} else "equirect_rgb_png"
        max_equi_size = _max_size(_param(safe_params, defaults, "max_equi_size", 1024), 1024)
        max_cube_size = _max_size(_param(safe_params, defaults, "max_cube_size", 256), 256)
        mesh_depth_jump_threshold = _safe_float(
            _param(
                safe_params,
                defaults,
                "mesh_depth_jump_threshold",
                dreamcube_mesh.DEFAULT_DEPTH_JUMP_THRESHOLD,
            ),
            dreamcube_mesh.DEFAULT_DEPTH_JUMP_THRESHOLD,
            minimum=0.0,
            maximum=5.0,
        )
        mesh_footprint_ratio_threshold = _safe_float(
            _param(
                safe_params,
                defaults,
                "mesh_footprint_ratio_threshold",
                dreamcube_mesh.DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
            ),
            dreamcube_mesh.DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
            minimum=0.0,
            maximum=100.0,
        )
        mesh_aspect_ratio_threshold = _safe_float(
            _param(
                safe_params,
                defaults,
                "mesh_aspect_ratio_threshold",
                dreamcube_mesh.DEFAULT_ASPECT_RATIO_THRESHOLD,
            ),
            dreamcube_mesh.DEFAULT_ASPECT_RATIO_THRESHOLD,
            minimum=0.0,
            maximum=100.0,
        )
        save_all_outputs = _safe_bool(
            _param(safe_params, defaults, "save_all_outputs", node_id == SCENE_NODE_ID),
            default=node_id == SCENE_NODE_ID,
        )
        save_input_depth = _safe_bool(_param(safe_params, defaults, "save_input_depth", True), default=True)
        auto_depth_variant = _safe_choice(
            _param(safe_params, defaults, "auto_depth_variant", AUTO_DEPTH_DEFAULT_VARIANT),
            tuple(AUTO_DEPTH_VARIANTS),
            AUTO_DEPTH_DEFAULT_VARIANT,
        )
        seed = _safe_int(_param(safe_params, defaults, "seed", -1), -1, minimum=-1, maximum=2_147_483_647)
        height = _safe_int(safe_params.get("height"), 512, minimum=64, maximum=2048)
        width = _safe_int(safe_params.get("width"), 512, minimum=64, maximum=2048)

        if depth_path is None:
            _raise_if_cancelled(cancel_event)
            _progress(progress_cb, 8, "auto-depth")
            depth_path = self._save_auto_depth_image(image_path, run_dir, auto_depth_variant)
            depth_metadata: dict[str, Any] = {
                "mode": "auto",
                "source": "internal-auto-depth",
                "path": Path(depth_path).name,
                "canonical_size": rgb_metadata["canonical_size"],
                "crop_source": "canonical_rgb",
            }
            if not save_input_depth:
                # DreamCube still needs the file for inference. The parameter controls
                # user-facing retention; remove it only after inference has consumed it.
                safe_params["_remove_auto_depth_after_inference"] = True
        else:
            depth_path, depth_metadata = self._save_manual_depth_image(
                Path(depth_path),
                run_dir,
                rgb_original_size,
                rgb_crop_box,
            )

        run_metadata: dict[str, Any] = {
            "created_at": _utc_now(),
            "node_id": node_id,
            "output_format": output_format,
            "original_rgb_size": rgb_metadata["original_size"],
            "canonical_rgb_size": rgb_metadata["canonical_size"],
            "crop_box": rgb_metadata["crop_box"],
            "input_rgb": rgb_metadata,
            "depth_mode": depth_metadata["mode"],
            "depth_source": depth_metadata["source"],
            "depth": depth_metadata,
            "auto_depth_variant": auto_depth_variant,
            "prompts": dict(zip(PROMPT_FIELDS, prompts)),
            "reconstruction": {
                "mode": pano_to_3d_mode,
                "max_equi_size": max_equi_size,
                "max_cube_size": max_cube_size,
                "mesh_depth_jump_threshold": mesh_depth_jump_threshold,
                "mesh_footprint_ratio_threshold": mesh_footprint_ratio_threshold,
                "mesh_aspect_ratio_threshold": mesh_aspect_ratio_threshold,
            },
        }
        if node_id in {SCENE_NODE_ID, MANUAL_SCENE_NODE_ID}:
            run_metadata["coordinate_frame"] = _coordinate_frame_payload()
            run_metadata["presentation"] = _presentation_payload()
        self._write_run_metadata(run_dir, run_metadata)

        _raise_if_cancelled(cancel_event)
        _progress(progress_cb, 10, "load")
        self.load()
        seed_kwargs = self._build_seed_kwargs(seed)

        _raise_if_cancelled(cancel_event)
        _progress(progress_cb, 25, "infer")
        predictions = self._run_inference(
            image_path=image_path,
            depth_path=depth_path,
            prompts=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            normalize_scale=normalize_scale,
            height=height,
            width=width,
            seed_kwargs=seed_kwargs,
        )
        if safe_params.get("_remove_auto_depth_after_inference"):
            try:
                Path(depth_path).unlink(missing_ok=True)
            except OSError:
                pass

        _raise_if_cancelled(cancel_event)
        _progress(progress_cb, 75, "postprocess")
        outputs = self._save_postprocessed_outputs(
            predictions=predictions,
            run_dir=run_dir,
            save_all_outputs=save_all_outputs,
        )

        _raise_if_cancelled(cancel_event)
        _progress(progress_cb, 88, "export")
        if save_all_outputs:
            face_outputs = self._save_output_faces(outputs=outputs, run_dir=run_dir)
            existing_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            existing_metadata["output_faces"] = face_outputs
            self._write_run_metadata(run_dir, existing_metadata)

        if node_id == SCENE_NODE_ID:
            mesh_stats: dict[str, Any] | None = None
            try:
                glb_path, mesh_stats = self._export_scene(
                    outputs=outputs,
                    run_dir=run_dir,
                    mode=pano_to_3d_mode,
                    max_equi_size=max_equi_size,
                    max_cube_size=max_cube_size,
                    mesh_depth_jump_threshold=mesh_depth_jump_threshold,
                    mesh_footprint_ratio_threshold=mesh_footprint_ratio_threshold,
                    mesh_aspect_ratio_threshold=mesh_aspect_ratio_threshold,
                )
                scene_manifest_path = self._write_scene_manifest(run_dir, glb_path)
            except dreamcube_mesh.MeshExportError as exc:
                stats = exc.stats if isinstance(exc.stats, Mapping) else {}
                vertices = stats.get("vertices") if isinstance(stats.get("vertices"), Mapping) else {}
                triangles = stats.get("triangles") if isinstance(stats.get("triangles"), Mapping) else {}
                invalid = vertices.get("invalid") if isinstance(vertices.get("invalid"), Mapping) else {}
                _log(
                    "Mesh export failed "
                    f"mode={stats.get('mode', 'unknown')} "
                    f"triangles={triangles.get('exported', 'unknown')}/{triangles.get('candidate', 'unknown')} "
                    f"removed_invalid={triangles.get('removed_invalid_or_repaired', 'unknown')} "
                    f"removed_jump={triangles.get('removed_depth_discontinuity', 'unknown')} "
                    f"removed_footprint={triangles.get('removed_footprint_ratio', 'unknown')} "
                    f"removed_aspect={triangles.get('removed_aspect_ratio', 'unknown')} "
                    f"adaptive_diagonals={triangles.get('adaptive_diagonals', 'unknown')} "
                    f"vertices={vertices.get('exported', 'unknown')}/{vertices.get('candidate', 'unknown')} "
                    f"invalid={invalid.get('total', 'unknown') if isinstance(invalid, Mapping) else 'unknown'} "
                    f"repaired={vertices.get('repaired', 'unknown')} repair_rounds={vertices.get('repair_rounds', 'unknown')} "
                    f"retention={triangles.get('retention_percent', 'unknown')} "
                    f"thresholds=jump:{mesh_depth_jump_threshold:g},footprint:{mesh_footprint_ratio_threshold:g},aspect:{mesh_aspect_ratio_threshold:g}"
                )
                self._record_failure_metadata(
                    run_dir,
                    {
                        "status": "failed",
                        "error": str(exc),
                        "stats": exc.stats,
                    },
                )
                raise
            except SceneGenerationError as exc:
                _log(f"Scene generation failed stage={exc.stage} error={exc}")
                self._record_failure_metadata(
                    run_dir,
                    {
                        "status": "failed",
                        "stage": exc.stage,
                        "error": str(exc),
                        "stats": exc.stats if exc.stats is not None else mesh_stats,
                        "diagnostics": exc.diagnostics,
                    },
                )
                raise

            existing_metadata = json.loads(
                (run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            existing_metadata.update(
                {
                    "primary_output": {
                        "status": "success",
                        "format": "glb",
                        "path": Path(glb_path).name,
                    },
                    "mesh_export": {
                        "status": "success",
                        "format": "glb",
                        "path": Path(glb_path).name,
                        "stats": mesh_stats,
                    },
                    "scene_manifest": {
                        "status": "success",
                        "path": Path(scene_manifest_path).name,
                    },
                }
            )
            self._write_run_metadata(run_dir, existing_metadata)
            result_path = Path(glb_path)
        else:
            result_path = Path(outputs["equi_rgb_path"])

        _progress(progress_cb, 100, "done")
        return Path(result_path)


assert DreamCubeGenerator.__name__ == "DreamCubeGenerator"
