#!/usr/bin/env python3
"""Setup script for the Modly DreamCube extension.

Modly/Electron calls this script as:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":128}'

Legacy positional form is also supported:
    python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXTENSION_ID = "dreamcube"
STATUS_SCHEMA = "modly.setup-status.v1"
SCRIPT_DIR = Path(__file__).resolve().parent

UPSTREAM_REPO_URL = "https://github.com/Yukun-Huang/DreamCube.git"
UPSTREAM_REF = "main"
UPSTREAM_RELATIVE_PATH = Path(".modly") / "upstream" / "DreamCube"

HF_REPO = "KevinHuang/DreamCube"
DOWNLOAD_CHECK = "model_index.json"
AUTO_DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
AUTO_DEPTH_DEFAULT_VARIANT = "vits"
AUTO_DEPTH_RELATIVE_CACHE = Path(".modly") / "auto-depth" / "cache"
CANONICAL_WEIGHT_OWNER_PATH = Path(EXTENSION_ID) / EXTENSION_ID
DEFAULT_MODEL_RELATIVE_PATH = Path("models") / CANONICAL_WEIGHT_OWNER_PATH

SETUP_RELATIVE_PATH = Path(".modly") / "setup"
STATUS_RELATIVE_PATH = SETUP_RELATIVE_PATH / "setup-status.json"
LOG_RELATIVE_PATH = SETUP_RELATIVE_PATH / "logs" / "setup.log"

BOOTSTRAP_PACKAGES = ["pip", "setuptools", "wheel", "huggingface_hub>=0.24,<1.0"]
PYTORCH3D_UPSTREAM_PACKAGE = "pytorch3d" + "==0.7.8"
PYTORCH3D_SOURCE_URL = "git+https://github.com/facebookresearch/pytorch3d.git@stable"
PYTORCH3D_MODES = {"auto", "source", "shim", "required"}
OPEN3D_PACKAGE = "open3d"
OPEN3D_DOCS_WHEEL_INDEX = "https://www.open3d.org/docs/latest/getting_started.html"
OPEN3D_SOURCE_URL = "git+https://github.com/isl-org/Open3D.git"
OPEN3D_MODES = {"auto", "wheel", "source", "shim", "required"}

UPSTREAM_DEPENDENCIES = [
    "diffusers==0.32.0",
    "transformers==4.48.3",
    "scipy==1.15.3",
    "gsplat==1.5.2",
    "einops",
    "matplotlib",
    "opencv_python",
    "numpy",
    "pillow",
    "tqdm",
    "trimesh",
]
PROBE_IMPORTS = [
    "torch",
    "diffusers",
    "transformers",
    "gsplat",
    "pytorch3d.transforms",
    "open3d",
    "trimesh",
    "models.dreamcube",
    "utils.pano_to_3d",
]

SECRET_KEYWORDS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_key",
    "auth",
)
SECRET_ENV_NAMES = {
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GITHUB_TOKEN",
    "GIT_ASKPASS",
}


class SetupError(RuntimeError):
    """Expected setup failure that should be written as a status blocker."""

    def __init__(self, message: str, code: str = "setup-error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SetupConfig:
    python_exe: str
    ext_dir: Path
    gpu_sm: int
    cuda_version: int
    model_dir: Path
    validate_only: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TorchLane:
    label: str
    packages: list[str]
    index_url: str
    cuda_tag: str
    rationale: str

    def as_status(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "packages": self.packages,
            "index_url": self.index_url,
            "cuda_tag": self.cuda_tag,
            "rationale": self.rationale,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return upper in SECRET_ENV_NAMES or any(keyword.upper() in upper for keyword in SECRET_KEYWORDS)


def _secret_env_values(env: dict[str, str] | None = None) -> list[str]:
    source = os.environ if env is None else env
    values: list[str] = []
    for key, value in source.items():
        if not value or len(value) < 7:
            continue
        if _is_secret_env_name(key):
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def redact_text(value: Any, env: dict[str, str] | None = None) -> str:
    """Redact likely credentials from user-visible setup logs."""
    text = str(value)
    for secret in _secret_env_values(env):
        text = text.replace(secret, "[REDACTED]")

    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(://)([^/@:\s]+):([^/@\s]+)@", r"\1[REDACTED]@", text)
    text = re.sub(
        r"(?i)\b(token|secret|password|passwd|api[_-]?key|access[_-]?key|auth[_-]?token)(\s*[:=]\s*)([^\s,\"']+)",
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )
    text = re.sub(r"hf_[A-Za-z0-9]{8,}", "hf_[REDACTED]", text)
    return text


class SetupLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.log_path.open("a", encoding="utf-8")
        self.info(f"Log opened: {self.log_path}")

    def close(self) -> None:
        self._handle.close()

    def info(self, message: str) -> None:
        self.raw(f"[setup:{EXTENSION_ID}] {message}")

    def raw(self, message: str) -> None:
        sanitized = redact_text(message.rstrip())
        print(sanitized, flush=True)
        self._handle.write(sanitized + "\n")
        self._handle.flush()


class StatusTracker:
    def __init__(self, config: SetupConfig, log_path: Path) -> None:
        self.status_path = config.ext_dir / STATUS_RELATIVE_PATH
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_phase: dict[str, Any] | None = None
        self.data: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "extension_id": EXTENSION_ID,
            "status": "running",
            "phases": [],
            "blockers": [],
            "started_at": utc_now(),
            "completed_at": None,
            "gpu_sm": config.gpu_sm,
            "cuda_version": config.cuda_version,
            "python_exe": config.python_exe,
            "upstream_repo": UPSTREAM_REPO_URL,
            "upstream_ref": UPSTREAM_REF,
            "upstream_commit": None,
            "hf_repo": HF_REPO,
            "download_check": DOWNLOAD_CHECK,
            "model_dir": str(config.model_dir),
            "model_weights_owner": "modly-ui",
            "model_weights_note": "Model weights are managed by the Modly UI/downloader, not setup.py.",
            "venv_dir": str(config.ext_dir / "venv"),
            "log_path": str(log_path),
            "torch_lane": None,
            "pytorch3d_provider": None,
            "open3d_provider": None,
            "auto_depth_provider": {
                "repo_id": AUTO_DEPTH_REPO,
                "default_variant": AUTO_DEPTH_DEFAULT_VARIANT,
                "cache_dir": str(config.ext_dir / AUTO_DEPTH_RELATIVE_CACHE),
                "managed_by_setup": False,
                "download_behavior": "lazy-runtime-cache",
                "note": "Internal DreamCube auto-depth; setup installs dependencies but does not download Depth Anything weights.",
            },
        }
        self.write()

    def write(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.status_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.status_path)

    def start_phase(self, phase_id: str, message: str) -> None:
        phase = {
            "id": phase_id,
            "status": "running",
            "message": message,
            "started_at": utc_now(),
            "completed_at": None,
        }
        self.data["phases"].append(phase)
        self.current_phase = phase
        self.write()

    def complete_current_phase(self, details: dict[str, Any] | None = None) -> None:
        if self.current_phase is None:
            return
        self.current_phase["status"] = "complete"
        self.current_phase["completed_at"] = utc_now()
        if details:
            self.current_phase["details"] = details
        self.current_phase = None
        self.write()

    def fail_current_phase(self, message: str) -> None:
        if self.current_phase is None:
            return
        self.current_phase["status"] = "failed"
        self.current_phase["completed_at"] = utc_now()
        self.current_phase["error"] = redact_text(message)
        self.current_phase = None
        self.write()

    def add_blocker(self, code: str, message: str) -> None:
        self.data["blockers"].append({"code": code, "message": redact_text(message)})
        self.write()

    def mark_ready(self) -> None:
        self.data["status"] = "ready"
        self.data["completed_at"] = utc_now()
        self.write()

    def mark_failed(self) -> None:
        self.data["status"] = "failed"
        self.data["completed_at"] = utc_now()
        self.write()


def usage() -> str:
    return (
        "Usage: python setup.py '<json-payload>' [--validate-only]\n"
        "   or: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version] [--validate-only]"
    )


def normalize_gpu_sm(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise SystemExit("gpu_sm must be numeric, not boolean")
    text = str(value).strip().lower()
    text = text.replace("compute_", "").replace("sm_", "").replace("sm", "")
    try:
        if "." in text:
            major, minor = text.split(".", 1)
            return int(major) * 10 + int((minor or "0")[0])
        number = int(text)
        return number * 10 if 0 < number < 10 else number
    except ValueError as exc:
        raise SystemExit(f"gpu_sm must be numeric; got {value!r}") from exc


def normalize_cuda_version(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise SystemExit("cuda_version must be numeric, not boolean")
    text = str(value).strip().lower()
    for prefix in ("cuda", "cu", "v"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.strip()
    try:
        if "." in text:
            major, minor = text.split(".", 1)
            minor_digits = "".join(ch for ch in minor if ch.isdigit())
            return int(major) * 10 + int((minor_digits or "0")[0])
        number = int(text)
        return number * 10 if 0 < number < 20 else number
    except ValueError as exc:
        raise SystemExit(f"cuda_version must be numeric; got {value!r}") from exc


def resolve_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base / expanded).resolve()


def _candidate_modly_settings_paths(env: dict[str, str]) -> list[Path]:
    candidates: list[Path] = []

    direct = env.get("MODLY_SETTINGS_PATH")
    if direct and direct.strip():
        candidates.append(Path(os.path.expandvars(direct.strip())).expanduser())

    xdg_config_home = env.get("XDG_CONFIG_HOME")
    if xdg_config_home and xdg_config_home.strip():
        candidates.append(Path(os.path.expandvars(xdg_config_home.strip())).expanduser() / "Modly" / "settings.json")

    home = env.get("HOME")
    if home and home.strip():
        candidates.append(Path(os.path.expandvars(home.strip())).expanduser() / ".config" / "Modly" / "settings.json")

    appdata = env.get("APPDATA")
    if appdata and appdata.strip():
        candidates.append(Path(os.path.expandvars(appdata.strip())).expanduser() / "Modly" / "settings.json")

    userprofile = env.get("USERPROFILE")
    if userprofile and userprofile.strip():
        base = Path(os.path.expandvars(userprofile.strip())).expanduser()
        candidates.append(base / "AppData" / "Roaming" / "Modly" / "settings.json")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def resolve_models_dir_from_settings(env: dict[str, str]) -> Path | None:
    """Resolve Modly's configured modelsDir when Electron does not pass it."""

    for settings_path in _candidate_modly_settings_paths(env):
        if not settings_path.is_file():
            continue
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("modelsDir")
        if isinstance(value, str) and value.strip():
            return Path(os.path.expandvars(value.strip())).expanduser().resolve()
    return None


def resolve_model_dir(payload: dict[str, Any], ext_dir: Path, env: dict[str, str]) -> Path:
    direct_payload_keys = (
        "model_dir",
        "modelDir",
        "model_path",
        "modelPath",
        "hf_model_dir",
        "hfModelDir",
        "weights_dir",
        "weightsDir",
        "download_dir",
        "downloadDir",
    )
    for key in direct_payload_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return resolve_path(value.strip(), ext_dir)

    direct_env_keys = (
        "DREAMCUBE_MODEL_DIR",
        "MODLY_DREAMCUBE_MODEL_DIR",
        "MODLY_MODEL_DIR",
        "MODEL_DIR",
    )
    for key in direct_env_keys:
        value = env.get(key)
        if value and value.strip():
            return resolve_path(value.strip(), ext_dir)

    root_payload_keys = ("models_dir", "modelsDir", "model_root", "modelRoot", "models_root", "modelsRoot")
    for key in root_payload_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return resolve_path(value.strip(), ext_dir) / CANONICAL_WEIGHT_OWNER_PATH

    root_env_keys = ("MODLY_MODELS_DIR", "MODELS_DIR", "MODEL_ROOT", "MODELS_ROOT")
    for key in root_env_keys:
        value = env.get(key)
        if value and value.strip():
            return resolve_path(value.strip(), ext_dir) / CANONICAL_WEIGHT_OWNER_PATH

    configured_models_dir = resolve_models_dir_from_settings(env)
    if configured_models_dir is not None:
        return configured_models_dir / CANONICAL_WEIGHT_OWNER_PATH

    return (ext_dir / DEFAULT_MODEL_RELATIVE_PATH).resolve()


def normalize_payload(payload: dict[str, Any], *, validate_only: bool, env: dict[str, str]) -> SetupConfig:
    python_exe = payload.get("python_exe") or payload.get("pythonExe")
    ext_dir = payload.get("ext_dir") or payload.get("extDir")
    if not isinstance(python_exe, str) or not python_exe.strip():
        raise SystemExit("setup payload must include a non-empty python_exe")
    if not isinstance(ext_dir, str) or not ext_dir.strip():
        raise SystemExit("setup payload must include a non-empty ext_dir")

    ext_path = resolve_path(ext_dir.strip(), SCRIPT_DIR)
    gpu_sm = normalize_gpu_sm(payload.get("gpu_sm", payload.get("gpuSm", 0)))
    cuda_version = normalize_cuda_version(payload.get("cuda_version", payload.get("cudaVersion", 0)))
    model_dir = resolve_model_dir(payload, ext_path, env)
    return SetupConfig(
        python_exe=python_exe.strip(),
        ext_dir=ext_path,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        model_dir=model_dir,
        validate_only=validate_only,
        payload=dict(payload),
    )


def parse_args(argv: list[str] | None = None, env: dict[str, str] | None = None) -> SetupConfig:
    args = list(sys.argv[1:] if argv is None else argv)
    source_env = dict(os.environ if env is None else env)
    validate_only = False
    payload_args: list[str] = []

    for arg in args:
        if arg == "--validate-only":
            validate_only = True
        elif arg in {"-h", "--help"}:
            raise SystemExit(usage())
        else:
            payload_args.append(arg)

    if len(payload_args) == 1:
        try:
            payload = json.loads(payload_args[0])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"setup payload must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("setup payload must be a JSON object")
        return normalize_payload(payload, validate_only=validate_only, env=source_env)

    if len(payload_args) in {3, 4}:
        payload: dict[str, Any] = {
            "python_exe": payload_args[0],
            "ext_dir": payload_args[1],
            "gpu_sm": payload_args[2],
        }
        if len(payload_args) == 4:
            payload["cuda_version"] = payload_args[3]
        return normalize_payload(payload, validate_only=validate_only, env=source_env)

    if not payload_args and validate_only:
        payload = {
            "python_exe": sys.executable,
            "ext_dir": str(SCRIPT_DIR),
            "gpu_sm": 0,
            "cuda_version": 0,
        }
        return normalize_payload(payload, validate_only=True, env=source_env)

    raise SystemExit(usage())


def venv_python(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def command_display(cmd: list[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def run_command(
    cmd: list[str],
    logger: SetupLogger,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    code: str = "command-failed",
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    command_text = command_display(cmd)
    if cwd is not None:
        logger.info(f"$ {command_text}  (cwd={cwd})")
    else:
        logger.info(f"$ {command_text}")

    output_lines: list[str] = []
    try:
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd) if cwd is not None else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise SetupError(f"Unable to start command {command_text}: {exc}", code=code) from exc

    assert process.stdout is not None
    for line in process.stdout:
        sanitized = redact_text(line.rstrip("\n"), merged_env)
        output_lines.append(sanitized)
        logger.raw(sanitized)

    return_code = process.wait()
    stdout = "\n".join(output_lines)
    if check and return_code != 0:
        raise SetupError(f"Command failed with exit code {return_code}: {command_text}", code=code)
    return subprocess.CompletedProcess([str(part) for part in cmd], return_code, stdout, None)


def pip_install(py: Path, logger: SetupLogger, packages: list[str], extra_args: list[str] | None = None) -> None:
    args = [str(py), "-m", "pip", "install", "--no-cache-dir", "--retries", "5", "--timeout", "120"]
    if extra_args:
        args.extend(extra_args)
    args.extend(packages)
    run_command(args, logger, code="pip-install-failed")


def select_torch_lane(gpu_sm: int, cuda_version: int) -> TorchLane:
    if gpu_sm >= 100 or cuda_version >= 128:
        return TorchLane(
            label="pytorch-cu128-modern-gpu",
            packages=["torch==2.7.0", "torchvision==0.22.0"],
            index_url="https://download.pytorch.org/whl/cu128",
            cuda_tag="cu128",
            rationale="SM >= 100 or CUDA >= 12.8 requires a cu128-capable PyTorch lane when available.",
        )

    if cuda_version >= 124:
        cuda_tag = "cu124"
    elif cuda_version >= 121 or cuda_version == 0:
        cuda_tag = "cu121"
    else:
        cuda_tag = "cu118"

    return TorchLane(
        label=f"upstream-pins-{cuda_tag}",
        packages=["torch==2.4.1", "torchvision==0.19.1"],
        index_url=f"https://download.pytorch.org/whl/{cuda_tag}",
        cuda_tag=cuda_tag,
        rationale="Using DreamCube upstream PyTorch pins for pre-Blackwell CUDA environments.",
    )


def dependency_name(package: str) -> str:
    return re.split(r"[<>=!~\[]", package, maxsplit=1)[0].strip().lower()


def validate_internal_config(config: SetupConfig) -> dict[str, Any]:
    if HF_REPO != "KevinHuang/DreamCube":
        raise SetupError(f"Unexpected HF repo constant: {HF_REPO}", code="invalid-config")
    if DOWNLOAD_CHECK != "model_index.json":
        raise SetupError(f"Unexpected download check constant: {DOWNLOAD_CHECK}", code="invalid-config")
    if AUTO_DEPTH_REPO != "depth-anything/Depth-Anything-V2-Small-hf":
        raise SetupError(f"Unexpected auto-depth repo constant: {AUTO_DEPTH_REPO}", code="invalid-config")
    if AUTO_DEPTH_DEFAULT_VARIANT != "vits":
        raise SetupError(f"Unexpected auto-depth variant constant: {AUTO_DEPTH_DEFAULT_VARIANT}", code="invalid-config")
    if not UPSTREAM_DEPENDENCIES:
        raise SetupError("No upstream dependency list configured", code="invalid-config")

    required_payloads = ("manifest.json", "generator.py", "dreamcube_mesh.py")
    missing_payloads = [name for name in required_payloads if not (SCRIPT_DIR / name).is_file()]
    if missing_payloads:
        raise SetupError(
            "Missing required extension payload(s): " + ", ".join(missing_payloads),
            code="missing-extension-payload",
        )

    manifest_path = SCRIPT_DIR / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SetupError(f"manifest.json is invalid JSON: {exc}", code="invalid-manifest") from exc

    if manifest.get("id") != EXTENSION_ID:
        raise SetupError(f"manifest id must be {EXTENSION_ID!r}; got {manifest.get('id')!r}", code="invalid-manifest")
    if manifest.get("type") != "model":
        raise SetupError("manifest type must be 'model'", code="invalid-manifest")
    if manifest.get("hf_repo") != HF_REPO:
        raise SetupError(f"manifest hf_repo must be {HF_REPO!r}", code="invalid-manifest")
    if manifest.get("download_check") != DOWNLOAD_CHECK:
        raise SetupError(f"manifest download_check must be {DOWNLOAD_CHECK!r}", code="invalid-manifest")

    if PYTORCH3D_UPSTREAM_PACKAGE in UPSTREAM_DEPENDENCIES:
        raise SetupError(
            f"PyTorch3D must be managed by the provider phase, not regular dependencies: {PYTORCH3D_UPSTREAM_PACKAGE}",
            code="invalid-config",
        )
    open3d_dependency = next((package for package in UPSTREAM_DEPENDENCIES if dependency_name(package) == OPEN3D_PACKAGE), None)
    if open3d_dependency is not None:
        raise SetupError(
            f"Open3D must be managed by the provider phase, not regular dependencies: {open3d_dependency}",
            code="invalid-config",
        )

    dependency_names = {dependency_name(package) for package in UPSTREAM_DEPENDENCIES}
    required_auto_depth_deps = {"transformers", "pillow", "numpy"}
    missing_auto_depth_deps = sorted(required_auto_depth_deps.difference(dependency_names))
    if missing_auto_depth_deps:
        raise SetupError(
            f"Auto-depth dependencies missing from DreamCube runtime dependencies: {', '.join(missing_auto_depth_deps)}",
            code="invalid-config",
        )

    setup_cache = manifest.get("setup", {}).get("managed_runtime_cache", {}) if isinstance(manifest.get("setup"), dict) else {}
    auto_depth_manifest = setup_cache.get("auto_depth") if isinstance(setup_cache, dict) else None
    if not isinstance(auto_depth_manifest, dict) or auto_depth_manifest.get("repo_id") != AUTO_DEPTH_REPO:
        raise SetupError("manifest.setup.managed_runtime_cache.auto_depth must declare the internal auto-depth repo", code="invalid-manifest")
    if auto_depth_manifest.get("managed_by_setup") is not False:
        raise SetupError("manifest auto-depth weights must not be managed/downloaded by setup", code="invalid-manifest")

    lane = select_torch_lane(config.gpu_sm, config.cuda_version)
    return {
        "manifest": str(manifest_path),
        "torch_lane": lane.as_status(),
        "dependency_count": len(UPSTREAM_DEPENDENCIES),
        "excluded_dependencies": [PYTORCH3D_UPSTREAM_PACKAGE, OPEN3D_PACKAGE],
        "model_dir": str(config.model_dir),
        "auto_depth_provider": {
            "repo_id": AUTO_DEPTH_REPO,
            "default_variant": AUTO_DEPTH_DEFAULT_VARIANT,
            "cache_dir": str(config.ext_dir / AUTO_DEPTH_RELATIVE_CACHE),
            "managed_by_setup": False,
            "download_behavior": "lazy-runtime-cache",
        },
    }


def phase(
    tracker: StatusTracker,
    logger: SetupLogger,
    phase_id: str,
    message: str,
    callback: Any,
) -> Any:
    logger.info(f"PHASE {phase_id}: {message}")
    tracker.start_phase(phase_id, message)
    result = callback()
    details = result if isinstance(result, dict) else None
    tracker.complete_current_phase(details)
    return result


def preflight(config: SetupConfig, logger: SetupLogger) -> dict[str, Any]:
    details = validate_internal_config(config)
    if not config.ext_dir.exists():
        raise SetupError(f"Extension directory does not exist: {config.ext_dir}", code="missing-extension-dir")

    manifest_at_ext_dir = config.ext_dir / "manifest.json"
    if not manifest_at_ext_dir.exists():
        raise SetupError(f"Expected manifest at extension directory: {manifest_at_ext_dir}", code="missing-manifest")

    py_version = run_command([config.python_exe, "--version"], logger, code="python-executable-failed").stdout.strip()
    git_version = run_command(["git", "--version"], logger, code="git-unavailable").stdout.strip()
    details.update({"python_version": py_version, "git_version": git_version})
    return details


def ensure_venv(config: SetupConfig, logger: SetupLogger) -> dict[str, Any]:
    venv_dir = config.ext_dir / "venv"
    py = venv_python(venv_dir)
    if py.exists():
        logger.info(f"Reusing venv at {venv_dir}")
        return {"venv_dir": str(venv_dir), "python": str(py), "created": False}

    logger.info(f"Creating venv at {venv_dir}")
    run_command([config.python_exe, "-m", "venv", str(venv_dir)], logger, code="venv-create-failed")
    if not py.exists():
        raise SetupError(f"Venv Python was not created at {py}", code="venv-create-failed")
    return {"venv_dir": str(venv_dir), "python": str(py), "created": True}


def bootstrap_venv(py: Path, logger: SetupLogger) -> dict[str, Any]:
    run_command([str(py), "-m", "ensurepip", "--upgrade"], logger, check=False, code="ensurepip-failed")
    run_command(
        [str(py), "-m", "pip", "install", "--upgrade", "--no-cache-dir", *BOOTSTRAP_PACKAGES],
        logger,
        code="bootstrap-pip-failed",
    )
    return {"packages": BOOTSTRAP_PACKAGES}


def sync_upstream(config: SetupConfig, logger: SetupLogger) -> dict[str, Any]:
    upstream_dir = config.ext_dir / UPSTREAM_RELATIVE_PATH
    upstream_dir.parent.mkdir(parents=True, exist_ok=True)
    git_dir = upstream_dir / ".git"

    if upstream_dir.exists() and not git_dir.exists():
        if any(upstream_dir.iterdir()):
            raise SetupError(
                f"Upstream path exists but is not a git checkout: {upstream_dir}",
                code="upstream-path-not-git",
            )
        run_command(
            ["git", "clone", "--branch", UPSTREAM_REF, "--single-branch", UPSTREAM_REPO_URL, str(upstream_dir)],
            logger,
            code="upstream-clone-failed",
        )
    elif git_dir.exists():
        logger.info(f"Updating existing DreamCube checkout at {upstream_dir}")
        run_command(["git", "remote", "set-url", "origin", UPSTREAM_REPO_URL], logger, cwd=upstream_dir, code="upstream-update-failed")
        run_command(["git", "fetch", "--prune", "origin", UPSTREAM_REF], logger, cwd=upstream_dir, code="upstream-fetch-failed")
    else:
        run_command(
            ["git", "clone", "--branch", UPSTREAM_REF, "--single-branch", UPSTREAM_REPO_URL, str(upstream_dir)],
            logger,
            code="upstream-clone-failed",
        )

    run_command(["git", "checkout", "-B", UPSTREAM_REF, f"origin/{UPSTREAM_REF}"], logger, cwd=upstream_dir, code="upstream-checkout-failed")
    commit = run_command(["git", "rev-parse", "HEAD"], logger, cwd=upstream_dir, code="upstream-rev-parse-failed").stdout.strip().splitlines()[-1]
    return {"upstream_dir": str(upstream_dir), "ref": UPSTREAM_REF, "commit": commit}


def ensure_managed_model_directory(config: SetupConfig, logger: SetupLogger) -> dict[str, Any]:
    """Create/record the Modly-managed model directory without downloading weights."""

    config.model_dir.mkdir(parents=True, exist_ok=True)
    check_path = config.model_dir / DOWNLOAD_CHECK
    logger.info(
        "Model weights are managed by the Modly UI/downloader; "
        f"setup only records the model directory at {config.model_dir}"
    )
    return {
        "repo_id": HF_REPO,
        "model_dir": str(config.model_dir),
        "download_check": str(check_path),
        "download_check_exists": check_path.exists(),
        "weights_managed_by": "modly-ui",
        "message": "Model weights are managed by the Modly UI/downloader. Setup does not download weights.",
    }


def install_torch(config: SetupConfig, py: Path, logger: SetupLogger, tracker: StatusTracker) -> dict[str, Any]:
    lane = select_torch_lane(config.gpu_sm, config.cuda_version)
    tracker.data["torch_lane"] = lane.as_status()
    tracker.write()
    pip_install(py, logger, lane.packages, ["--index-url", lane.index_url])
    return lane.as_status()


def install_upstream_dependencies(py: Path, logger: SetupLogger) -> dict[str, Any]:
    provider_packages = {PYTORCH3D_UPSTREAM_PACKAGE, OPEN3D_PACKAGE}
    dependencies = [package for package in UPSTREAM_DEPENDENCIES if package != PYTORCH3D_UPSTREAM_PACKAGE and dependency_name(package) != OPEN3D_PACKAGE]
    pip_install(py, logger, dependencies)
    return {"packages": dependencies, "excluded": sorted(provider_packages)}


def resolve_pytorch3d_mode(config: SetupConfig, env: dict[str, str] | None = None) -> str:
    source_env = os.environ if env is None else env
    value = (
        config.payload.get("pytorch3d_mode")
        or config.payload.get("pytorch3dMode")
        or source_env.get("DREAMCUBE_PYTORCH3D_MODE")
        or "auto"
    )
    mode = str(value).strip().lower()
    if mode not in PYTORCH3D_MODES:
        raise SetupError(
            f"Invalid PyTorch3D provider mode {value!r}; expected one of {sorted(PYTORCH3D_MODES)}",
            code="invalid-pytorch3d-mode",
        )
    return mode


def source_build_supported_for_lane(lane: TorchLane) -> bool:
    return any(package == "torch==2.4.1" for package in lane.packages)


def probe_pytorch3d_provider(py: Path, logger: SetupLogger) -> dict[str, Any]:
    code = (
        "import json\n"
        "try:\n"
        "    import pytorch3d\n"
        "    from pytorch3d.transforms import matrix_to_quaternion\n"
        "    transforms = __import__('pytorch3d.transforms').transforms\n"
        "    print(json.dumps({\n"
        "        'importable': callable(matrix_to_quaternion),\n"
        "        'is_shim': bool(getattr(pytorch3d, '__dreamcube_shim__', False)),\n"
        "        'file': getattr(pytorch3d, '__file__', None),\n"
        "        'transforms_file': getattr(transforms, '__file__', None),\n"
        "    }))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'importable': False, 'error': f'{type(exc).__name__}: {exc}'}))\n"
        "    raise SystemExit(1)\n"
    )
    result = run_command([str(py), "-c", code], logger, check=False, code="pytorch3d-probe-failed")
    details: dict[str, Any] = {"importable": False, "returncode": result.returncode}
    for line in reversed((result.stdout or "").splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            details.update(parsed)
            break
    if result.returncode != 0:
        details["importable"] = False
    return details


def extension_site_packages(py: Path, logger: SetupLogger) -> Path:
    code = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    result = run_command([str(py), "-c", code], logger, code="site-packages-probe-failed")
    path = result.stdout.strip().splitlines()[-1]
    if not path:
        raise SetupError("Could not resolve extension venv site-packages", code="site-packages-probe-failed")
    return Path(path)


SHIM_TRANSFORMS_SOURCE = '"""Minimal transforms subset required by DreamCube."""\n\nimport torch\nimport torch.nn.functional as F\n\n\ndef _sqrt_positive_part(x):\n    result = torch.zeros_like(x)\n    positive = x > 0\n    result[positive] = torch.sqrt(x[positive])\n    return result\n\n\ndef matrix_to_quaternion(matrix):\n    """Convert rotation matrices to quaternions in (w, x, y, z) order."""\n    if matrix.size(-1) != 3 or matrix.size(-2) != 3:\n        raise ValueError(f"Invalid rotation matrix shape {tuple(matrix.shape)}; expected (..., 3, 3)")\n    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]\n    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]\n    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]\n    q_abs = _sqrt_positive_part(torch.stack([\n        1.0 + m00 + m11 + m22,\n        1.0 + m00 - m11 - m22,\n        1.0 - m00 + m11 - m22,\n        1.0 - m00 - m11 + m22,\n    ], dim=-1))\n    candidates = torch.stack([\n        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),\n        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),\n        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1),\n        torch.stack([m10 - m01, m20 + m02, m12 + m21, q_abs[..., 3] ** 2], dim=-1),\n    ], dim=-2)\n    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)\n    candidates = candidates / (2.0 * torch.maximum(q_abs[..., None], floor))\n    return candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0].reshape(matrix.shape[:-2] + (4,))\n'


def install_pytorch3d_shim(py: Path, logger: SetupLogger, *, mode: str, reason: str) -> dict[str, Any]:
    site_packages = extension_site_packages(py, logger)
    package_dir = site_packages / "pytorch3d"
    transforms_dir = package_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        '\n'.join([
            '"""DreamCube limited PyTorch3D compatibility provider."""',
            '__dreamcube_shim__ = True',
            '__version__ = "dreamcube-compat"',
            '',
        ]),
        encoding="utf-8",
    )
    (transforms_dir / "__init__.py").write_text(SHIM_TRANSFORMS_SOURCE, encoding="utf-8")
    probe = probe_pytorch3d_provider(py, logger)
    if not probe.get("importable"):
        raise SetupError(f"PyTorch3D shim did not import after installation: {probe}", code="pytorch3d-shim-failed")
    logger.info("Installed limited PyTorch3D compatibility provider for matrix_to_quaternion only; this is not full PyTorch3D.")
    return {
        "provider": "shim",
        "mode": mode,
        "reason": reason,
        "limited": True,
        "package_dir": str(package_dir),
        "probe": probe,
    }


def install_pytorch3d_provider(config: SetupConfig, py: Path, logger: SetupLogger, tracker: StatusTracker) -> dict[str, Any]:
    mode = resolve_pytorch3d_mode(config)
    lane = select_torch_lane(config.gpu_sm, config.cuda_version)
    existing = probe_pytorch3d_provider(py, logger)

    if mode == "required":
        if existing.get("importable") and not existing.get("is_shim"):
            details = {"provider": "installed", "mode": mode, "reason": "real PyTorch3D already importable", "probe": existing}
            tracker.data["pytorch3d_provider"] = details
            tracker.write()
            return details
        raise SetupError("PyTorch3D mode 'required' needs an already installed real pytorch3d.transforms provider", code="pytorch3d-required-missing")

    if mode == "shim":
        details = install_pytorch3d_shim(py, logger, mode=mode, reason="shim mode requested")
        tracker.data["pytorch3d_provider"] = details
        tracker.write()
        return details

    if existing.get("importable") and not existing.get("is_shim"):
        details = {"provider": "installed", "mode": mode, "reason": "real PyTorch3D already importable", "probe": existing}
        tracker.data["pytorch3d_provider"] = details
        tracker.write()
        return details

    if mode == "source" or source_build_supported_for_lane(lane):
        logger.info(f"Attempting real PyTorch3D source install from {PYTORCH3D_SOURCE_URL}")
        try:
            pip_install(py, logger, [PYTORCH3D_SOURCE_URL])
            probe = probe_pytorch3d_provider(py, logger)
            if probe.get("importable") and not probe.get("is_shim"):
                details = {
                    "provider": "source",
                    "mode": mode,
                    "reason": "real PyTorch3D installed from source",
                    "source_url": PYTORCH3D_SOURCE_URL,
                    "probe": probe,
                }
                tracker.data["pytorch3d_provider"] = details
                tracker.write()
                return details
            raise SetupError(f"Source install completed but real pytorch3d.transforms is not importable: {probe}", code="pytorch3d-source-import-failed")
        except SetupError:
            if mode == "source":
                raise
            logger.info("PyTorch3D source install failed in auto mode; falling back to limited compatibility shim.")
            details = install_pytorch3d_shim(
                py,
                logger,
                mode=mode,
                reason="source install failed in auto mode",
            )
            details["source_url"] = PYTORCH3D_SOURCE_URL
            tracker.data["pytorch3d_provider"] = details
            tracker.write()
            return details

    details = install_pytorch3d_shim(
        py,
        logger,
        mode=mode,
        reason=f"selected torch lane {lane.label} is outside PyTorch3D official source-install support used by this setup",
    )
    tracker.data["pytorch3d_provider"] = details
    tracker.write()
    return details


def resolve_open3d_mode(config: SetupConfig, env: dict[str, str] | None = None) -> str:
    source_env = os.environ if env is None else env
    value = (
        config.payload.get("open3d_mode")
        or config.payload.get("open3dMode")
        or source_env.get("DREAMCUBE_OPEN3D_MODE")
        or "auto"
    )
    mode = str(value).strip().lower()
    if mode not in OPEN3D_MODES:
        raise SetupError(
            f"Invalid Open3D provider mode {value!r}; expected one of {sorted(OPEN3D_MODES)}",
            code="invalid-open3d-mode",
        )
    return mode


def probe_open3d_provider(py: Path, logger: SetupLogger) -> dict[str, Any]:
    code = (
        "import json\n"
        "try:\n"
        "    import open3d as o3d\n"
        "    o3d.geometry.TriangleMesh()\n"
        "    o3d.geometry.PointCloud()\n"
        "    details = {\n"
        "        'importable': True,\n"
        "        'is_shim': bool(getattr(o3d, '__dreamcube_shim__', False)),\n"
        "        'file': getattr(o3d, '__file__', None),\n"
        "        'has_triangle_mesh': hasattr(o3d.geometry, 'TriangleMesh'),\n"
        "        'has_point_cloud': hasattr(o3d.geometry, 'PointCloud'),\n"
        "        'has_vector3d': hasattr(o3d.utility, 'Vector3dVector'),\n"
        "        'has_vector3i': hasattr(o3d.utility, 'Vector3iVector'),\n"
        "        'has_write_mesh': hasattr(o3d.io, 'write_triangle_mesh'),\n"
        "        'has_write_point_cloud': hasattr(o3d.io, 'write_point_cloud'),\n"
        "    }\n"
        "    details['importable'] = all(details[k] for k in [\n"
        "        'has_triangle_mesh', 'has_point_cloud', 'has_vector3d',\n"
        "        'has_vector3i', 'has_write_mesh', 'has_write_point_cloud'\n"
        "    ])\n"
        "    print(json.dumps(details))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'importable': False, 'error': f'{type(exc).__name__}: {exc}'}))\n"
        "    raise SystemExit(1)\n"
    )
    result = run_command([str(py), "-c", code], logger, check=False, code="open3d-probe-failed")
    details: dict[str, Any] = {"importable": False, "returncode": result.returncode}
    for line in reversed((result.stdout or "").splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            details.update(parsed)
            break
    if result.returncode != 0:
        details["importable"] = False
    return details


OPEN3D_INIT_SOURCE = '''# DreamCube limited Open3D compatibility provider.

__dreamcube_shim__ = True
__version__ = "dreamcube-compat"

from . import geometry, io, utility

__all__ = ["geometry", "io", "utility"]
'''

OPEN3D_GEOMETRY_SOURCE = '''# Minimal geometry subset required by DreamCube.


class TriangleMesh:
    def __init__(self):
        self.vertices = []
        self.triangles = []
        self.vertex_colors = []

    def remove_unreferenced_vertices(self):
        return self

    def remove_degenerate_triangles(self):
        return self


class PointCloud:
    def __init__(self):
        self.points = []
        self.colors = []
'''

OPEN3D_UTILITY_SOURCE = '''# Minimal vector helpers required by DreamCube.

import numpy as np


def Vector3dVector(values):
    return np.asarray(values, dtype=float)


def Vector3iVector(values):
    return np.asarray(values, dtype=np.int64)
'''

OPEN3D_IO_SOURCE = '''# Minimal Open3D I/O subset required by DreamCube.
# This shim writes OBJ meshes and ASCII PLY point clouds directly, and uses
# trimesh for additional mesh formats when available.

from pathlib import Path

import numpy as np


def _array(values, *, dtype=float, columns=3):
    arr = np.asarray(values, dtype=dtype)
    if arr.size == 0:
        return np.empty((0, columns), dtype=dtype)
    return arr.reshape((-1, columns))


def _write_obj(path, vertices, faces, colors):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# DreamCube Open3D compatibility OBJ\\n")
        for index, vertex in enumerate(vertices):
            color = colors[index] if index < len(colors) else None
            if color is None:
                handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\\n")
            else:
                handle.write(
                    f"v {vertex[0]} {vertex[1]} {vertex[2]} "
                    f"{color[0]} {color[1]} {color[2]}\\n"
                )
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\\n")


def _write_point_ply(path, points, colors):
    has_colors = len(colors) == len(points) and len(points) > 0
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\\nformat ascii 1.0\\n")
        handle.write(f"element vertex {len(points)}\\n")
        handle.write("property float x\\nproperty float y\\nproperty float z\\n")
        if has_colors:
            handle.write("property uchar red\\nproperty uchar green\\nproperty uchar blue\\n")
        handle.write("end_header\\n")
        for index, point in enumerate(points):
            if has_colors:
                color = np.clip(colors[index] * 255.0, 0, 255).astype(int)
                handle.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\\n")
            else:
                handle.write(f"{point[0]} {point[1]} {point[2]}\\n")


def _write_mesh_ply(path, vertices, faces, colors):
    has_colors = len(colors) == len(vertices) and len(vertices) > 0
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\\nformat ascii 1.0\\n")
        handle.write(f"element vertex {len(vertices)}\\n")
        handle.write("property float x\\nproperty float y\\nproperty float z\\n")
        if has_colors:
            handle.write("property uchar red\\nproperty uchar green\\nproperty uchar blue\\n")
        handle.write(f"element face {len(faces)}\\n")
        handle.write("property list uchar int vertex_indices\\nend_header\\n")
        for index, vertex in enumerate(vertices):
            if has_colors:
                color = np.clip(colors[index] * 255.0, 0, 255).astype(int)
                handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\\n")
            else:
                handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\\n")


def write_triangle_mesh(path, mesh):
    path = str(path)
    suffix = Path(path).suffix.lower()
    vertices = _array(getattr(mesh, "vertices", []), dtype=float)
    faces = _array(getattr(mesh, "triangles", []), dtype=np.int64)
    colors = _array(getattr(mesh, "vertex_colors", []), dtype=float)

    if suffix == ".obj":
        _write_obj(path, vertices, faces, colors)
        return True
    if suffix == ".ply":
        _write_mesh_ply(path, vertices, faces, colors)
        return True

    try:
        import trimesh

        visual = None
        if len(colors) == len(vertices) and len(vertices) > 0:
            visual = trimesh.visual.ColorVisuals(vertex_colors=np.clip(colors * 255.0, 0, 255).astype(np.uint8))
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        tm.export(path)
        return True
    except Exception as exc:
        raise RuntimeError(f"Open3D compatibility shim cannot write mesh format {suffix!r}: {exc}") from exc


def write_point_cloud(path, pcd):
    path = str(path)
    suffix = Path(path).suffix.lower()
    points = _array(getattr(pcd, "points", []), dtype=float)
    colors = _array(getattr(pcd, "colors", []), dtype=float)

    if suffix in {".ply", ""}:
        _write_point_ply(path, points, colors)
        return True
    if suffix == ".obj":
        _write_obj(path, points, np.empty((0, 3), dtype=np.int64), colors)
        return True
    raise RuntimeError(f"Open3D compatibility shim can only write point clouds as PLY or OBJ, got {suffix!r}")
'''


def install_open3d_shim(py: Path, logger: SetupLogger, *, mode: str, reason: str) -> dict[str, Any]:
    site_packages = extension_site_packages(py, logger)
    package_dir = site_packages / "open3d"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(OPEN3D_INIT_SOURCE, encoding="utf-8")
    (package_dir / "geometry.py").write_text(OPEN3D_GEOMETRY_SOURCE, encoding="utf-8")
    (package_dir / "utility.py").write_text(OPEN3D_UTILITY_SOURCE, encoding="utf-8")
    (package_dir / "io.py").write_text(OPEN3D_IO_SOURCE, encoding="utf-8")
    probe = probe_open3d_provider(py, logger)
    if not probe.get("importable"):
        raise SetupError(f"Open3D shim did not import after installation: {probe}", code="open3d-shim-failed")
    logger.info("Installed limited Open3D compatibility provider for DreamCube pano_to_3d I/O only; this is not full Open3D.")
    return {"provider": "shim", "mode": mode, "limited": True, "reason": reason, "package_dir": str(package_dir), "probe": probe}


def install_open3d_provider(config: SetupConfig, py: Path, logger: SetupLogger, tracker: StatusTracker) -> dict[str, Any]:
    mode = resolve_open3d_mode(config)
    existing = probe_open3d_provider(py, logger)

    if mode == "required":
        if existing.get("importable") and not existing.get("is_shim"):
            details = {"provider": "installed", "mode": mode, "limited": False, "reason": "real Open3D already importable", "probe": existing}
            tracker.data["open3d_provider"] = details
            tracker.write()
            return details
        raise SetupError("Open3D mode 'required' needs an already installed real open3d provider", code="open3d-required-missing")

    if mode == "shim":
        details = install_open3d_shim(py, logger, mode=mode, reason="shim mode requested")
        tracker.data["open3d_provider"] = details
        tracker.write()
        return details

    if existing.get("importable") and not existing.get("is_shim"):
        details = {"provider": "installed", "mode": mode, "limited": False, "reason": "real Open3D already importable", "probe": existing}
        tracker.data["open3d_provider"] = details
        tracker.write()
        return details

    if mode in {"auto", "wheel"}:
        attempted_indexes: list[str | None] = []
        for extra_args, index_url in ((None, None), (["-f", OPEN3D_DOCS_WHEEL_INDEX], OPEN3D_DOCS_WHEEL_INDEX)):
            attempted_indexes.append(index_url)
            try:
                pip_install(py, logger, [OPEN3D_PACKAGE], extra_args)
                probe = probe_open3d_provider(py, logger)
                if probe.get("importable") and not probe.get("is_shim"):
                    details = {"provider": "wheel", "mode": mode, "limited": False, "reason": "real Open3D installed from wheel", "package": OPEN3D_PACKAGE, "index_url": index_url, "attempted_indexes": attempted_indexes, "probe": probe}
                    tracker.data["open3d_provider"] = details
                    tracker.write()
                    return details
                raise SetupError(f"Wheel install completed but real Open3D is not importable: {probe}", code="open3d-wheel-import-failed")
            except SetupError as exc:
                if mode == "wheel" and index_url == OPEN3D_DOCS_WHEEL_INDEX:
                    raise
                logger.info(f"Open3D wheel install attempt failed in {mode} mode: {exc}")

        details = install_open3d_shim(py, logger, mode=mode, reason="wheel install failed in auto mode; source builds are not attempted automatically")
        details["package"] = OPEN3D_PACKAGE
        details["attempted_indexes"] = attempted_indexes
        tracker.data["open3d_provider"] = details
        tracker.write()
        return details

    if mode == "source":
        logger.info(f"Attempting real Open3D source install from {OPEN3D_SOURCE_URL}")
        pip_install(py, logger, [OPEN3D_SOURCE_URL])
        probe = probe_open3d_provider(py, logger)
        if probe.get("importable") and not probe.get("is_shim"):
            details = {"provider": "source", "mode": mode, "limited": False, "reason": "real Open3D installed from source", "source_url": OPEN3D_SOURCE_URL, "probe": probe}
            tracker.data["open3d_provider"] = details
            tracker.write()
            return details
        raise SetupError(f"Source install completed but real Open3D is not importable: {probe}", code="open3d-source-import-failed")

    raise SetupError(f"Unhandled Open3D provider mode: {mode}", code="invalid-open3d-mode")

def probe_imports(py: Path, upstream_dir: Path, logger: SetupLogger) -> dict[str, Any]:
    code = (
        "import importlib\n"
        "import os\n"
        "import sys\n"
        "upstream = os.environ['DREAMCUBE_UPSTREAM_DIR']\n"
        "if upstream not in sys.path:\n"
        "    sys.path.insert(0, upstream)\n"
        f"modules = {PROBE_IMPORTS!r}\n"
        "failures = []\n"
        "for name in modules:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        failures.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "if failures:\n"
        "    print('Import probe failures:')\n"
        "    for failure in failures:\n"
        "        print(f'  - {failure}')\n"
        "    raise SystemExit(1)\n"
        "print('Import probes passed: ' + ', '.join(modules))\n"
    )
    pythonpath = str(upstream_dir)
    if os.environ.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + os.environ["PYTHONPATH"]
    run_command(
        [str(py), "-c", code],
        logger,
        env={"DREAMCUBE_UPSTREAM_DIR": str(upstream_dir), "PYTHONPATH": pythonpath},
        code="import-probe-failed",
    )
    return {"imports": PROBE_IMPORTS, "upstream_dir": str(upstream_dir)}


def run_setup(config: SetupConfig) -> int:
    log_path = config.ext_dir / LOG_RELATIVE_PATH
    logger = SetupLogger(log_path)
    tracker = StatusTracker(config, log_path)
    py = venv_python(config.ext_dir / "venv")
    upstream_dir = config.ext_dir / UPSTREAM_RELATIVE_PATH

    try:
        logger.info("Starting DreamCube setup")
        phase(tracker, logger, "preflight", "Validate payload, manifest, Python, and git", lambda: preflight(config, logger))
        phase(tracker, logger, "venv", "Create or reuse extension-local venv", lambda: ensure_venv(config, logger))
        phase(tracker, logger, "bootstrap", "Install pip/setuptools/wheel/huggingface_hub", lambda: bootstrap_venv(py, logger))
        upstream_details = phase(tracker, logger, "upstream-source", "Clone or update DreamCube upstream source", lambda: sync_upstream(config, logger))
        tracker.data["upstream_commit"] = upstream_details.get("commit")
        tracker.write()
        phase(
            tracker,
            logger,
            "model-directory",
            "Record Modly-managed model directory",
            lambda: ensure_managed_model_directory(config, logger),
        )
        phase(tracker, logger, "torch", "Install selected PyTorch lane", lambda: install_torch(config, py, logger, tracker))
        phase(tracker, logger, "dependencies", "Install DreamCube runtime dependencies except provider-managed packages", lambda: install_upstream_dependencies(py, logger))
        phase(
            tracker,
            logger,
            "open3d-provider",
            "Install or select real Open3D, or a limited DreamCube compatibility shim",
            lambda: install_open3d_provider(config, py, logger, tracker),
        )
        phase(
            tracker,
            logger,
            "pytorch3d-provider",
            "Install or select real PyTorch3D, or a limited matrix_to_quaternion compatibility shim",
            lambda: install_pytorch3d_provider(config, py, logger, tracker),
        )
        phase(tracker, logger, "import-probes", "Probe runtime and upstream imports", lambda: probe_imports(py, upstream_dir, logger))
        phase(tracker, logger, "finalize", "Mark setup ready", lambda: {"status": "ready"})
        tracker.mark_ready()
        logger.info("DreamCube setup completed successfully")
        return 0
    except Exception as exc:
        message = redact_text(str(exc))
        code = exc.code if isinstance(exc, SetupError) else "unexpected-setup-error"
        logger.info(f"ERROR: {message}")
        tracker.fail_current_phase(message)
        tracker.add_blocker(code, message)
        tracker.mark_failed()
        return 1
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        if config.validate_only:
            details = validate_internal_config(config)
            print(f"[setup:{EXTENSION_ID}] validate-only OK: {json.dumps(details, sort_keys=True)}", flush=True)
            return 0
        return run_setup(config)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[setup:{EXTENSION_ID}] ERROR: {redact_text(str(exc))}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
