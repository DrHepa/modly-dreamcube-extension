#!/usr/bin/env python3
"""Lightweight validation for the Modly DreamCube extension.

This script intentionally uses only the Python standard library and never imports
DreamCube, Torch, or other heavy runtime dependencies. It is safe to run before
extension setup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"

GENERATED_COMMON_PARAM_IDS = {
    "depth_mode",
    "auto_depth_variant",
    "depth_image_path",
    "save_input_depth",
    "prompt_front",
    "prompt_right",
    "prompt_back",
    "prompt_left",
    "prompt_top",
    "prompt_bottom",
    "num_inference_steps",
    "guidance_scale",
    "normalize_scale",
    "pano_to_3d_mode",
    "max_equi_size",
    "max_cube_size",
    "save_all_outputs",
    "seed",
}


MANUAL_IMAGE_PICKER_IDS = {
    "rgb_right_path",
    "rgb_back_path",
    "rgb_left_path",
    "rgb_top_path",
    "rgb_bottom_path",
    "depth_front_path",
    "depth_right_path",
    "depth_back_path",
    "depth_left_path",
    "depth_top_path",
    "depth_bottom_path",
}

MANUAL_FORBIDDEN_PARAM_IDS = {
    "depth_mode",
    "auto_depth_variant",
    "depth_image_path",
    "save_input_depth",
    "pano_to_3d_mode",
    "max_equi_size",
    "output_format",
}

PROMPT_PARAM_IDS = {
    "prompt_front",
    "prompt_right",
    "prompt_back",
    "prompt_left",
    "prompt_top",
    "prompt_bottom",
}


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def expect_equal(self, path: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            self.fail(f"{path}: expected {expected!r}, got {actual!r}")

    def expect_truthy(self, path: str, actual: Any) -> None:
        if not actual:
            self.fail(f"{path}: expected a non-empty value")


def load_manifest(validator: Validator) -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        validator.fail(f"missing manifest.json at {MANIFEST_PATH}")
        return None

    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        validator.fail(f"manifest.json is invalid JSON: {exc}")
        return None

    if not isinstance(data, dict):
        validator.fail("manifest.json must contain a JSON object")
        return None

    return data


def option_values(param: dict[str, Any]) -> set[str]:
    options = param.get("options")
    if not isinstance(options, list):
        return set()

    values: set[str] = set()
    for option in options:
        if isinstance(option, dict) and isinstance(option.get("value"), str):
            values.add(option["value"])
    return values


def params_by_id(node: dict[str, Any], validator: Validator, node_id: str) -> dict[str, dict[str, Any]]:
    params = node.get("params_schema")
    if not isinstance(params, list) or not params:
        validator.fail(f"nodes[{node_id}].params_schema: expected a non-empty list")
        return {}

    result: dict[str, dict[str, Any]] = {}
    for index, param in enumerate(params):
        if not isinstance(param, dict):
            validator.fail(f"nodes[{node_id}].params_schema[{index}]: expected an object")
            continue

        param_id = param.get("id")
        if not isinstance(param_id, str) or not param_id:
            validator.fail(f"nodes[{node_id}].params_schema[{index}].id: expected a non-empty string")
            continue

        if param_id in result:
            validator.fail(f"nodes[{node_id}].params_schema: duplicate parameter id {param_id!r}")
        result[param_id] = param

    return result


def validate_common_params(validator: Validator, node_id: str, params: dict[str, dict[str, Any]]) -> None:
    missing = sorted(GENERATED_COMMON_PARAM_IDS.difference(params))
    if missing:
        validator.fail(f"nodes[{node_id}].params_schema: missing params {', '.join(missing)}")
        return

    depth = params["depth_image_path"]
    depth_mode = params["depth_mode"]
    validator.expect_equal(f"nodes[{node_id}].depth_mode.type", depth_mode.get("type"), "select")
    validator.expect_equal(f"nodes[{node_id}].depth_mode.default", depth_mode.get("default"), "auto")
    missing_depth_modes = sorted({"auto", "manual"}.difference(option_values(depth_mode)))
    if missing_depth_modes:
        validator.fail(f"nodes[{node_id}].depth_mode.options: missing {', '.join(missing_depth_modes)}")

    auto_variant = params["auto_depth_variant"]
    validator.expect_equal(f"nodes[{node_id}].auto_depth_variant.type", auto_variant.get("type"), "select")
    validator.expect_equal(f"nodes[{node_id}].auto_depth_variant.default", auto_variant.get("default"), "vits")
    if "vits" not in option_values(auto_variant):
        validator.fail(f"nodes[{node_id}].auto_depth_variant.options: missing vits")

    depth = params["depth_image_path"]
    validator.expect_equal(f"nodes[{node_id}].depth_image_path.type", depth.get("type"), "string")
    validator.expect_equal(f"nodes[{node_id}].depth_image_path.required", depth.get("required"), False)
    validator.expect_equal(f"nodes[{node_id}].depth_image_path.pickerIntent", depth.get("pickerIntent"), "image")
    tooltip = str(depth.get("tooltip", ""))
    if "modly-depth-anything" not in tooltip or "supplied" not in tooltip.lower():
        validator.warn(f"nodes[{node_id}].depth_image_path.tooltip should document internal auto-depth and supplied-path override")

    filters = depth.get("filters")
    if not isinstance(filters, list) or not filters:
        validator.fail(f"nodes[{node_id}].depth_image_path.filters: expected image filters")

    for param_id in sorted(PROMPT_PARAM_IDS):
        param = params[param_id]
        validator.expect_equal(f"nodes[{node_id}].{param_id}.type", param.get("type"), "string")
        validator.expect_equal(f"nodes[{node_id}].{param_id}.required", param.get("required"), True)

    numeric_expectations = {
        "num_inference_steps": ("int", 50),
        "guidance_scale": ("float", 7.5),
        "normalize_scale": ("float", 0.6),
        "max_equi_size": ("int", 1024),
        "max_cube_size": ("int", 256),
        "seed": ("int", -1),
    }
    for param_id, (expected_type, expected_default) in numeric_expectations.items():
        param = params[param_id]
        validator.expect_equal(f"nodes[{node_id}].{param_id}.type", param.get("type"), expected_type)
        validator.expect_equal(f"nodes[{node_id}].{param_id}.default", param.get("default"), expected_default)

    pano_mode = params["pano_to_3d_mode"]
    validator.expect_equal(f"nodes[{node_id}].pano_to_3d_mode.type", pano_mode.get("type"), "select")
    required_modes = {"3D from RGB-D Cubemap", "3D from RGB-D Equirectangular"}
    missing_modes = sorted(required_modes.difference(option_values(pano_mode)))
    if missing_modes:
        validator.fail(f"nodes[{node_id}].pano_to_3d_mode.options: missing {', '.join(missing_modes)}")

    save_input_depth = params["save_input_depth"]
    validator.expect_equal(f"nodes[{node_id}].save_input_depth.type", save_input_depth.get("type"), "select")
    validator.expect_equal(f"nodes[{node_id}].save_input_depth.default", save_input_depth.get("default"), "true")
    missing_input_depth_values = sorted({"true", "false"}.difference(option_values(save_input_depth)))
    if missing_input_depth_values:
        validator.fail(f"nodes[{node_id}].save_input_depth.options: missing {', '.join(missing_input_depth_values)}")

    save_all = params["save_all_outputs"]
    validator.expect_equal(f"nodes[{node_id}].save_all_outputs.type", save_all.get("type"), "select")
    missing_save_values = sorted({"true", "false"}.difference(option_values(save_all)))
    if missing_save_values:
        validator.fail(f"nodes[{node_id}].save_all_outputs.options: missing {', '.join(missing_save_values)}")



def validate_manual_params(validator: Validator, node_id: str, params: dict[str, dict[str, Any]]) -> None:
    missing_pickers = sorted(MANUAL_IMAGE_PICKER_IDS.difference(params))
    if missing_pickers:
        validator.fail(f"nodes[{node_id}].params_schema: missing manual image pickers {', '.join(missing_pickers)}")
    for param_id in sorted(MANUAL_IMAGE_PICKER_IDS):
        param = params.get(param_id)
        if not param:
            continue
        validator.expect_equal(f"nodes[{node_id}].{param_id}.type", param.get("type"), "string")
        validator.expect_equal(f"nodes[{node_id}].{param_id}.required", param.get("required"), True)
        validator.expect_equal(f"nodes[{node_id}].{param_id}.pickerIntent", param.get("pickerIntent"), "image")

    forbidden = sorted(MANUAL_FORBIDDEN_PARAM_IDS.intersection(params))
    if forbidden:
        validator.fail(f"nodes[{node_id}].params_schema: manual node must omit {', '.join(forbidden)}")

    for param_id in sorted(PROMPT_PARAM_IDS):
        param = params.get(param_id)
        if not param:
            validator.fail(f"nodes[{node_id}].params_schema: missing {param_id}")
            continue
        validator.expect_equal(f"nodes[{node_id}].{param_id}.type", param.get("type"), "string")
        validator.expect_equal(f"nodes[{node_id}].{param_id}.required", param.get("required"), True)

    numeric_expectations = {
        "num_inference_steps": ("int", 50, 100),
        "guidance_scale": ("float", 7.5, 20.0),
        "normalize_scale": ("float", 0.6, 5.0),
        "max_cube_size": ("int", 512, 512, 256),
        "seed": ("int", -1, 2147483647),
    }
    for param_id, expectation in numeric_expectations.items():
        expected_type, expected_default, expected_max = expectation[:3]
        expected_min = expectation[3] if len(expectation) > 3 else None
        param = params.get(param_id)
        if not param:
            validator.fail(f"nodes[{node_id}].params_schema: missing {param_id}")
            continue
        validator.expect_equal(f"nodes[{node_id}].{param_id}.type", param.get("type"), expected_type)
        validator.expect_equal(f"nodes[{node_id}].{param_id}.default", param.get("default"), expected_default)
        validator.expect_equal(f"nodes[{node_id}].{param_id}.max", param.get("max"), expected_max)
        if expected_min is not None:
            validator.expect_equal(f"nodes[{node_id}].{param_id}.min", param.get("min"), expected_min)

    save_all = params.get("save_all_outputs")
    if not save_all:
        validator.fail(f"nodes[{node_id}].params_schema: missing save_all_outputs")
    else:
        validator.expect_equal(f"nodes[{node_id}].save_all_outputs.default", save_all.get("default"), "true")

    for param_id, default in {"mesh_depth_jump_threshold": 0.20, "mesh_footprint_ratio_threshold": 12, "mesh_aspect_ratio_threshold": 10}.items():
        param = params.get(param_id)
        if not param:
            validator.fail(f"nodes[{node_id}].params_schema: missing {param_id}")
            continue
        validator.expect_equal(f"nodes[{node_id}].{param_id}.type", param.get("type"), "float")
        validator.expect_equal(f"nodes[{node_id}].{param_id}.default", param.get("default"), default)

def validate_node(
    validator: Validator,
    manifest: dict[str, Any],
    node: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    node_id = expected["id"]
    validator.expect_equal(f"nodes[{node_id}].input", node.get("input"), expected["input"])
    validator.expect_equal(f"nodes[{node_id}].output", node.get("output"), expected["output"])
    validator.expect_equal(f"nodes[{node_id}].capability_id", node.get("capability_id"), expected["capability_id"])
    validator.expect_equal(f"nodes[{node_id}].weight_owner_id", node.get("weight_owner_id"), manifest.get("weight_owner_id"))
    validator.expect_equal(f"nodes[{node_id}].hf_repo", node.get("hf_repo"), manifest.get("hf_repo"))
    validator.expect_equal(f"nodes[{node_id}].download_check", node.get("download_check"), manifest.get("download_check"))

    params = params_by_id(node, validator, node_id)
    if not params:
        return

    if node_id == "generate-scene-manual-cubemap":
        validate_manual_params(validator, node_id, params)
        return

    validate_common_params(validator, node_id, params)
    output_format = params.get("output_format")
    if expected["output_format_required"]:
        if not output_format:
            validator.fail(f"nodes[{node_id}].params_schema: missing output_format")
        else:
            validator.expect_equal(f"nodes[{node_id}].output_format.type", output_format.get("type"), "select")
            validator.expect_equal(f"nodes[{node_id}].output_format.default", output_format.get("default"), expected["output_default"])
            missing_outputs = sorted(set(expected["output_options"]).difference(option_values(output_format)))
            if missing_outputs:
                validator.fail(f"nodes[{node_id}].output_format.options: missing {', '.join(missing_outputs)}")
    elif output_format:
        validator.fail(
            f"nodes[{node_id}].params_schema: output_format must be omitted; "
            "scene generation returns a GLB mesh; scene-manifest.json is an auxiliary sidecar"
        )

    validator.expect_equal(f"nodes[{node_id}].max_equi_size.max", params["max_equi_size"].get("max"), 2048)
    validator.expect_equal(f"nodes[{node_id}].max_cube_size.max", params["max_cube_size"].get("max"), 512)

    if node_id == "generate-scene":
        threshold = params.get("mesh_depth_jump_threshold")
        if not threshold:
            validator.fail("nodes[generate-scene].params_schema: missing mesh_depth_jump_threshold")
        else:
            validator.expect_equal("nodes[generate-scene].mesh_depth_jump_threshold.type", threshold.get("type"), "float")
            validator.expect_equal("nodes[generate-scene].mesh_depth_jump_threshold.default", threshold.get("default"), 0.20)
            validator.expect_equal("nodes[generate-scene].mesh_depth_jump_threshold.min", threshold.get("min"), 0)
            validator.expect_equal("nodes[generate-scene].mesh_depth_jump_threshold.max", threshold.get("max"), 5)
            validator.expect_equal("nodes[generate-scene].mesh_depth_jump_threshold.step", threshold.get("step"), 0.05)
            tooltip = str(threshold.get("tooltip", "")).lower()
            if "depth" not in tooltip or "0" not in tooltip or "disable" not in tooltip:
                validator.fail("nodes[generate-scene].mesh_depth_jump_threshold.tooltip must explain depth jumps and zero disabling filtering")
        geometry_thresholds = {
            "mesh_footprint_ratio_threshold": 12,
            "mesh_aspect_ratio_threshold": 10,
        }
        for param_id, default in geometry_thresholds.items():
            param = params.get(param_id)
            if not param:
                validator.fail(f"nodes[generate-scene].params_schema: missing {param_id}")
                continue
            validator.expect_equal(f"nodes[generate-scene].{param_id}.type", param.get("type"), "float")
            validator.expect_equal(f"nodes[generate-scene].{param_id}.default", param.get("default"), default)
            validator.expect_equal(f"nodes[generate-scene].{param_id}.min", param.get("min"), 0)
            validator.expect_equal(f"nodes[generate-scene].{param_id}.advanced", param.get("advanced"), True)
            tooltip = str(param.get("tooltip", "")).lower()
            if "0" not in tooltip or "disable" not in tooltip:
                validator.fail(f"nodes[generate-scene].{param_id}.tooltip must explain zero disabling filtering")
    elif "mesh_depth_jump_threshold" in params:
        validator.fail(f"nodes[{node_id}].params_schema: mesh_depth_jump_threshold is scene-only")

    save_default = params["save_all_outputs"].get("default")
    validator.expect_equal(f"nodes[{node_id}].save_all_outputs.default", save_default, expected["save_all_default"])


def validate_files(validator: Validator, manifest: dict[str, Any]) -> None:
    setup = manifest.get("setup")
    if setup is not None and not (ROOT / "setup.py").is_file():
        validator.fail("setup.py is required because manifest.setup is present")
    elif (ROOT / "setup.py").exists() and not (ROOT / "setup.py").is_file():
        validator.fail("setup.py exists but is not a regular file")

    required_runtime_payloads = {
        "dreamcube_mesh.py": "extension-owned mesh converter",
        "dreamcube_manual_cubemap.py": "extension-owned manual RGB-D cubemap runtime",
    }
    for filename, description in required_runtime_payloads.items():
        if not (ROOT / filename).is_file():
            validator.fail(f"{filename} is required as the {description}")

    generator_path = ROOT / "generator.py"
    if generator_path.exists() and not generator_path.is_file():
        validator.fail("generator.py exists but is not a regular file")
    elif generator_path.is_file():
        text = generator_path.read_text(encoding="utf-8", errors="replace")
        generator_class = manifest.get("generator_class")
        if isinstance(generator_class, str) and generator_class not in text:
            validator.fail(f"generator.py does not contain manifest generator_class {generator_class!r}")
    else:
        validator.warn("generator.py is not present; skipped generator class text check")


def validate_manifest(validator: Validator, manifest: dict[str, Any]) -> None:
    root_expectations = {
        "id": "dreamcube",
        "type": "model",
        "generator_class": "DreamCubeGenerator",
        "source": "https://github.com/DrHepa/modly-dreamcube-extension",
        "hf_repo": "KevinHuang/DreamCube",
        "download_check": "model_index.json",
        "weight_owner_id": "dreamcube",
    }
    for key, expected in root_expectations.items():
        validator.expect_equal(key, manifest.get(key), expected)

    validator.expect_equal("version", manifest.get("version"), "0.2.0")

    for key in ("name", "displayName", "description"):
        validator.expect_truthy(key, manifest.get(key))

    license_info = manifest.get("license")
    if not isinstance(license_info, dict):
        validator.fail("license: expected an object")
    else:
        validator.expect_equal("license.upstream", license_info.get("upstream"), "Apache-2.0")
        validator.expect_equal("license.weights", license_info.get("weights"), "Apache-2.0")

    setup = manifest.get("setup")
    if not isinstance(setup, dict):
        validator.fail("setup: expected an object")
    else:
        validator.expect_equal("setup.classification", setup.get("classification"), "model-managed-setup")
        argv_contract = setup.get("argv_contract")
        if not isinstance(argv_contract, dict):
            validator.fail("setup.argv_contract: expected an object")
        else:
            electron = str(argv_contract.get("electron", ""))
            if "setup.py" not in electron or "python_exe" not in electron or "ext_dir" not in electron:
                validator.fail("setup.argv_contract.electron must document setup.py JSON argv with python_exe and ext_dir")

        runtime_cache = setup.get("managed_runtime_cache")
        if not isinstance(runtime_cache, dict):
            validator.fail("setup.managed_runtime_cache: expected an object")
        else:
            upstream_source = runtime_cache.get("upstream_source")
            if not isinstance(upstream_source, dict):
                validator.fail("setup.managed_runtime_cache.upstream_source: expected an object")
            else:
                validator.expect_equal("setup.managed_runtime_cache.upstream_source.repo_url", upstream_source.get("repo_url"), "https://github.com/Yukun-Huang/DreamCube.git")
                validator.expect_equal("setup.managed_runtime_cache.upstream_source.ref", upstream_source.get("ref"), "main")
                validator.expect_equal("setup.managed_runtime_cache.upstream_source.commit", upstream_source.get("commit"), "aa04a53c6542581b5b0a6faa575865d2d57b5243")
                validator.expect_equal("setup.managed_runtime_cache.upstream_source.checkout", upstream_source.get("checkout"), "detached")
                validator.expect_equal("setup.managed_runtime_cache.upstream_source.pinned_revision", upstream_source.get("pinned_revision"), "aa04a53c6542581b5b0a6faa575865d2d57b5243")
            hf_snapshot = runtime_cache.get("hf_snapshot")
            if isinstance(hf_snapshot, dict):
                validator.expect_equal("setup.managed_runtime_cache.hf_snapshot.managed_by_setup", hf_snapshot.get("managed_by_setup"), False)
            auto_depth = runtime_cache.get("auto_depth")
            if not isinstance(auto_depth, dict):
                validator.fail("setup.managed_runtime_cache.auto_depth: expected an object")
            else:
                validator.expect_equal("setup.managed_runtime_cache.auto_depth.repo_id", auto_depth.get("repo_id"), "depth-anything/Depth-Anything-V2-Small-hf")
                validator.expect_equal("setup.managed_runtime_cache.auto_depth.default_variant", auto_depth.get("default_variant"), "vits")
                validator.expect_equal("setup.managed_runtime_cache.auto_depth.managed_by_setup", auto_depth.get("managed_by_setup"), False)

    docs = manifest.get("docs")
    if not isinstance(docs, dict):
        validator.fail("docs: expected an object")
    else:
        validator.expect_equal("docs.readme", docs.get("readme"), "./README.md")
        validator.expect_equal("docs.setup_status", docs.get("setup_status"), ".modly/setup/setup-status.json")
        validator.expect_equal("docs.setup_log", docs.get("setup_log"), ".modly/setup/logs/setup.log")

    nodes = manifest.get("nodes")
    if not isinstance(nodes, list):
        validator.fail("nodes: expected a list")
        return

    node_by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            validator.fail(f"nodes[{index}]: expected an object")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            validator.fail(f"nodes[{index}].id: expected a non-empty string")
            continue
        if node_id in node_by_id:
            validator.fail(f"nodes: duplicate id {node_id!r}")
        node_by_id[node_id] = node

    expected_nodes = {
        "generate-panorama": {
            "id": "generate-panorama",
            "input": "image",
            "output": "image",
            "capability_id": "image-to-panorama",
            "output_format_required": True,
            "output_default": "equirect_rgb_png",
            "output_options": ["equirect_rgb_png"],
            "save_all_default": "false",
        },
        "generate-scene": {
            "id": "generate-scene",
            "input": "image",
            "output": "mesh",
            "capability_id": "image-depth-to-scene",
            "output_format_required": False,
            "save_all_default": "true",
        },
        "generate-scene-manual-cubemap": {
            "id": "generate-scene-manual-cubemap",
            "input": "image",
            "output": "mesh",
            "capability_id": "rgbd-cubemap-to-scene",
            "output_format_required": False,
            "save_all_default": "true",
        },
    }

    missing_nodes = sorted(set(expected_nodes).difference(node_by_id))
    if missing_nodes:
        validator.fail(f"nodes: missing required node ids {', '.join(missing_nodes)}")

    for node_id, expected in expected_nodes.items():
        node = node_by_id.get(node_id)
        if node:
            validate_node(validator, manifest, node, expected)


def main() -> int:
    validator = Validator()
    manifest = load_manifest(validator)
    if manifest is not None:
        validate_manifest(validator, manifest)
        validate_files(validator, manifest)

    if validator.errors:
        print("DreamCube extension validation failed:", file=sys.stderr)
        for message in validator.errors:
            print(f"  - {message}", file=sys.stderr)
        if validator.warnings:
            print("Warnings:", file=sys.stderr)
            for message in validator.warnings:
                print(f"  - {message}", file=sys.stderr)
        return 1

    print("DreamCube extension validation passed.")
    if validator.warnings:
        print("Warnings:")
        for message in validator.warnings:
            print(f"  - {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
