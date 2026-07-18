from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


EXT_DIR = Path(__file__).resolve().parents[1]
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rgb_bytes(value: int, size: int = 6) -> bytes:
    image = Image.new("RGB", (size, size), (value, value // 2, 255 - value))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def png_header(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


class ManifestAndRuntimeTests(unittest.TestCase):
    def test_legacy_depth_contract_and_manual_node_shape(self):
        manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "0.3.0")
        nodes = {node["id"]: node for node in manifest["nodes"]}
        forbidden_node_keys = {"io" + "_" + "contract", "inputs", "outputs"}

        self.assertEqual(
            tuple(nodes),
            (
                "generate-panorama",
                "generate-scene",
                "estimate-cubemap-depths",
                "generate-scene-manual-cubemap",
            ),
        )

        depth = nodes["estimate-cubemap-depths"]
        self.assertFalse(forbidden_node_keys.intersection(depth))
        self.assertEqual(
            {item["id"] for item in depth["params_schema"]},
            {
                "rgb_right_path",
                "rgb_back_path",
                "rgb_left_path",
                "rgb_top_path",
                "rgb_bottom_path",
            },
        )
        for face in ("right", "back", "left", "top", "bottom"):
            key = f"rgb_{face}_path"
            param = next(item for item in depth["params_schema"] if item["id"] == key)
            self.assertEqual(param["type"], "string")
            self.assertFalse(param["required"])
            self.assertEqual(param["default"], "")
            self.assertEqual(param["pickerIntent"], "image")

        manual = nodes["generate-scene-manual-cubemap"]
        self.assertFalse(forbidden_node_keys.intersection(manual))
        manual_param_ids = {item["id"] for item in manual["params_schema"]}
        for key in (
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
            "prompt_front",
            "prompt_right",
            "prompt_back",
            "prompt_left",
            "prompt_top",
            "prompt_bottom",
            "num_inference_steps",
            "guidance_scale",
            "normalize_scale",
            "max_cube_size",
            "mesh_depth_jump_threshold",
            "mesh_footprint_ratio_threshold",
            "mesh_aspect_ratio_threshold",
            "save_all_outputs",
            "seed",
        ):
            self.assertIn(key, manual_param_ids)

        self.assertFalse(forbidden_node_keys.intersection(nodes["generate-panorama"]))
        self.assertFalse(forbidden_node_keys.intersection(nodes["generate-scene"]))

    def test_readme_documents_legacy_sparse_semantics_and_lazy_weights(self):
        readme = (EXT_DIR / "README.md").read_text(encoding="utf-8")
        for text in (
            "`front` RGB input is required",
            "estimated and non-metric",
            "select `front_depth.png` and the optional face-depth sidecars",
            ".modly/auto-depth/cache",
            "Setup is dependency-only: it never downloads",
            "Depth quality directly affects the final mesh.",
        ):
            self.assertIn(text, readme)

    def test_setup_is_dependency_only_and_has_no_weight_download_api(self):
        manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        auto_depth = manifest["setup"]["managed_runtime_cache"]["auto_depth"]
        self.assertFalse(auto_depth["managed_by_setup"])
        self.assertFalse(auto_depth["setup_downloads_weights"])
        setup_source = (EXT_DIR / "setup.py").read_text(encoding="utf-8")
        for token in ("from_pretrained(", "snapshot_download(", "hf_hub_download("):
            self.assertNotIn(token, setup_source)

    def test_runtime_has_no_removed_band_or_panorama_anchor_path(self):
        runtime_text = "\n".join(
            (EXT_DIR / filename).read_text(encoding="utf-8")
            for filename in (
                "generator.py",
                "dreamcube_cubemap_depth.py",
                "dreamcube_manual_cubemap.py",
                "manifest.json",
            )
        ).lower()
        forbidden = (
            "seam_refine_width_px",
            "blend_panorama_seam_anchor",
            "infer_panorama_relative_depth",
            "rgb_edge_mask",
            "32-pixel seam",
        )
        for token in forbidden:
            self.assertNotIn(token, runtime_text)


class CubemapDepthPostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.depth = load_module(
            "dreamcube_cubemap_depth_contract_test",
            EXT_DIR / "dreamcube_cubemap_depth.py",
        )
        cls.manual = load_module(
            "dreamcube_manual_cubemap_depth_contract_test",
            EXT_DIR / "dreamcube_manual_cubemap.py",
        )

    @staticmethod
    def predictions(faces: tuple[str, ...], size: int = 8) -> dict[str, np.ndarray]:
        yy, xx = np.indices((size, size), dtype=np.float64)
        values = {}
        for index, face in enumerate(faces):
            values[face] = (xx * (index + 1.0)) + yy + (index * 0.25)
        return values

    def test_joint_postprocess_finite_positive_affine_and_singleton_invariants(self):
        predictions = self.predictions(("front", "right", "top"))
        predictions["front"][0, 0] = np.nan
        predictions["right"][1, 1] = np.inf
        result = self.depth.postprocess_cubemap_depths(predictions)

        self.assertEqual(tuple(result.depths_mm), ("front", "right", "top"))
        for array in result.depths_mm.values():
            self.assertEqual(array.dtype, np.uint16)
            self.assertTrue(np.all(array > 0))
        self.assertFalse(result.metadata["metric"])
        self.assertFalse(result.metadata["missing_faces_fabricated"])
        self.assertEqual(result.metadata["global_z_mapping"]["z_range_mm"], [1000, 5000])
        self.assertEqual(result.metadata["finite_handling"]["front"]["replaced_non_finite"], 1)
        self.assertEqual(result.metadata["finite_handling"]["right"]["replaced_non_finite"], 1)
        for component in result.metadata["components"]:
            for transform in component["transforms"].values():
                self.assertGreater(transform["scale"], 0.0)

        singleton = self.depth.postprocess_cubemap_depths({
            "front": np.full((8, 8), 7.0, dtype=np.float64),
        })
        component = singleton.metadata["components"][0]
        self.assertTrue(component["singleton"])
        self.assertEqual(component["gauge"], "front")
        self.assertEqual(component["transforms"]["front"], {"scale": 1.0, "shift": 0.0})
        self.assertTrue(singleton.metadata["global_z_mapping"]["constant_mapping"])

    def test_exact_reconciliation_for_observed_edges_and_corners(self):
        faces = ("front", "right", "back", "left", "top", "bottom")
        result = self.depth.postprocess_cubemap_depths(self.predictions(faces))
        depths = result.depths_mm

        for first_face, first_edge, second_face, second_edge, reverse_second in self.manual.EDGE_PAIRS:
            first = self.manual._edge(depths[first_face], first_edge)
            second = self.manual._edge(depths[second_face], second_edge)
            if reverse_second:
                second = second[::-1]
            np.testing.assert_array_equal(first, second)

        front_corner = int(depths["front"][0, -1])
        self.assertEqual(front_corner, int(depths["right"][0, 0]))
        self.assertEqual(front_corner, int(depths["top"][-1, -1]))

    def test_disconnected_subset_is_marked_degraded_without_fabrication(self):
        result = self.depth.postprocess_cubemap_depths(
            self.predictions(("front", "back"))
        )
        self.assertEqual(tuple(result.depths_mm), ("front", "back"))
        self.assertTrue(result.metadata["degraded"])
        self.assertEqual(result.metadata["quality_status"], "degraded")
        self.assertEqual(result.metadata["component_count"], 2)
        self.assertIn("disconnected", result.metadata["warning"].lower())
        self.assertEqual(result.metadata["observed_shared_edge_count"], 0)
        self.assertFalse(result.metadata["missing_faces_fabricated"])


class CubemapDepthGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_module(
            "dreamcube_generator_depth_contract_test",
            EXT_DIR / "generator.py",
        )

    @staticmethod
    def _fake_prediction(image, _variant):
        base = float(image.getpixel((0, 0))[0])
        yy, xx = np.indices((image.height, image.width), dtype=np.float64)
        return base + xx + (yy * 0.5)

    @staticmethod
    def _write_rgb_picker(root: Path, face: str, value: int) -> Path:
        path = root / f"{face}-picker-rgb.png"
        with Image.open(io.BytesIO(rgb_bytes(value))) as image:
            image.save(path)
        return path

    def _params_for_optional_faces(self, root: Path, optional: tuple[str, ...]) -> dict[str, str]:
        params = {
            "node_id": self.generator.CUBEMAP_DEPTH_NODE_ID,
        }
        for index, face in enumerate(optional):
            params[f"rgb_{face}_path"] = str(
                self._write_rgb_picker(root, face, 20 + (index * 25))
            )
        return params

    def _estimate_depth(self, instance, front: bytes, params: dict[str, str] | None = None) -> Path:
        with mock.patch.object(
            instance,
            "_predict_auto_depth_image",
            side_effect=self._fake_prediction,
        ):
            return instance.generate(
                front,
                params or {"node_id": self.generator.CUBEMAP_DEPTH_NODE_ID},
            )

    @staticmethod
    def _read_depth_metadata(front_depth: Path) -> dict[str, object]:
        metadata_path = front_depth.parent / "depth-estimation.json"
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _depth_file_names(front_depth: Path) -> list[str]:
        return sorted(path.name for path in front_depth.parent.glob("*_depth.png"))

    def test_front_only_depth_estimation_returns_primary_front_depth_and_no_fabrication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instance = self.generator.DreamCubeGenerator(root / "model", root)
            instance.node_id = self.generator.CUBEMAP_DEPTH_NODE_ID

            result = self._estimate_depth(instance, rgb_bytes(40))
            self.assertTrue(result.is_absolute())
            self.assertTrue(result.is_file())
            self.assertEqual(result.name, "front_depth.png")
            self.assertEqual(png_header(result), (6, 6))
            self.assertEqual(self._depth_file_names(result), ["front_depth.png"])
            metadata = self._read_depth_metadata(result)
            self.assertEqual(metadata["supplied_faces"], ["front"])
            self.assertEqual(metadata["omitted_faces"], ["right", "back", "left", "top", "bottom"])
            self.assertFalse(metadata["postprocess"]["missing_faces_fabricated"])

    def test_optional_adjacent_faces_preserve_subset(self):
        optional_sets = [
            ("right",),
            ("right", "top"),
            ("right", "back", "left"),
            ("top", "bottom"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instance = self.generator.DreamCubeGenerator(root / "model", root)
            instance.node_id = self.generator.CUBEMAP_DEPTH_NODE_ID
            for optional in optional_sets:
                params = self._params_for_optional_faces(root, optional)
                result = self._estimate_depth(instance, rgb_bytes(40), params)
                expected = sorted(f"{face}_depth.png" for face in ("front", *optional))
                self.assertEqual(self._depth_file_names(result), expected)
                metadata = self._read_depth_metadata(result)
                self.assertEqual(metadata["supplied_faces"], ["front", *optional])

    def test_disconnected_front_back_warning_for_optional_disjoint_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instance = self.generator.DreamCubeGenerator(root / "model", root)
            instance.node_id = self.generator.CUBEMAP_DEPTH_NODE_ID
            params = self._params_for_optional_faces(root, ("back",))
            result = self._estimate_depth(instance, rgb_bytes(40), params)
            metadata = self._read_depth_metadata(result)
            post = metadata["postprocess"]
            self.assertTrue(post["degraded"])
            self.assertEqual(post["quality_status"], "degraded")
            self.assertIn("disconnected", post["warning"].lower())
            self.assertEqual(post["component_count"], 2)
            self.assertEqual(post["observed_shared_edge_count"], 0)
            self.assertEqual(metadata["postprocess"]["missing_faces_fabricated"], False)

    def test_all_six_faces_emit_canonical_depth_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instance = self.generator.DreamCubeGenerator(root / "model", root)
            instance.node_id = self.generator.CUBEMAP_DEPTH_NODE_ID
            all_faces = ("front", "right", "back", "left", "top", "bottom")
            params = self._params_for_optional_faces(root, all_faces[1:])
            result = self._estimate_depth(instance, rgb_bytes(15), params)
            self.assertEqual(
                self._depth_file_names(result),
                sorted(f"{face}_depth.png" for face in all_faces),
            )
            metadata = self._read_depth_metadata(result)
            self.assertEqual(metadata["supplied_faces"], list(all_faces))
            self.assertFalse(metadata["postprocess"]["missing_faces_fabricated"])
            self.assertEqual(result.name, "front_depth.png")

    def test_depth_only_load_and_readiness_ignore_dreamcube_weights_and_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup_status = root / "setup-status.json"
            setup_status.write_text('{"status":"ready"}', encoding="utf-8")
            instance = self.generator.DreamCubeGenerator(root / "missing-model-index", root)
            instance.node_id = self.generator.CUBEMAP_DEPTH_NODE_ID

            self.assertTrue(instance.is_downloaded())
            with mock.patch.object(
                instance,
                "_validate_runtime_files",
                side_effect=AssertionError("DreamCube runtime files must not be checked"),
            ), mock.patch.object(
                instance,
                "_import_upstream_app",
                side_effect=AssertionError("DreamCube app must not be imported"),
            ):
                instance.load()
            self.assertTrue(instance.is_loaded())
            self.assertIsNone(instance._pipe)

            with mock.patch.object(self.generator, "SETUP_STATUS_PATH", setup_status):
                status = instance.readiness_status()
            self.assertTrue(status["ok"])
            self.assertFalse(status["details"]["dreamcube_weights_required"])
            self.assertFalse(status["details"]["dreamcube_pipeline_required"])
            self.assertNotIn("model_index", status["details"])

            instance.unload()
            self.assertFalse(instance.is_loaded())


class ManualPathInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = load_module(
            "dreamcube_manual_legacy_contract_test",
            EXT_DIR / "dreamcube_manual_cubemap.py",
        )
        cls.generator = load_module(
            "dreamcube_generator_manual_legacy_contract_test",
            EXT_DIR / "generator.py",
        )

    @staticmethod
    def picker_params(root: Path) -> dict[str, str]:
        params: dict[str, str] = {}
        for face in ("right", "back", "left", "top", "bottom"):
            path = root / f"{face}-picker-rgb.png"
            Image.new("RGB", (6, 6), (80, 100, 120)).save(path)
            params[f"rgb_{face}_path"] = str(path)
        for face in ("front", "right", "back", "left", "top", "bottom"):
            path = root / f"{face}-picker-depth.png"
            Image.fromarray(np.full((6, 6), 2000, dtype=np.uint16)).save(path)
            params[f"depth_{face}_path"] = str(path)
        for field in (
            "front",
            "right",
            "back",
            "left",
            "top",
            "bottom",
        ):
            params[f"prompt_{field}"] = f"prompt {field}"
        return params

    def test_load_manual_inputs_prefers_path_only_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = self.picker_params(root)
            inputs = self.manual.load_manual_cubemap_inputs(
                front_rgb_bytes=rgb_bytes(64),
                params=params,
                outputs_dir=root,
            )
            self.assertEqual(inputs.source_stats["faces"]["front"]["rgb_source"], "modly_image_input")
            for face in ("right", "back", "left", "top", "bottom"):
                key = f"{face}"
                self.assertEqual(
                    Path(inputs.source_stats["faces"][face]["rgb_source"]),
                    Path(params[f"rgb_{face}_path"]).resolve(),
                )
                self.assertEqual(
                    Path(inputs.source_stats["faces"][face]["depth_source"]),
                    Path(params[f"depth_{face}_path"]).resolve(),
                )
            self.assertNotIn("workflow_port", str(inputs.source_stats["faces"]["right"]))

    def test_manual_generate_returns_primary_glb_path(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            instance = gen.DreamCubeGenerator(root / "model", root)
            instance.node_id = gen.MANUAL_SCENE_NODE_ID

            with tempfile.TemporaryDirectory() as output_root:
                out_root = Path(output_root)

                def fake_generate(
                    image_bytes,
                    safe_params,
                    defaults,
                    progress_cb,
                    cancel_event,
                ):
                    self.assertEqual(image_bytes, rgb_bytes(64))
                    run_dir = out_root / "manual-run"
                    run_dir.mkdir()
                    mesh = run_dir / "output_mesh.glb"
                    mesh.write_bytes(b"glb")
                    scene_manifest = run_dir / "scene-manifest.json"
                    scene_manifest.write_text("{}", encoding="utf-8")
                    return mesh

                params = self.picker_params(root)
                with mock.patch.object(instance, "load"), \
                     mock.patch.object(instance, "_generate_manual_cubemap", side_effect=fake_generate):
                    result = instance.generate(rgb_bytes(64), params)

            self.assertEqual(result.name, "output_mesh.glb")


if __name__ == "__main__":
    unittest.main()
