from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
import io


EXT_DIR = Path(__file__).resolve().parents[1]
GENERATOR_PATH = EXT_DIR / "generator.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("dreamcube_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prompt_params(depth_image_path: str = "") -> dict[str, str]:
    return {
        "depth_image_path": depth_image_path,
        "prompt_front": "front room",
        "prompt_right": "right wall",
        "prompt_back": "back room",
        "prompt_left": "left wall",
        "prompt_top": "ceiling",
        "prompt_bottom": "floor",
    }


def rgb_png_bytes(width: int, height: int, columns: list[tuple[int, int, int]] | None = None) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), color=(127, 64, 32))
    if columns is not None:
        for x, color in enumerate(columns):
            for y in range(height):
                image.putpixel((x, y), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def tiny_png_bytes() -> bytes:
    return rgb_png_bytes(2, 2)


def save_uint16_depth(path: Path, width: int, height: int) -> None:
    from PIL import Image

    image = Image.new("I;16", (width, height))
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), 1000 + x + (y * 100))
    image.save(path)


class DreamCubeGeneratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator_module()

    def test_safe_helper_parsing(self):
        gen = self.generator

        self.assertEqual(gen._safe_int("7", 1, maximum=5), 5)
        self.assertEqual(gen._safe_int(True, 3), 3)
        self.assertEqual(gen._safe_float("0.25", 1.0, minimum=0.5), 0.5)
        self.assertTrue(gen._safe_bool("yes"))
        self.assertFalse(gen._safe_bool("0", default=True))
        self.assertEqual(gen._safe_choice("OBJ", ("glb", "obj"), "glb"), "obj")

    def test_validate_request_resolves_outputs_relative_depth(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / "outputs"
            outputs_dir.mkdir()
            depth_path = outputs_dir / "front-depth.png"
            depth_path.write_bytes(b"depth")
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", outputs_dir)

            resolved_depth, prompts = instance._validate_request(
                b"rgb-bytes",
                prompt_params("front-depth.png"),
            )

            self.assertEqual(resolved_depth, depth_path.resolve())
            self.assertEqual(len(prompts), 6)

    def test_auto_mode_missing_depth_is_allowed_before_model_load(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", root / "outputs")

            depth_path, prompts = instance._validate_request(b"rgb-bytes", prompt_params())

            self.assertIsNone(depth_path)
            self.assertEqual(len(prompts), 6)

    def test_manual_mode_missing_depth_fails_before_model_load(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", root / "outputs")
            params = prompt_params()
            params["depth_mode"] = "manual"

            with self.assertRaises(ValueError) as ctx:
                instance._validate_request(b"rgb-bytes", params)

            self.assertIn("depth_image_path", str(ctx.exception))

    def test_missing_prompt_validation_is_clear(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            depth_path = root / "front-depth.png"
            depth_path.write_bytes(b"depth")
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", root / "outputs")
            params = prompt_params(str(depth_path))
            params["prompt_bottom"] = ""

            with self.assertRaises(ValueError) as ctx:
                instance._validate_request(b"rgb-bytes", params)

            self.assertIn("prompt_bottom", str(ctx.exception))


    def test_non_square_rgb_is_center_cropped_and_metadata_written(self):
        from PIL import Image

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / "outputs"
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", outputs_dir)
            calls: dict[str, Path] = {}

            def fake_auto_depth(image_path: Path, run_dir: Path, variant: str) -> Path:
                calls["auto_depth_image"] = image_path
                with Image.open(image_path) as canonical_rgb:
                    self.assertEqual(canonical_rgb.size, (2, 2))
                    self.assertEqual(canonical_rgb.getpixel((0, 0)), (0, 255, 0))
                    self.assertEqual(canonical_rgb.getpixel((1, 0)), (0, 0, 255))
                depth = run_dir / "input_front_depth_auto.png"
                depth.write_bytes(b"depth")
                return depth

            def fake_inference(**kwargs):
                calls["inference_image"] = Path(kwargs["image_path"])
                calls["inference_depth"] = Path(kwargs["depth_path"])
                return {"ok": True}

            def fake_outputs(*, predictions, run_dir, save_all_outputs):
                result = run_dir / "output_equi_rgb.png"
                result.write_bytes(b"png")
                return {"equi_rgb_path": result}

            image_bytes = rgb_png_bytes(
                4,
                2,
                columns=[
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 255),
                ],
            )

            with mock.patch.object(instance, "_save_auto_depth_image", side_effect=fake_auto_depth), \
                mock.patch.object(instance, "load", return_value=None), \
                mock.patch.object(instance, "_build_seed_kwargs", return_value={}), \
                mock.patch.object(instance, "_run_inference", side_effect=fake_inference), \
                mock.patch.object(instance, "_save_postprocessed_outputs", side_effect=fake_outputs):
                result = instance.generate(image_bytes, prompt_params())

            saved_rgb = result.parent / "input_front_rgb.png"
            self.assertEqual(calls["auto_depth_image"], saved_rgb)
            self.assertEqual(calls["inference_image"], saved_rgb)
            with Image.open(saved_rgb) as canonical_rgb:
                self.assertEqual(canonical_rgb.size, (2, 2))
                self.assertEqual(canonical_rgb.getpixel((0, 0)), (0, 255, 0))
                self.assertEqual(canonical_rgb.getpixel((1, 0)), (0, 0, 255))

            metadata = json.loads((result.parent / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["original_rgb_size"], [4, 2])
            self.assertEqual(metadata["canonical_rgb_size"], [2, 2])
            self.assertEqual(metadata["crop_box"], [1, 0, 3, 2])
            self.assertEqual(metadata["depth_mode"], "auto")
            self.assertEqual(metadata["auto_depth_variant"], "vits")
            self.assertEqual(metadata["node_id"], gen.PANORAMA_NODE_ID)
            self.assertEqual(metadata["output_format"], "equirect_rgb_png")
            self.assertEqual(metadata["prompts"]["prompt_front"], "front room")


    def test_auto_mode_writes_generated_depth_and_uses_it_for_inference(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / "outputs"
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", outputs_dir)
            calls: dict[str, Path] = {}

            def fake_auto_depth(image_path: Path, run_dir: Path, variant: str) -> Path:
                self.assertEqual(image_path.name, "input_front_rgb.png")
                self.assertEqual(variant, "vits")
                depth = run_dir / "input_front_depth_auto.png"
                depth.write_bytes(b"depth")
                calls["auto_depth"] = depth
                return depth

            def fake_inference(**kwargs):
                calls["inference_depth"] = Path(kwargs["depth_path"])
                return {"ok": True}

            def fake_outputs(*, predictions, run_dir, save_all_outputs):
                result = run_dir / "output_equi_rgb.png"
                result.write_bytes(b"png")
                return {"equi_rgb_path": result}

            with mock.patch.object(instance, "_save_auto_depth_image", side_effect=fake_auto_depth), \
                mock.patch.object(instance, "load", return_value=None), \
                mock.patch.object(instance, "_build_seed_kwargs", return_value={}), \
                mock.patch.object(instance, "_run_inference", side_effect=fake_inference), \
                mock.patch.object(instance, "_save_postprocessed_outputs", side_effect=fake_outputs):
                result = instance.generate(tiny_png_bytes(), prompt_params())

            self.assertEqual(result.name, "output_equi_rgb.png")
            self.assertEqual(calls["inference_depth"], calls["auto_depth"])
            self.assertTrue(calls["auto_depth"].is_file())

    def test_supplied_matching_manual_depth_is_cropped_with_rgb_crop_box(self):
        from PIL import Image

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / "outputs"
            outputs_dir.mkdir()
            supplied_depth = outputs_dir / "front-depth.png"
            save_uint16_depth(supplied_depth, 4, 2)
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", outputs_dir)
            calls: dict[str, Path] = {}

            def fake_inference(**kwargs):
                calls["inference_depth"] = Path(kwargs["depth_path"])
                return {"ok": True}

            def fake_outputs(*, predictions, run_dir, save_all_outputs):
                result = run_dir / "output_equi_rgb.png"
                result.write_bytes(b"png")
                return {"equi_rgb_path": result}

            with mock.patch.object(instance, "_save_auto_depth_image", side_effect=AssertionError("auto-depth should not run")), \
                mock.patch.object(instance, "load", return_value=None), \
                mock.patch.object(instance, "_build_seed_kwargs", return_value={}), \
                mock.patch.object(instance, "_run_inference", side_effect=fake_inference), \
                mock.patch.object(instance, "_save_postprocessed_outputs", side_effect=fake_outputs):
                result = instance.generate(rgb_png_bytes(4, 2), prompt_params("front-depth.png"))

            canonical_depth = calls["inference_depth"]
            self.assertEqual(canonical_depth.parent, result.parent)
            self.assertEqual(canonical_depth.name, "input_front_depth_manual.png")
            self.assertNotEqual(canonical_depth, supplied_depth.resolve())
            with Image.open(canonical_depth) as depth_image:
                self.assertEqual(depth_image.size, (2, 2))
                self.assertEqual(len(depth_image.getbands()), 1)
                self.assertEqual(depth_image.getpixel((0, 0)), 1001)
                self.assertEqual(depth_image.getpixel((1, 0)), 1002)

            metadata = json.loads((result.parent / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["depth_mode"], "manual")
            self.assertEqual(metadata["depth"]["crop_box"], [1, 0, 3, 2])
            self.assertEqual(metadata["depth"]["crop_source"], "rgb")
            self.assertEqual(metadata["depth_source"], str(supplied_depth.resolve()))

    def test_readiness_status_shape_is_lightweight(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", root / "outputs")
            status = instance.readiness_status()

            self.assertIn("ok", status)
            self.assertIn("machine_code", status)
            self.assertIn("reason", status)
            self.assertIn("details", status)
            self.assertEqual(status["details"]["readiness_source"], "generator.py")

    def test_node_id_resolution_prefers_instance_node_id(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = gen.DreamCubeGenerator(root / "models" / "dreamcube", root / "outputs")
            instance.node_id = "generate-scene"

            self.assertEqual(instance._resolve_node_id({}), gen.SCENE_NODE_ID)
            self.assertEqual(instance._resolve_node_id({"output_format": "glb"}), gen.SCENE_NODE_ID)


if __name__ == "__main__":
    unittest.main()

class DreamCubeMeshGeneratorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator_module()

    @staticmethod
    def mesh_stats(mode: str) -> dict[str, object]:
        return {
            'mode': mode,
            'vertices': {'candidate': 24, 'exported': 12},
            'triangles': {'candidate': 32, 'exported': 16},
        }

    def test_export_scene_passes_same_canonical_equi_rays_to_mesh_and_3dgs(self):
        import numpy as np
        from PIL import Image

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = gen.DreamCubeGenerator(Path(temp_dir) / 'models', Path(temp_dir) / 'outputs')
            instance._pipe = type('Pipe', (), {'device': 'cpu'})()
            calls: dict[str, dict[str, object]] = {}
            instance._app = type('App', (), {
                'convert_rgbd_equi_to_3dgs': staticmethod(lambda **kwargs: calls.setdefault('3dgs', kwargs)),
            })()
            outputs = {
                'equi_rgb': Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8)),
                'equi_depth_raw': Image.fromarray(np.full((2, 3), 1000, dtype=np.uint16)),
            }
            canonical_rays = np.full((2, 3, 3), 0.25, dtype=np.float32)
            fake_torch = types.SimpleNamespace(
                tensor=lambda value, device=None, dtype=None: np.asarray(value),
                float32=np.float32,
            )

            def owned_mesh(**kwargs):
                calls['mesh'] = kwargs
                return type('Result', (), {'stats': self.mesh_stats('equirectangular')})()

            def convert_glb(obj_path, glb_path, mesh_stats):
                glb_path.write_bytes(b'glb')
                return glb_path

            with mock.patch.dict(sys.modules, {'torch': fake_torch}), \
                 mock.patch.object(gen.dreamcube_mesh, 'convert_rgbd_equi_to_mesh', side_effect=owned_mesh), \
                 mock.patch.object(gen.dreamcube_mesh, 'equi_unit_rays', return_value=canonical_rays) as build_rays, \
                 mock.patch.object(instance, '_convert_obj_to_glb', side_effect=convert_glb), \
                 mock.patch.object(gen, '_log') as log:
                result, stats = instance._export_scene(
                    outputs=outputs,
                    run_dir=Path(temp_dir),
                    mode=gen.PANO_TO_3D_EQUIRECTANGULAR,
                    max_equi_size=512,
                    max_cube_size=256,
                    mesh_depth_jump_threshold=0.35,
                )

            self.assertEqual(result, Path(temp_dir) / 'output_mesh.glb')
            self.assertEqual(stats['mode'], 'equirectangular')
            self.assertEqual(calls['mesh']['max_size'], 512)
            self.assertEqual(calls['mesh']['depth_jump_threshold'], 0.35)
            self.assertEqual(calls['3dgs']['max_size'], 512)
            self.assertEqual(calls['3dgs']['save_path'], str(Path(temp_dir) / 'output_3dgs.splat'))
            self.assertIs(calls['mesh']['rays'], canonical_rays)
            self.assertIs(calls['3dgs']['rays'], canonical_rays)
            build_rays.assert_called_once_with(2, 3, device='cpu')
            log.assert_called_once()

    def test_export_scene_passes_same_canonical_cube_rays_to_mesh_and_3dgs(self):
        import numpy as np

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = gen.DreamCubeGenerator(Path(temp_dir) / 'models', Path(temp_dir) / 'outputs')
            instance._pipe = type('Pipe', (), {'device': 'cpu'})()
            calls: dict[str, dict[str, object]] = {}
            instance._app = type('App', (), {
                'convert_rgbd_cube_to_3dgs': staticmethod(lambda **kwargs: calls.setdefault('3dgs', kwargs)),
            })()
            outputs = {
                'images_pred': np.full((1, 6, 2, 2, 3), 127, dtype=np.float32),
                'depths_distance': np.full((1, 6, 2, 2, 1), 1000, dtype=np.float32),
            }
            canonical_rays = np.full((6, 2, 2, 3), 0.25, dtype=np.float32)

            def owned_mesh(**kwargs):
                calls['mesh'] = kwargs
                return type('Result', (), {'stats': self.mesh_stats('cubemap')})()

            def convert_glb(obj_path, glb_path, mesh_stats):
                glb_path.write_bytes(b'glb')
                return glb_path

            fake_torch = types.SimpleNamespace(
                tensor=lambda value, device=None, dtype=None: np.asarray(value),
                float32=np.float32,
            )
            with mock.patch.dict(sys.modules, {'torch': fake_torch}), \
                 mock.patch.object(gen.dreamcube_mesh, 'convert_rgbd_cube_to_mesh', side_effect=owned_mesh), \
                 mock.patch.object(gen.dreamcube_mesh, 'cube_unit_rays', return_value=canonical_rays) as build_rays, \
                 mock.patch.object(instance, '_convert_obj_to_glb', side_effect=convert_glb), \
                 mock.patch.object(gen, '_log') as log:
                result, stats = instance._export_scene(
                    outputs=outputs,
                    run_dir=Path(temp_dir),
                    mode=gen.PANO_TO_3D_CUBEMAP,
                    max_equi_size=1024,
                    max_cube_size=128,
                    mesh_depth_jump_threshold=0.20,
                )

            self.assertEqual(result, Path(temp_dir) / 'output_mesh.glb')
            self.assertEqual(stats['mode'], 'cubemap')
            self.assertEqual(calls['mesh']['max_size'], 128)
            self.assertEqual(calls['3dgs']['max_size'], 128)
            self.assertIs(calls['mesh']['rays'], canonical_rays)
            self.assertIs(calls['3dgs']['rays'], canonical_rays)
            build_rays.assert_called_once_with(2, device='cpu')
            log.assert_called_once()

    def test_export_scene_uses_real_torch_canonical_rays_for_equi_and_cube(self):
        import numpy as np
        from PIL import Image

        try:
            import torch
            import torch.nn.functional as functional
        except ModuleNotFoundError as exc:
            if exc.name == 'torch':
                self.skipTest('Torch is not installed')
            raise

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = gen.DreamCubeGenerator(root / 'models', root / 'outputs')
            instance._pipe = type('Pipe', (), {'device': torch.device('cpu')})()
            calls: dict[str, dict[str, dict[str, object]]] = {
                'equi': {},
                'cube': {},
            }

            instance._app = type('App', (), {
                'convert_rgbd_equi_to_3dgs': staticmethod(
                    lambda **kwargs: calls['equi'].setdefault('3dgs', kwargs)
                ),
                'convert_rgbd_cube_to_3dgs': staticmethod(
                    lambda **kwargs: calls['cube'].setdefault('3dgs', kwargs)
                ),
            })()

            def equi_mesh(**kwargs):
                calls['equi']['mesh'] = kwargs
                return type('Result', (), {'stats': self.mesh_stats('equirectangular')})()

            def cube_mesh(**kwargs):
                calls['cube']['mesh'] = kwargs
                return type('Result', (), {'stats': self.mesh_stats('cubemap')})()

            def convert_glb(obj_path, glb_path, mesh_stats):
                glb_path.write_bytes(b'glb')
                return glb_path

            equi_outputs = {
                'equi_rgb': Image.fromarray(np.full((4, 8, 3), 127, dtype=np.uint8)),
                'equi_depth_raw': Image.fromarray(np.full((4, 8), 1000, dtype=np.uint16)),
            }
            cube_outputs = {
                'images_pred': np.full((1, 6, 4, 4, 3), 127, dtype=np.float32),
                'depths_distance': np.full((1, 6, 4, 4, 1), 1000, dtype=np.float32),
            }
            (root / 'equi').mkdir()
            (root / 'cube').mkdir()

            with mock.patch.object(
                gen.dreamcube_mesh,
                'convert_rgbd_equi_to_mesh',
                side_effect=equi_mesh,
            ), mock.patch.object(
                gen.dreamcube_mesh,
                'convert_rgbd_cube_to_mesh',
                side_effect=cube_mesh,
            ), mock.patch.object(
                instance,
                '_convert_obj_to_glb',
                side_effect=convert_glb,
            ), mock.patch.object(gen, '_log'):
                instance._export_scene(
                    outputs=equi_outputs,
                    run_dir=root / 'equi',
                    mode=gen.PANO_TO_3D_EQUIRECTANGULAR,
                    max_equi_size=4,
                    max_cube_size=2,
                    mesh_depth_jump_threshold=0.35,
                )
                instance._export_scene(
                    outputs=cube_outputs,
                    run_dir=root / 'cube',
                    mode=gen.PANO_TO_3D_CUBEMAP,
                    max_equi_size=4,
                    max_cube_size=2,
                    mesh_depth_jump_threshold=0.35,
                )

            equi_mesh_rays = calls['equi']['mesh']['rays']
            equi_3dgs_rays = calls['equi']['3dgs']['rays']
            self.assertIsNotNone(equi_3dgs_rays)
            self.assertEqual(tuple(equi_mesh_rays.shape), (4, 8, 3))
            self.assertEqual(tuple(equi_3dgs_rays.shape), (2, 4, 3))
            self.assertEqual(equi_mesh_rays.device.type, 'cpu')
            self.assertEqual(equi_3dgs_rays.device.type, 'cpu')
            expected_equi = functional.interpolate(
                equi_mesh_rays.permute(2, 0, 1).unsqueeze(0),
                scale_factor=0.5,
                mode='bilinear',
                align_corners=False,
                recompute_scale_factor=False,
            ).squeeze(0).permute(1, 2, 0)
            expected_equi = expected_equi / (
                expected_equi.norm(dim=-1, keepdim=True) + 1e-8
            )
            self.assertTrue(torch.allclose(equi_3dgs_rays, expected_equi))
            flipped_equi = expected_equi.clone()
            flipped_equi[..., :2] *= -1
            self.assertFalse(torch.allclose(equi_3dgs_rays, flipped_equi))

            cube_mesh_rays = calls['cube']['mesh']['rays']
            cube_3dgs_rays = calls['cube']['3dgs']['rays']
            self.assertIsNotNone(cube_3dgs_rays)
            self.assertEqual(tuple(cube_mesh_rays.shape), (6, 4, 4, 3))
            self.assertEqual(tuple(cube_3dgs_rays.shape), (6, 2, 2, 3))
            self.assertEqual(cube_mesh_rays.device.type, 'cpu')
            self.assertEqual(cube_3dgs_rays.device.type, 'cpu')
            expected_cube = functional.interpolate(
                cube_mesh_rays.permute(0, 3, 1, 2),
                scale_factor=0.5,
                mode='bilinear',
                align_corners=False,
                recompute_scale_factor=False,
            ).permute(0, 2, 3, 1)
            expected_cube = expected_cube / (
                expected_cube.norm(dim=-1, keepdim=True) + 1e-8
            )
            self.assertTrue(torch.allclose(cube_3dgs_rays, expected_cube))
            flipped_cube = expected_cube.clone()
            flipped_cube[..., :2] *= -1
            self.assertFalse(torch.allclose(cube_3dgs_rays, flipped_cube))

    def test_glb_conversion_loads_obj_without_processing_and_exports_materialized_file(self):
        import numpy as np

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            instance = gen.DreamCubeGenerator(Path(temp_dir) / 'models', Path(temp_dir) / 'outputs')
            calls: dict[str, object] = {}

            class FakeMesh:
                @property
                def vertex_normals(self):
                    calls['normals_requested'] = True
                    return np.zeros((3, 3))

                def export(self, path):
                    calls['export_path'] = path
                    Path(path).write_bytes(b'glb')

            class FakePBRMaterial:
                def __init__(self, **kwargs):
                    calls['material_kwargs'] = kwargs

            fake_mesh = FakeMesh()
            fake_trimesh = types.SimpleNamespace(
                load=lambda *args, **kwargs: (calls.__setitem__('load', (args, kwargs)) or fake_mesh),
                visual=types.SimpleNamespace(material=types.SimpleNamespace(PBRMaterial=FakePBRMaterial)),
            )
            obj_path = Path(temp_dir) / 'output_mesh.obj'
            glb_path = Path(temp_dir) / 'output_mesh.glb'
            obj_path.write_text('v 0 0 0 1 0 0\nf 1 1 1\n', encoding='utf-8')

            with mock.patch.dict(sys.modules, {'trimesh': fake_trimesh}):
                result = instance._convert_obj_to_glb(obj_path, glb_path, self.mesh_stats('cubemap'))

            self.assertEqual(result, glb_path)
            load_args, load_kwargs = calls['load']
            self.assertEqual(load_args, (str(obj_path),))
            self.assertEqual(load_kwargs, {'force': 'mesh', 'process': False, 'maintain_order': True})
            self.assertEqual(calls['material_kwargs'], {'doubleSided': True})
            self.assertTrue(calls['normals_requested'])

    def test_glb_conversion_error_is_required_and_diagnostic(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = gen.DreamCubeGenerator(root / 'models', root / 'outputs')
            obj_path = root / 'output_mesh.obj'
            glb_path = root / 'output_mesh.glb'
            obj_path.write_text('v 0 0 0\n', encoding='utf-8')
            fake_trimesh = types.SimpleNamespace(load=mock.Mock(side_effect=ValueError('bad obj')))

            with mock.patch.dict(sys.modules, {'trimesh': fake_trimesh}), \
                 mock.patch.object(gen, '_log') as log, \
                 self.assertRaises(gen.SceneGenerationError) as raised:
                instance._convert_obj_to_glb(obj_path, glb_path, self.mesh_stats('cubemap'))

            self.assertEqual(raised.exception.stage, 'glb_conversion')
            self.assertIn('OBJ-to-GLB conversion failed', str(raised.exception))
            self.assertFalse(raised.exception.diagnostics['glb_exists'])
            self.assertEqual(raised.exception.stats['mode'], 'cubemap')
            log.assert_called_once()

    def test_glb_missing_or_empty_after_export_is_a_generation_error(self):
        gen = self.generator
        for case in ('missing', 'empty'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                instance = gen.DreamCubeGenerator(root / 'models', root / 'outputs')
                obj_path = root / 'output_mesh.obj'
                glb_path = root / 'output_mesh.glb'
                obj_path.write_text('v 0 0 0\n', encoding='utf-8')

                class FakeMesh:
                    vertex_normals = ()
                    visual = types.SimpleNamespace(material=None)

                    def export(self, path):
                        if case == 'empty':
                            Path(path).write_bytes(b'')

                fake_trimesh = types.SimpleNamespace(
                    load=lambda *args, **kwargs: FakeMesh(),
                    visual=types.SimpleNamespace(
                        material=types.SimpleNamespace(PBRMaterial=lambda **kwargs: object())
                    ),
                )
                with mock.patch.dict(sys.modules, {'trimesh': fake_trimesh}), \
                     mock.patch.object(gen, '_log'), \
                     self.assertRaises(gen.SceneGenerationError) as raised:
                    instance._convert_obj_to_glb(obj_path, glb_path, self.mesh_stats('cubemap'))

                self.assertEqual(raised.exception.stage, 'glb_validation')
                self.assertIn('missing or empty', str(raised.exception))
                self.assertEqual(raised.exception.diagnostics['glb_size_bytes'], 0)

    def test_scene_manifest_contains_one_workspace_relative_base_glb(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace_dir = root / 'workspace'
            outputs_dir = workspace_dir / 'Default'
            run_dir = outputs_dir / 'dreamcube-run'
            run_dir.mkdir(parents=True)
            glb_path = run_dir / 'output_mesh.glb'
            glb_path.write_bytes(b'glb')
            instance = gen.DreamCubeGenerator(root / 'models', outputs_dir)

            with mock.patch.dict(os.environ, {'WORKSPACE_DIR': str(workspace_dir)}):
                manifest_path = instance._write_scene_manifest(run_dir, glb_path)

            self.assertEqual(manifest_path, run_dir / 'scene-manifest.json')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['schema'], 'modly.scene-manifest.v1')
            self.assertEqual(manifest['sceneRoot'], '.')
            self.assertEqual(manifest['generator'], 'modly.worlds')
            self.assertEqual(manifest['version'], 1)
            self.assertEqual(datetime.fromisoformat(manifest['createdAt']).utcoffset().total_seconds(), 0)
            self.assertNotIn('collisionZones', manifest)
            self.assertEqual(manifest['initialView'], {
                'position': [0, 0, 0],
                'target': [0, 0, 1],
                'up': [0, 1, 0],
            })
            self.assertEqual(len(manifest['assets']), 1)
            asset = manifest['assets'][0]
            self.assertEqual(asset['role'], 'base-scene')
            self.assertEqual(asset['kind'], 'glb')
            self.assertTrue(asset['visible'])
            self.assertEqual(asset['workspacePath'], 'Default/dreamcube-run/output_mesh.glb')
            self.assertEqual(asset['transform'], {
                'position': [0, 0, 0],
                'rotation': [0, 0, 0],
                'scale': [1, 1, 1],
            })
            self.assertNotIn(str(workspace_dir), asset['workspacePath'])

    def test_scene_manifest_rejects_asset_outside_official_workspace(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace_dir = root / 'workspace'
            outputs_dir = workspace_dir / 'Default'
            run_dir = outputs_dir / 'dreamcube-run'
            run_dir.mkdir(parents=True)
            outside_glb = root / 'outside.glb'
            outside_glb.write_bytes(b'glb')
            instance = gen.DreamCubeGenerator(root / 'models', outputs_dir)

            with mock.patch.dict(os.environ, {'WORKSPACE_DIR': str(workspace_dir)}), \
                 self.assertRaises(gen.SceneGenerationError) as raised:
                instance._write_scene_manifest(run_dir, outside_glb)

            self.assertEqual(raised.exception.stage, 'workspace_path')
            self.assertIn('outside the official workspace root', str(raised.exception))
            self.assertFalse((run_dir / 'scene-manifest.json').exists())

    def test_scene_manifest_requires_workspace_dir_instead_of_guessing_outputs_parent(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'workspace' / 'Default'
            run_dir = outputs_dir / 'dreamcube-run'
            run_dir.mkdir(parents=True)
            glb_path = run_dir / 'output_mesh.glb'
            glb_path.write_bytes(b'glb')
            instance = gen.DreamCubeGenerator(root / 'models', outputs_dir)

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('WORKSPACE_DIR', None)
                with self.assertRaises(gen.SceneGenerationError) as raised:
                    instance._write_scene_manifest(run_dir, glb_path)

            self.assertEqual(raised.exception.stage, 'workspace_path')
            self.assertIn('WORKSPACE_DIR is required', str(raised.exception))
            self.assertEqual(raised.exception.diagnostics['workspace_root_source'], 'missing')
            self.assertFalse((run_dir / 'scene-manifest.json').exists())

    def test_scene_manifest_rejects_invalid_workspace_dir(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'outputs'
            run_dir = outputs_dir / 'dreamcube-run'
            run_dir.mkdir(parents=True)
            glb_path = run_dir / 'output_mesh.glb'
            glb_path.write_bytes(b'glb')
            workspace_file = root / 'workspace-file'
            workspace_file.write_text('not a directory', encoding='utf-8')
            instance = gen.DreamCubeGenerator(root / 'models', outputs_dir)

            for workspace_value in ('relative/workspace', str(workspace_file)):
                with (
                    self.subTest(workspace_value=workspace_value),
                    mock.patch.dict(os.environ, {'WORKSPACE_DIR': workspace_value}),
                    self.assertRaises(gen.SceneGenerationError) as raised,
                ):
                    instance._write_scene_manifest(run_dir, glb_path)

                self.assertEqual(raised.exception.stage, 'workspace_path')
                self.assertIn('WORKSPACE_DIR', str(raised.exception))
                self.assertFalse((run_dir / 'scene-manifest.json').exists())

    def test_scene_returns_manifest_and_records_glb_coordinate_and_presentation_metadata(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'outputs'
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / 'models' / 'dreamcube', outputs_dir)
            instance.node_id = gen.SCENE_NODE_ID
            captured: dict[str, object] = {}

            def auto_depth(image_path, run_dir, variant):
                path = run_dir / 'depth.png'
                path.write_bytes(b'depth')
                return path

            def outputs(*, predictions, run_dir, save_all_outputs):
                captured['run_dir'] = run_dir
                path = run_dir / 'output_equi_rgb.png'
                path.write_bytes(b'png')
                return {'equi_rgb_path': path}

            def export(**kwargs):
                captured['export'] = kwargs
                path = kwargs['run_dir'] / 'output_mesh.glb'
                path.write_bytes(b'glb')
                return path, self.mesh_stats('cubemap')

            params = prompt_params()
            params.update({
                'node_id': gen.SCENE_NODE_ID,
                'max_cube_size': 128,
                'mesh_depth_jump_threshold': 0.35,
                'output_format': 'obj',
            })
            with mock.patch.dict(os.environ, {'WORKSPACE_DIR': str(root)}), \
                 mock.patch.object(instance, '_save_auto_depth_image', side_effect=auto_depth), \
                 mock.patch.object(instance, 'load'), \
                 mock.patch.object(instance, '_build_seed_kwargs', return_value={}), \
                 mock.patch.object(instance, '_run_inference', return_value={}), \
                 mock.patch.object(instance, '_save_postprocessed_outputs', side_effect=outputs), \
                 mock.patch.object(instance, '_export_scene', side_effect=export):
                result = instance.generate(tiny_png_bytes(), params)

            self.assertEqual(result.name, 'scene-manifest.json')
            manifest = json.loads(result.read_text(encoding='utf-8'))
            self.assertEqual(manifest['assets'][0]['workspacePath'], f'outputs/{result.parent.name}/output_mesh.glb')
            metadata = json.loads((result.parent / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['output_format'], 'glb')
            self.assertEqual(metadata['mesh_export']['status'], 'success')
            self.assertEqual(metadata['mesh_export']['stats']['mode'], 'cubemap')
            self.assertEqual(metadata['reconstruction']['max_cube_size'], 128)
            self.assertEqual(metadata['reconstruction']['mesh_depth_jump_threshold'], 0.35)
            self.assertEqual(metadata['coordinate_frame'], {
                'handedness': 'right-handed',
                'units': 'meters',
                'origin': 'camera',
                'axes': {'x': 'left', 'y': 'up', 'z': 'forward'},
            })
            self.assertEqual(metadata['presentation']['type'], 'navigable-panorama')
            self.assertEqual(metadata['presentation']['viewpoint'], 'interior')
            self.assertEqual(metadata['presentation']['initial_view']['position'], [0, 0, 0])
            self.assertEqual(captured['export']['mesh_depth_jump_threshold'], 0.35)
            self.assertNotIn('output_format', captured['export'])

    def test_required_glb_failure_persists_diagnostics_before_reraise(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'outputs'
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / 'models' / 'dreamcube', outputs_dir)
            instance.node_id = gen.SCENE_NODE_ID
            captured: dict[str, Path] = {}

            def auto_depth(image_path, run_dir, variant):
                path = run_dir / 'depth.png'
                path.write_bytes(b'depth')
                return path

            def outputs(*, predictions, run_dir, save_all_outputs):
                captured['run_dir'] = run_dir
                path = run_dir / 'output_equi_rgb.png'
                path.write_bytes(b'png')
                return {'equi_rgb_path': path}

            error = gen.SceneGenerationError(
                'DreamCube OBJ-to-GLB conversion failed: bad obj',
                stage='glb_conversion',
                stats=self.mesh_stats('cubemap'),
                diagnostics={
                    'obj_exists': True,
                    'obj_size_bytes': 12,
                    'glb_exists': False,
                    'glb_size_bytes': 0,
                    'splat_exists': True,
                    'splat_size_bytes': 8,
                },
            )
            params = prompt_params()
            params['node_id'] = gen.SCENE_NODE_ID
            with mock.patch.object(instance, '_save_auto_depth_image', side_effect=auto_depth), \
                 mock.patch.object(instance, 'load'), \
                 mock.patch.object(instance, '_build_seed_kwargs', return_value={}), \
                 mock.patch.object(instance, '_run_inference', return_value={}), \
                 mock.patch.object(instance, '_save_postprocessed_outputs', side_effect=outputs), \
                 mock.patch.object(instance, '_export_scene', side_effect=error), \
                 mock.patch.object(gen, '_log') as log:
                with self.assertRaises(gen.SceneGenerationError) as raised:
                    instance.generate(tiny_png_bytes(), params)

            self.assertIs(raised.exception, error)
            metadata = json.loads((captured['run_dir'] / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['mesh_export']['status'], 'failed')
            self.assertEqual(metadata['mesh_export']['stage'], 'glb_conversion')
            self.assertEqual(metadata['mesh_export']['error'], str(error))
            self.assertEqual(metadata['mesh_export']['stats']['mode'], 'cubemap')
            self.assertEqual(metadata['mesh_export']['diagnostics']['glb_size_bytes'], 0)
            log.assert_called_once()

    def test_nonprimitive_failure_diagnostics_are_json_safe_and_preserve_original_error(self):
        import numpy as np

        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'outputs'
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / 'models' / 'dreamcube', outputs_dir)
            instance.node_id = gen.SCENE_NODE_ID
            captured: dict[str, Path] = {}

            def auto_depth(image_path, run_dir, variant):
                path = run_dir / 'depth.png'
                path.write_bytes(b'depth')
                return path

            def outputs(*, predictions, run_dir, save_all_outputs):
                captured['run_dir'] = run_dir
                path = run_dir / 'output_equi_rgb.png'
                path.write_bytes(b'png')
                return {'equi_rgb_path': path}

            class DiagnosticToken:
                def __repr__(self):
                    return '<diagnostic-token>'

            stats = {
                'mode': 'cubemap',
                'vertices': {
                    'candidate': np.int64(24),
                    'bounds': np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32),
                },
            }
            diagnostics: dict[object, object] = {
                'artifact': root / 'output_mesh.glb',
                'scores': np.array([1.25, 2.5], dtype=np.float32),
                Path('path-key'): (np.bool_(True), np.float32(0.5)),
                'fallback': DiagnosticToken(),
            }
            try:
                import torch
            except ModuleNotFoundError as exc:
                if exc.name != 'torch':
                    raise
            else:
                diagnostics['tensor'] = torch.tensor([[3.0, 4.0]], device='cpu')

            error = gen.SceneGenerationError(
                'GLB conversion failed with structured diagnostics',
                stage='glb_conversion',
                stats=stats,
                diagnostics=diagnostics,
            )
            params = prompt_params()
            params['node_id'] = gen.SCENE_NODE_ID

            with (
                mock.patch.object(instance, '_save_auto_depth_image', side_effect=auto_depth),
                mock.patch.object(instance, 'load'),
                mock.patch.object(instance, '_build_seed_kwargs', return_value={}),
                mock.patch.object(instance, '_run_inference', return_value={}),
                mock.patch.object(
                    instance,
                    '_save_postprocessed_outputs',
                    side_effect=outputs,
                ),
                mock.patch.object(instance, '_export_scene', side_effect=error),
                mock.patch.object(gen, '_log'),
            ):
                with self.assertRaises(gen.SceneGenerationError) as raised:
                    instance.generate(tiny_png_bytes(), params)

            self.assertIs(raised.exception, error)
            metadata = json.loads(
                (captured['run_dir'] / 'run_metadata.json').read_text(encoding='utf-8')
            )
            failure = metadata['mesh_export']
            self.assertEqual(failure['stats']['vertices']['candidate'], 24)
            self.assertEqual(
                failure['stats']['vertices']['bounds'],
                [[-1.0, 1.0], [-2.0, 2.0]],
            )
            self.assertEqual(failure['diagnostics']['artifact'], str(root / 'output_mesh.glb'))
            self.assertEqual(failure['diagnostics']['scores'], [1.25, 2.5])
            self.assertEqual(failure['diagnostics']['path-key'], [True, 0.5])
            self.assertEqual(failure['diagnostics']['fallback'], '<diagnostic-token>')
            if 'tensor' in diagnostics:
                self.assertEqual(failure['diagnostics']['tensor'], [[3.0, 4.0]])

    def test_scene_mesh_failure_persists_metadata_before_reraise(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'outputs'
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / 'models' / 'dreamcube', outputs_dir)
            instance.node_id = gen.SCENE_NODE_ID
            captured: dict[str, Path] = {}

            def auto_depth(image_path, run_dir, variant):
                path = run_dir / 'depth.png'
                path.write_bytes(b'depth')
                return path

            def outputs(*, predictions, run_dir, save_all_outputs):
                captured['run_dir'] = run_dir
                path = run_dir / 'output_equi_rgb.png'
                path.write_bytes(b'png')
                return {'equi_rgb_path': path}

            error_stats = self.mesh_stats('equirectangular')
            error = gen.dreamcube_mesh.MeshExportError('empty mesh', stats=error_stats)
            params = prompt_params()
            params.update({
                'node_id': gen.SCENE_NODE_ID,
                'pano_to_3d_mode': gen.PANO_TO_3D_EQUIRECTANGULAR,
                'mesh_depth_jump_threshold': 0.35,
            })
            stderr = io.StringIO()
            with mock.patch.object(instance, '_save_auto_depth_image', side_effect=auto_depth), \
                 mock.patch.object(instance, 'load'), \
                 mock.patch.object(instance, '_build_seed_kwargs', return_value={}), \
                 mock.patch.object(instance, '_run_inference', return_value={}), \
                 mock.patch.object(instance, '_save_postprocessed_outputs', side_effect=outputs), \
                 mock.patch.object(instance, '_export_scene', side_effect=error), \
                 contextlib.redirect_stderr(stderr):
                with self.assertRaises(gen.dreamcube_mesh.MeshExportError) as raised:
                    instance.generate(tiny_png_bytes(), params)
            self.assertIs(raised.exception, error)
            summaries = [line for line in stderr.getvalue().splitlines() if 'Mesh export failed' in line]
            self.assertEqual(summaries, [
                '[dreamcube] Mesh export failed mode=equirectangular triangles=16/32 '
                'removed_invalid=unknown removed_jump=unknown removed_footprint=unknown '
                'removed_aspect=unknown adaptive_diagonals=unknown vertices=12/24 '
                'invalid=unknown repaired=unknown repair_rounds=unknown retention=unknown '
                'thresholds=jump:0.35,footprint:12,aspect:10',
            ])
            metadata = json.loads((captured['run_dir'] / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['mesh_export']['status'], 'failed')
            self.assertEqual(metadata['mesh_export']['error'], 'empty mesh')
            self.assertEqual(metadata['mesh_export']['stats'], error_stats)
            self.assertEqual(metadata['reconstruction']['mode'], gen.PANO_TO_3D_EQUIRECTANGULAR)

    def test_scene_mesh_failure_without_stats_logs_unknowns(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs_dir = root / 'outputs'
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / 'models' / 'dreamcube', outputs_dir)
            instance.node_id = gen.SCENE_NODE_ID
            captured: dict[str, Path] = {}

            def auto_depth(image_path, run_dir, variant):
                path = run_dir / 'depth.png'
                path.write_bytes(b'depth')
                return path

            def outputs(*, predictions, run_dir, save_all_outputs):
                captured['run_dir'] = run_dir
                path = run_dir / 'output_equi_rgb.png'
                path.write_bytes(b'png')
                return {'equi_rgb_path': path}

            error = gen.dreamcube_mesh.MeshExportError('empty mesh')
            params = prompt_params()
            params.update({
                'node_id': gen.SCENE_NODE_ID,
                'mesh_depth_jump_threshold': 0.2,
            })
            stderr = io.StringIO()
            with mock.patch.object(instance, '_save_auto_depth_image', side_effect=auto_depth), \
                 mock.patch.object(instance, 'load'), \
                 mock.patch.object(instance, '_build_seed_kwargs', return_value={}), \
                 mock.patch.object(instance, '_run_inference', return_value={}), \
                 mock.patch.object(instance, '_save_postprocessed_outputs', side_effect=outputs), \
                 mock.patch.object(instance, '_export_scene', side_effect=error), \
                 contextlib.redirect_stderr(stderr):
                with self.assertRaises(gen.dreamcube_mesh.MeshExportError) as raised:
                    instance.generate(tiny_png_bytes(), params)

            self.assertIs(raised.exception, error)
            summaries = [line for line in stderr.getvalue().splitlines() if 'Mesh export failed' in line]
            self.assertEqual(summaries, [
                '[dreamcube] Mesh export failed mode=unknown triangles=unknown/unknown '
                'removed_invalid=unknown removed_jump=unknown removed_footprint=unknown '
                'removed_aspect=unknown adaptive_diagonals=unknown vertices=unknown/unknown '
                'invalid=unknown repaired=unknown repair_rounds=unknown retention=unknown '
                'thresholds=jump:0.2,footprint:12,aspect:10',
            ])
            metadata = json.loads((captured['run_dir'] / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['mesh_export'], {
                'status': 'failed',
                'error': 'empty mesh',
                'stats': None,
            })
