from __future__ import annotations

import contextlib
import io
import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

EXT_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rgb_bytes(size: int = 4) -> bytes:
    image = Image.new('RGB', (size, size), (20, 40, 60))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def write_rgb(path: Path, size: int = 4) -> None:
    Image.new('RGB', (size, size), (80, 100, 120)).save(path)


def write_depth(path: Path, size: int = 4, value: int = 2000) -> None:
    arr = np.full((size, size), value, dtype=np.uint16)
    Image.fromarray(np.asarray(arr, dtype=np.uint16)).save(path)

def png_ihdr_bit_depth_and_color_type(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:29]
    assert header[:8] == b'\x89PNG\r\n\x1a\n'
    assert header[12:16] == b'IHDR'
    return int(header[24]), int(header[25])



def prompt_params() -> dict[str, str]:
    return {
        'prompt_front': 'front',
        'prompt_right': 'right',
        'prompt_back': 'back',
        'prompt_left': 'left',
        'prompt_top': 'top',
        'prompt_bottom': 'bottom',
    }


def manual_params(root: Path, size: int = 4) -> dict[str, str]:
    params = prompt_params()
    for face in ['right', 'back', 'left', 'top', 'bottom']:
        path = root / f'{face}_rgb.png'
        write_rgb(path, size=size)
        params[f'rgb_{face}_path'] = str(path)
    for face in ['front', 'right', 'back', 'left', 'top', 'bottom']:
        path = root / f'{face}_depth.png'
        write_depth(path, size=size)
        params[f'depth_{face}_path'] = str(path)
    return params


def physical_cube_depths(size: int) -> dict[str, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing='xy')
    y_down = -yy
    coords = {
        'front': (-xx, y_down, np.ones_like(xx)),
        'right': (-np.ones_like(xx), y_down, -xx),
        'back': (xx, y_down, -np.ones_like(xx)),
        'left': (np.ones_like(xx), y_down, xx),
        'top': (-xx, np.ones_like(xx), yy),
        'bottom': (-xx, -np.ones_like(xx), -yy),
    }
    return {
        face: np.rint(30000 + 1000 * x + 100 * y + 10 * z).astype(np.uint16)
        for face, (x, y, z) in coords.items()
    }


class ManualCubemapHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manual = load_module('dreamcube_manual_cubemap_test', EXT_DIR / 'dreamcube_manual_cubemap.py')

    def test_face_order_axes_and_prompt_prefixes_are_official(self):
        self.assertEqual(self.manual.FACE_ORDER, ('front', 'right', 'back', 'left', 'top', 'bottom'))
        self.assertEqual([self.manual.FACE_AXES[f] for f in self.manual.FACE_ORDER], ['+Z', '-X', '-Z', '+X', '+Y', '-Y'])
        self.assertEqual(self.manual.prefixed_prompts(['a', 'b', 'c', 'd', 'e', 'f']), [
            'This is one view of a scene. a',
            'This is one view of a scene. b',
            'This is one view of a scene. c',
            'This is one view of a scene. d',
            'This a upward view of a scene. e',
            'This a downward view of a scene. f',
        ])

    def test_strict_depth_format_dimensions_and_invalid_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = manual_params(root)
            bad = root / 'bad_depth.png'
            Image.new('RGB', (4, 4), (1, 2, 3)).save(bad)
            params['depth_front_path'] = str(bad)
            with self.assertRaises(ValueError) as raised:
                self.manual.load_manual_cubemap_inputs(front_rgb_bytes=rgb_bytes(), params=params, outputs_dir=root)
            self.assertIn('single-channel 16-bit', str(raised.exception))

            params = manual_params(root)
            bad_size = root / 'bad_size.png'
            write_depth(bad_size, size=3)
            params['depth_front_path'] = str(bad_size)
            with self.assertRaises(ValueError) as raised:
                self.manual.load_manual_cubemap_inputs(front_rgb_bytes=rgb_bytes(), params=params, outputs_dir=root)
            self.assertIn('size', str(raised.exception))

            params = manual_params(root, size=10)
            arr = np.full((10, 10), 2000, dtype=np.uint16)
            arr[:2, :] = 0
            invalid = root / 'invalid.png'
            Image.fromarray(np.asarray(arr, dtype=np.uint16)).save(invalid)
            params['depth_front_path'] = str(invalid)
            with self.assertRaises(ValueError) as raised:
                self.manual.load_manual_cubemap_inputs(front_rgb_bytes=rgb_bytes(10), params=params, outputs_dir=root)
            self.assertIn('exceeds 1%', str(raised.exception))

    def test_depth_seam_topology_is_orientation_aware(self):
        expected_pairs = (
            ('front', 'right', 'right', 'left', False),
            ('front', 'left', 'left', 'right', False),
            ('front', 'top', 'top', 'bottom', False),
            ('front', 'bottom', 'bottom', 'top', False),
            ('back', 'right', 'left', 'left', False),
            ('back', 'left', 'right', 'right', False),
            ('back', 'top', 'top', 'top', True),
            ('back', 'bottom', 'bottom', 'bottom', True),
            ('right', 'top', 'top', 'right', True),
            ('right', 'bottom', 'bottom', 'right', False),
            ('left', 'top', 'top', 'left', False),
            ('left', 'bottom', 'bottom', 'left', True),
        )
        self.assertEqual(self.manual.EDGE_PAIRS, expected_pairs)

        depths = physical_cube_depths(16)
        metrics = self.manual.validate_depth_seams(depths)
        self.assertEqual(metrics['max_p95_relative_mismatch'], 0.0)

        original_pairs = self.manual.EDGE_PAIRS
        try:
            self.manual.EDGE_PAIRS = tuple((a_face, a_edge, b_face, b_edge, False) for a_face, a_edge, b_face, b_edge, _ in original_pairs)
            no_reverse_metrics = self.manual.validate_depth_seams(depths)
        finally:
            self.manual.EDGE_PAIRS = original_pairs
        self.assertGreater(no_reverse_metrics['max_p95_relative_mismatch'], 0.01)

    def test_manual_domain_has_no_mask_helper(self):
        mask_helpers = [
            name
            for name, value in vars(self.manual).items()
            if 'mask' in name.lower() and callable(value)
        ]
        self.assertEqual(mask_helpers, [])

    def test_depth_mismatch_threshold(self):
        good = {face: np.full((4, 4), 1000, dtype=np.uint16) for face in self.manual.FACE_ORDER}
        metrics = self.manual.validate_depth_seams(good)
        self.assertEqual(len(metrics['edges']), 12)
        self.assertEqual(metrics['max_p95_relative_mismatch'], 0.0)
        bad = dict(good)
        bad['right'] = np.full((4, 4), 3000, dtype=np.uint16)
        metrics = self.manual.validate_depth_seams(bad)
        self.assertGreater(metrics['max_p95_relative_mismatch'], 0.50)

    def test_workspace_relative_manual_picker_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outputs = root / 'outputs'
            workspace = root / 'workspace'
            outputs.mkdir()
            workspace.mkdir()
            image_path = workspace / 'picked.png'
            write_rgb(image_path)
            with mock.patch.dict('os.environ', {'WORKSPACE_DIR': str(workspace)}):
                resolved = self.manual.resolve_existing_path('picked.png', outputs, label='rgb_right_path')
            self.assertEqual(resolved, image_path.resolve())

    def test_resolve_existing_path_expands_environment_and_reports_checked_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / 'env-picked.png'
            write_rgb(target)
            with mock.patch.dict('os.environ', {'DREAMCUBE_PICK': str(target)}):
                self.assertEqual(
                    self.manual.resolve_existing_path('$DREAMCUBE_PICK', root / 'outputs', label='rgb_right_path'),
                    target.resolve(),
                )
            with self.assertRaises(FileNotFoundError) as raised:
                self.manual.resolve_existing_path('missing.png', root / 'outputs', label='rgb_right_path')
            message = str(raised.exception)
            self.assertIn('Checked:', message)
            self.assertIn('missing.png', message)

    def test_deterministic_input_sidecars(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inputs = self.manual.load_manual_cubemap_inputs(front_rgb_bytes=rgb_bytes(), params=manual_params(root), outputs_dir=root)
            self.manual.save_input_faces(root, inputs)
            for face in self.manual.FACE_ORDER:
                self.assertTrue((root / 'input_faces' / f'{face}_rgb.png').is_file())
                self.assertTrue((root / 'input_faces' / f'{face}_depth_mm.png').is_file())


class ManualCubemapGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_module('dreamcube_generator_manual_test', EXT_DIR / 'generator.py')

    def test_manifest_version_shared_weights_and_manual_schema(self):
        manifest = json.loads((EXT_DIR / 'manifest.json').read_text())
        self.assertEqual(manifest['version'], '0.3.0')
        self.assertEqual(len(manifest['nodes']), 4)
        node = next(n for n in manifest['nodes'] if n['id'] == 'generate-scene-manual-cubemap')
        self.assertEqual(node['capability_id'], 'rgbd-cubemap-to-scene')
        self.assertEqual(node['input'], 'image')
        self.assertEqual(node['output'], 'mesh')
        self.assertEqual(node['weight_owner_id'], manifest['weight_owner_id'])
        self.assertEqual(node['hf_repo'], manifest['hf_repo'])
        self.assertEqual(node['download_check'], manifest['download_check'])
        ids = {p['id'] for p in node['params_schema']}
        self.assertEqual(ids, {
            'rgb_right_path',
            'rgb_back_path',
            'rgb_left_path',
            'rgb_top_path',
            'rgb_bottom_path',
            'depth_front_path',
            'depth_right_path',
            'depth_back_path',
            'depth_left_path',
            'depth_top_path',
            'depth_bottom_path',
            'prompt_front',
            'prompt_right',
            'prompt_back',
            'prompt_left',
            'prompt_top',
            'prompt_bottom',
            'num_inference_steps',
            'guidance_scale',
            'normalize_scale',
            'max_cube_size',
            'mesh_depth_jump_threshold',
            'mesh_footprint_ratio_threshold',
            'mesh_aspect_ratio_threshold',
            'save_all_outputs',
            'seed',
        })
        self.assertEqual(len([i for i in ids if i.startswith('rgb_') or i.startswith('depth_')]), 11)
        self.assertNotIn('depth_mode', ids)
        self.assertNotIn('auto_depth_variant', ids)
        self.assertNotIn('output_format', ids)
        max_cube_size = next(p for p in node['params_schema'] if p['id'] == 'max_cube_size')
        self.assertEqual(max_cube_size['min'], 256)
        self.assertEqual(max_cube_size['max'], 512)
        self.assertTrue((EXT_DIR / 'dreamcube_manual_cubemap.py').is_file())
        self.assertIn('dreamcube_manual_cubemap.py', (EXT_DIR / 'README.md').read_text(encoding='utf-8'))

    def test_manual_node_calls_loaded_pipeline_directly_with_six_pairs_and_mask(self):
        class FakeTensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            @property
            def shape(self):
                return self.value.shape

            @property
            def dtype(self):
                return self.value.dtype

            def to(self, _device):
                return self

            def permute(self, *axes):
                return FakeTensor(np.transpose(self.value, axes))

            def unsqueeze(self, axis):
                return FakeTensor(np.expand_dims(self.value, axis))

            def contiguous(self):
                return self

            def any(self):
                return self.value.any()

        zero_like_calls = []
        fake_torch = types.ModuleType('torch')
        fake_torch.bool = np.bool_
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.inference_mode = contextlib.nullcontext
        fake_torch.from_numpy = FakeTensor

        def zeros_like(value, *, dtype):
            zero_like_calls.append((value, dtype))
            return FakeTensor(np.zeros_like(value.value, dtype=dtype))

        fake_torch.zeros_like = zeros_like
        gen = self.generator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            instance = gen.DreamCubeGenerator(root / 'model', root)
            instance._app = types.SimpleNamespace(depth_to_z_distance=lambda depth, fov_x, fov_y: depth)
            calls = {}
            class Pipe:
                device = 'cpu'
                def __call__(self, **kwargs):
                    calls.update(kwargs)
                    size = kwargs['height']
                    return types.SimpleNamespace(
                        images=np.zeros((6, size, size, 3), dtype=np.float32),
                        depths=np.ones((6, size, size, 1), dtype=np.float32),
                    )
            instance._pipe = Pipe()
            inputs = gen.dreamcube_manual_cubemap.load_manual_cubemap_inputs(front_rgb_bytes=rgb_bytes(), params=manual_params(root), outputs_dir=root)
            with mock.patch.dict(sys.modules, {'torch': fake_torch}):
                predictions, diag = instance._run_manual_cubemap_inference(
                    inputs=inputs,
                    prompts=['a','b','c','d','e','f'],
                    num_inference_steps=5,
                    guidance_scale=7.5,
                    normalize_scale=0.6,
                    max_cube_size=4,
                    seed_kwargs={},
                )
            self.assertEqual(list(calls['cube_rgbs'].shape), [1, 6, 3, 256, 256])
            self.assertEqual(list(calls['cube_depths'].shape), [1, 6, 1, 256, 256])
            boundary_cube_masks = calls['cube_masks']
            self.assertEqual(list(boundary_cube_masks.shape), [1, 6, 1, 256, 256])
            self.assertEqual(boundary_cube_masks.dtype, np.dtype(np.bool_))
            self.assertFalse(boundary_cube_masks.any().item())
            self.assertEqual(len(zero_like_calls), 1)
            self.assertIs(zero_like_calls[0][0], calls['cube_depths'])
            self.assertIs(zero_like_calls[0][1], np.bool_)
            self.assertEqual(calls['prompt'][4], 'This a upward view of a scene. e')
            self.assertEqual(predictions['images'].shape, (1, 6, 256, 256, 3))
            self.assertEqual(diag['conditioning_mode'], 'manual-rgbd-cubemap')
            self.assertEqual(set(diag), {
                'conditioning_mode',
                'inference_size',
                'cube_rgbs_shape',
                'cube_depths_shape',
                'cube_masks_shape',
                'prompt_prefixes',
            })

    def test_manual_generator_signature_has_only_fixed_conditioning_inputs(self):
        parameters = inspect.signature(
            self.generator.DreamCubeGenerator._run_manual_cubemap_inference
        ).parameters
        self.assertEqual(tuple(parameters), (
            'self',
            'inputs',
            'prompts',
            'num_inference_steps',
            'guidance_scale',
            'normalize_scale',
            'max_cube_size',
            'seed_kwargs',
        ))

    def test_manual_generation_failure_metadata_uses_manual_section(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            instance = gen.DreamCubeGenerator(root / 'model', root)
            params = prompt_params()
            with self.assertRaises(ValueError):
                instance._generate_manual_cubemap(rgb_bytes(), params, {}, None, None)
            run_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith('dreamcube-')]
            self.assertEqual(len(run_dirs), 1)
            metadata = json.loads((run_dirs[0] / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertNotIn('mesh_export', metadata)
            self.assertEqual(metadata['manual_cubemap']['status'], 'failed')
            self.assertEqual(metadata['manual_cubemap']['stage'], 'manual-cubemap')

    def test_no_auto_depth_or_app_inference_for_manual_generation(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outputs = root / 'outputs'
            outputs.mkdir()
            instance = gen.DreamCubeGenerator(root / 'model', outputs)
            params = manual_params(root)
            instance.node_id = gen.MANUAL_SCENE_NODE_ID
            result_path = outputs / 'run' / 'scene-manifest.json'
            result_path.parent.mkdir()
            with mock.patch.object(instance, '_save_auto_depth_image', side_effect=AssertionError('auto-depth must not run')), \
                 mock.patch.object(instance, '_run_inference', side_effect=AssertionError('app inference fallback must not run')), \
                 mock.patch.object(instance, '_generate_manual_cubemap', return_value=result_path) as manual:
                result = instance.generate(rgb_bytes(), params)
            self.assertEqual(result, result_path)
            manual.assert_called_once()

    def test_manual_generation_returns_glb_and_records_scene_manifest_sidecar(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outputs_dir = root / 'outputs'
            outputs_dir.mkdir()
            instance = gen.DreamCubeGenerator(root / 'model', outputs_dir)
            params = manual_params(root)

            def save_outputs(*, predictions, run_dir, save_all_outputs):
                path = run_dir / 'output_equi_rgb.png'
                path.write_bytes(b'png')
                return {
                    'equi_rgb_path': path,
                    'depths_distance': np.ones((1, 6, 1, 1, 1), dtype=np.float32),
                }

            def export(**kwargs):
                path = kwargs['run_dir'] / 'output_mesh.glb'
                path.write_bytes(b'glb')
                return path, {'mode': 'cubemap'}

            with mock.patch.dict('os.environ', {'WORKSPACE_DIR': str(root)}), \
                 mock.patch.object(instance, 'load'), \
                 mock.patch.object(instance, '_build_seed_kwargs', return_value={}), \
                 mock.patch.object(instance, '_run_manual_cubemap_inference', return_value=({}, {'status': 'ok'})), \
                 mock.patch.object(instance, '_save_postprocessed_outputs', side_effect=save_outputs), \
                 mock.patch.object(instance, '_save_output_faces', return_value={'status': 'saved'}), \
                 mock.patch.object(gen.dreamcube_manual_cubemap, 'validate_depth_seams', return_value={'edges': [], 'max_p95_relative_mismatch': 0.0}), \
                 mock.patch.object(instance, '_export_scene', side_effect=export):
                result = instance._generate_manual_cubemap(rgb_bytes(), params, {}, None, None)

            self.assertEqual(result.name, 'output_mesh.glb')
            self.assertTrue(result.is_absolute())
            manifest_path = result.parent / 'scene-manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['schema'], 'modly.scene-manifest.v1')
            self.assertEqual(manifest['assets'][0]['workspacePath'], f'outputs/{result.parent.name}/output_mesh.glb')
            metadata = json.loads((result.parent / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(set(metadata), {
                'created_at',
                'node_id',
                'output_format',
                'conditioning_mode',
                'face_order',
                'face_axes',
                'input_faces',
                'seam_metrics_preflight',
                'prompts',
                'reconstruction',
                'coordinate_frame',
                'presentation',
                'manual_cubemap',
                'output_faces',
                'seam_metrics_postprocess',
                'primary_output',
                'mesh_export',
                'scene_manifest',
            })
            self.assertEqual(metadata['primary_output']['path'], 'output_mesh.glb')
            self.assertEqual(metadata['scene_manifest']['path'], 'scene-manifest.json')

    def test_output_face_sidecars_are_deterministic(self):
        gen = self.generator
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            instance = gen.DreamCubeGenerator(root / 'model', root)
            outputs = {
                'images_pred': np.full((1, 6, 2, 2, 3), 127, dtype=np.uint8),
                'depths_distance': np.full((1, 6, 2, 2, 1), 1500, dtype=np.float32),
            }
            result = instance._save_output_faces(outputs=outputs, run_dir=root)
            self.assertEqual(result['status'], 'saved')
            for face in gen.dreamcube_manual_cubemap.FACE_ORDER:
                self.assertTrue((root / 'output_faces' / f'{face}_rgb.png').is_file())
                depth_path = root / 'output_faces' / f'{face}_depth_mm.png'
                self.assertTrue(depth_path.is_file())
                self.assertEqual(png_ihdr_bit_depth_and_color_type(depth_path), (16, 0))


if __name__ == '__main__':
    unittest.main()
