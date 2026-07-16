from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


EXT_DIR = Path(__file__).resolve().parents[1]
SETUP_PATH = EXT_DIR / "setup.py"


def load_setup_module():
    spec = importlib.util.spec_from_file_location("dreamcube_setup", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeLogger:
    def info(self, message: str) -> None:
        pass


class SetupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = load_setup_module()

    def test_run_command_defaults_to_checked_and_raises_for_failing_child(self):
        signature = inspect.signature(self.setup.run_command)
        self.assertIs(signature.parameters["check"].default, True)

        with self.assertRaises(self.setup.SetupError) as ctx:
            self.setup.run_command(
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                mock.Mock(),
                code="child-command-failed",
            )

        self.assertEqual(ctx.exception.code, "child-command-failed")
        self.assertIn("exit code 7", str(ctx.exception))

    def test_parse_electron_json_payload(self):
        payload = {
            "python_exe": sys.executable,
            "ext_dir": str(EXT_DIR),
            "gpu_sm": 86,
            "cuda_version": 128,
            "model_dir": "custom-models/dreamcube",
        }

        config = self.setup.parse_args([json.dumps(payload)], env={})

        self.assertEqual(config.python_exe, sys.executable)
        self.assertEqual(config.ext_dir, EXT_DIR)
        self.assertEqual(config.gpu_sm, 86)
        self.assertEqual(config.cuda_version, 128)
        self.assertEqual(config.model_dir, EXT_DIR / "custom-models" / "dreamcube")
        self.assertFalse(config.validate_only)

    def test_parse_legacy_positional_payload(self):
        config = self.setup.parse_args([sys.executable, str(EXT_DIR), "8.6", "12.4"], env={})

        self.assertEqual(config.python_exe, sys.executable)
        self.assertEqual(config.ext_dir, EXT_DIR)
        self.assertEqual(config.gpu_sm, 86)
        self.assertEqual(config.cuda_version, 124)
        self.assertEqual(config.model_dir, EXT_DIR / "models" / "dreamcube" / "dreamcube")

    def test_model_dir_uses_modly_settings_when_payload_has_no_models_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_root = Path(temp_dir) / "config"
            settings_dir = config_root / "Modly"
            settings_dir.mkdir(parents=True)
            models_dir = Path(temp_dir) / "Modly" / "models"
            (settings_dir / "settings.json").write_text(
                json.dumps({"modelsDir": str(models_dir)}),
                encoding="utf-8",
            )

            config = self.setup.parse_args(
                [sys.executable, str(EXT_DIR), "8.6", "12.4"],
                env={"XDG_CONFIG_HOME": str(config_root)},
            )

            self.assertEqual(config.model_dir, models_dir / "dreamcube" / "dreamcube")

    def test_validate_only_without_payload_is_local_and_lightweight(self):
        config = self.setup.parse_args(["--validate-only"], env={})

        self.assertTrue(config.validate_only)
        self.assertEqual(config.ext_dir, EXT_DIR)
        details = self.setup.validate_internal_config(config)
        self.assertEqual(details["model_dir"], str(EXT_DIR / "models" / "dreamcube" / "dreamcube"))
        self.assertEqual(details["auto_depth_provider"]["repo_id"], "depth-anything/Depth-Anything-V2-Small-hf")
        self.assertEqual(details["auto_depth_provider"]["default_variant"], "vits")
        self.assertFalse(details["auto_depth_provider"]["managed_by_setup"])

    def test_model_dir_prefers_env_direct_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup.parse_args(
                [json.dumps({"python_exe": sys.executable, "ext_dir": str(EXT_DIR), "gpu_sm": 86})],
                env={"MODLY_DREAMCUBE_MODEL_DIR": temp_dir},
            )

            self.assertEqual(config.model_dir, Path(temp_dir))

    def test_managed_model_directory_does_not_require_downloaded_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "models" / "dreamcube" / "dreamcube"
            config = self.setup.SetupConfig(
                python_exe=sys.executable,
                ext_dir=EXT_DIR,
                gpu_sm=86,
                cuda_version=128,
                model_dir=model_dir,
            )

            details = self.setup.ensure_managed_model_directory(config, _FakeLogger())

            self.assertTrue(model_dir.is_dir())
            self.assertEqual(details["model_dir"], str(model_dir))
            self.assertFalse(details["download_check_exists"])
            self.assertEqual(details["weights_managed_by"], "modly-ui")
            self.assertIn("Modly UI", details["message"])

    def test_setup_contract_does_not_call_hf_snapshot_fetch(self):
        source = SETUP_PATH.read_text(encoding="utf-8")

        self.assertNotIn("snapshot" + "_download", source)
        self.assertNotIn("model-snapshot", source)
        self.assertIn("model-directory", source)

    def test_auto_depth_setup_contract_is_internal_and_lazy(self):
        manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        runtime_cache = manifest["setup"]["managed_runtime_cache"]
        auto_depth = runtime_cache["auto_depth"]

        self.assertEqual(auto_depth["repo_id"], "depth-anything/Depth-Anything-V2-Small-hf")
        self.assertEqual(auto_depth["default_variant"], "vits")
        self.assertFalse(auto_depth["managed_by_setup"])
        self.assertIn(".modly/auto-depth/cache", auto_depth["path"])
        manifest_text = json.dumps(manifest)
        self.assertIn("modly-depth-anything", manifest_text)
        self.assertNotIn('"dependency": "modly-depth-anything"', manifest_text)
        self.assertNotIn('"depends_on": "modly-depth-anything"', manifest_text)
        self.assertIn("transformers", {self.setup.dependency_name(package) for package in self.setup.UPSTREAM_DEPENDENCIES})
        self.assertIn("numpy", {self.setup.dependency_name(package) for package in self.setup.UPSTREAM_DEPENDENCIES})
        self.assertIn("pillow", {self.setup.dependency_name(package) for package in self.setup.UPSTREAM_DEPENDENCIES})



    def test_upstream_constants_pin_tested_dreamcube_revision(self):
        self.assertEqual(self.setup.UPSTREAM_REPO_URL, "https://github.com/Yukun-Huang/DreamCube.git")
        self.assertEqual(self.setup.UPSTREAM_REF, "main")
        self.assertEqual(self.setup.UPSTREAM_COMMIT, "aa04a53c6542581b5b0a6faa575865d2d57b5243")

        manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
        upstream = manifest["setup"]["managed_runtime_cache"]["upstream_source"]
        self.assertEqual(upstream["repo_url"], self.setup.UPSTREAM_REPO_URL)
        self.assertEqual(upstream["ref"], self.setup.UPSTREAM_REF)
        self.assertEqual(upstream["commit"], self.setup.UPSTREAM_COMMIT)
        self.assertEqual(upstream["pinned_revision"], self.setup.UPSTREAM_COMMIT)
        self.assertEqual(upstream["checkout"], "detached")

    def test_sync_upstream_fresh_clone_checks_out_pinned_commit_detached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup.SetupConfig(
                python_exe=sys.executable,
                ext_dir=Path(temp_dir),
                gpu_sm=86,
                cuda_version=128,
                model_dir=Path(temp_dir) / "models",
            )
            calls = []

            def fake_run(cmd, logger, **kwargs):
                calls.append((list(cmd), kwargs.get("cwd")))
                if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                    return self.setup.subprocess.CompletedProcess(cmd, 0, self.setup.UPSTREAM_COMMIT + "\n", None)
                return self.setup.subprocess.CompletedProcess(cmd, 0, "", None)

            with mock.patch.object(self.setup, "run_command", side_effect=fake_run):
                details = self.setup.sync_upstream(config, _FakeLogger())

            upstream_dir = Path(temp_dir) / self.setup.UPSTREAM_RELATIVE_PATH
            self.assertEqual(details["commit"], self.setup.UPSTREAM_COMMIT)
            self.assertIn((["git", "clone", "--no-checkout", "--branch", "main", "--single-branch", self.setup.UPSTREAM_REPO_URL, str(upstream_dir)], None), calls)
            self.assertIn((["git", "checkout", "--detach", self.setup.UPSTREAM_COMMIT], upstream_dir), calls)
            self.assertNotIn((["git", "checkout", "-B", "main", "origin/main"], upstream_dir), calls)

    def test_sync_upstream_existing_checkout_repairs_and_verifies_pinned_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            upstream_dir = Path(temp_dir) / self.setup.UPSTREAM_RELATIVE_PATH
            (upstream_dir / ".git").mkdir(parents=True)
            config = self.setup.SetupConfig(
                python_exe=sys.executable,
                ext_dir=Path(temp_dir),
                gpu_sm=86,
                cuda_version=128,
                model_dir=Path(temp_dir) / "models",
            )
            calls = []

            def fake_run(cmd, logger, **kwargs):
                calls.append((list(cmd), kwargs.get("cwd")))
                if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                    return self.setup.subprocess.CompletedProcess(cmd, 0, self.setup.UPSTREAM_COMMIT + "\n", None)
                return self.setup.subprocess.CompletedProcess(cmd, 0, "", None)

            with mock.patch.object(self.setup, "run_command", side_effect=fake_run):
                details = self.setup.sync_upstream(config, _FakeLogger())

            self.assertEqual(details["commit"], self.setup.UPSTREAM_COMMIT)
            self.assertIn((["git", "remote", "set-url", "origin", self.setup.UPSTREAM_REPO_URL], upstream_dir), calls)
            self.assertIn((["git", "fetch", "--prune", "origin", self.setup.UPSTREAM_REF], upstream_dir), calls)
            self.assertIn((["git", "fetch", "origin", self.setup.UPSTREAM_COMMIT], upstream_dir), calls)
            self.assertIn((["git", "checkout", "--detach", self.setup.UPSTREAM_COMMIT], upstream_dir), calls)

    def test_sync_upstream_rejects_wrong_final_head(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.setup.SetupConfig(
                python_exe=sys.executable,
                ext_dir=Path(temp_dir),
                gpu_sm=86,
                cuda_version=128,
                model_dir=Path(temp_dir) / "models",
            )

            def fake_run(cmd, logger, **kwargs):
                if cmd[:3] == ["git", "rev-parse", "HEAD"]:
                    return self.setup.subprocess.CompletedProcess(cmd, 0, "0" * 40 + "\n", None)
                return self.setup.subprocess.CompletedProcess(cmd, 0, "", None)

            with mock.patch.object(self.setup, "run_command", side_effect=fake_run):
                with self.assertRaises(self.setup.SetupError) as ctx:
                    self.setup.sync_upstream(config, _FakeLogger())

            self.assertEqual(ctx.exception.code, "upstream-pinned-commit-mismatch")

    def test_dependencies_exclude_provider_managed_packages(self):
        self.assertNotIn(self.setup.PYTORCH3D_UPSTREAM_PACKAGE, self.setup.UPSTREAM_DEPENDENCIES)
        self.assertFalse(any(self.setup.dependency_name(package) == self.setup.OPEN3D_PACKAGE for package in self.setup.UPSTREAM_DEPENDENCIES))
        self.assertIn("pytorch3d.transforms", self.setup.PROBE_IMPORTS)
        self.assertIn("open3d", self.setup.PROBE_IMPORTS)

    def test_pytorch3d_source_url_is_immutable_v078_tag(self):
        self.assertEqual(
            self.setup.PYTORCH3D_SOURCE_URL,
            "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8",
        )
        self.assertNotIn("@stable", self.setup.PYTORCH3D_SOURCE_URL)


    def test_pytorch3d_mode_defaults_and_payload_env_precedence(self):
        base = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
        )
        self.assertEqual(self.setup.resolve_pytorch3d_mode(base, env={}), "auto")

        env_config = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
        )
        self.assertEqual(self.setup.resolve_pytorch3d_mode(env_config, env={"DREAMCUBE_PYTORCH3D_MODE": "SHIM"}), "shim")

        payload_config = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
            payload={"pytorch3dMode": "required"},
        )
        self.assertEqual(self.setup.resolve_pytorch3d_mode(payload_config, env={"DREAMCUBE_PYTORCH3D_MODE": "shim"}), "required")

    def test_shim_writes_package_and_identity_quaternion(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is required to execute the shim contract")

        with tempfile.TemporaryDirectory() as temp_dir:
            site_packages = Path(temp_dir)
            with mock.patch.object(self.setup, "extension_site_packages", return_value=site_packages), mock.patch.object(
                self.setup,
                "probe_pytorch3d_provider",
                return_value={"importable": True, "is_shim": True},
            ):
                details = self.setup.install_pytorch3d_shim(Path(sys.executable), _FakeLogger(), mode="shim", reason="test")

            self.assertEqual(details["provider"], "shim")
            self.assertTrue((site_packages / "pytorch3d" / "__init__.py").is_file())
            self.assertTrue((site_packages / "pytorch3d" / "transforms" / "__init__.py").is_file())

            for name in ["pytorch3d.transforms", "pytorch3d"]:
                sys.modules.pop(name, None)
            sys.path.insert(0, str(site_packages))
            try:
                import torch
                from pytorch3d.transforms import matrix_to_quaternion

                quat = matrix_to_quaternion(torch.eye(3).reshape(1, 3, 3))[0]
                self.assertEqual(quat.tolist(), [1.0, 0.0, 0.0, 0.0])
            finally:
                sys.path.remove(str(site_packages))
                for name in ["pytorch3d.transforms", "pytorch3d"]:
                    sys.modules.pop(name, None)

    def test_required_mode_fails_without_real_pytorch3d(self):
        class _Tracker:
            data = {"pytorch3d_provider": None}

            def write(self):
                pass

        config = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
            payload={"pytorch3d_mode": "required"},
        )
        with mock.patch.object(self.setup, "probe_pytorch3d_provider", return_value={"importable": False}):
            with self.assertRaises(self.setup.SetupError):
                self.setup.install_pytorch3d_provider(config, Path(sys.executable), _FakeLogger(), _Tracker())


    def test_open3d_mode_defaults_and_payload_env_precedence(self):
        base = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
        )
        self.assertEqual(self.setup.resolve_open3d_mode(base, env={}), "auto")

        env_config = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
        )
        self.assertEqual(self.setup.resolve_open3d_mode(env_config, env={"DREAMCUBE_OPEN3D_MODE": "SHIM"}), "shim")

        payload_config = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
            payload={"open3dMode": "required"},
        )
        self.assertEqual(self.setup.resolve_open3d_mode(payload_config, env={"DREAMCUBE_OPEN3D_MODE": "shim"}), "required")

    def test_open3d_shim_writes_package_obj_mesh_and_ply_point_cloud(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is required to execute the Open3D shim contract")

        with tempfile.TemporaryDirectory() as temp_dir:
            site_packages = Path(temp_dir)
            with mock.patch.object(self.setup, "extension_site_packages", return_value=site_packages), mock.patch.object(
                self.setup,
                "probe_open3d_provider",
                return_value={"importable": True, "is_shim": True},
            ):
                details = self.setup.install_open3d_shim(Path(sys.executable), _FakeLogger(), mode="shim", reason="test")

            self.assertEqual(details["provider"], "shim")
            self.assertTrue((site_packages / "open3d" / "__init__.py").is_file())
            self.assertTrue((site_packages / "open3d" / "geometry.py").is_file())
            self.assertTrue((site_packages / "open3d" / "io.py").is_file())

            for name in ["open3d.io", "open3d.utility", "open3d.geometry", "open3d"]:
                sys.modules.pop(name, None)
            sys.path.insert(0, str(site_packages))
            try:
                import open3d as o3d

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
                mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
                mesh.vertex_colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                obj_path = Path(temp_dir) / "mesh.obj"
                self.assertTrue(o3d.io.write_triangle_mesh(obj_path, mesh))
                obj_text = obj_path.read_text(encoding="utf-8")
                self.assertIn("v 0.0 0.0 0.0", obj_text)
                self.assertIn("f 1 2 3", obj_text)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 1, 1]])
                pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0]])
                ply_path = Path(temp_dir) / "points.ply"
                self.assertTrue(o3d.io.write_point_cloud(ply_path, pcd))
                ply_text = ply_path.read_text(encoding="utf-8")
                self.assertIn("ply", ply_text)
                self.assertIn("element vertex 2", ply_text)
            finally:
                sys.path.remove(str(site_packages))
                for name in ["open3d.io", "open3d.utility", "open3d.geometry", "open3d"]:
                    sys.modules.pop(name, None)

    def test_required_mode_fails_without_real_open3d(self):
        class _Tracker:
            data = {"open3d_provider": None}

            def write(self):
                pass

        config = self.setup.SetupConfig(
            python_exe=sys.executable,
            ext_dir=EXT_DIR,
            gpu_sm=86,
            cuda_version=128,
            model_dir=EXT_DIR / "models" / "dreamcube" / "dreamcube",
            payload={"open3d_mode": "required"},
        )
        with mock.patch.object(self.setup, "probe_open3d_provider", return_value={"importable": False}):
            with self.assertRaises(self.setup.SetupError):
                self.setup.install_open3d_provider(config, Path(sys.executable), _FakeLogger(), _Tracker())

    def test_redacts_tokens_and_secret_assignments(self):
        text = (
            "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz "
            "Authorization: Bearer abc.def.ghi "
            "https://user:password@secret.invalid/path "
            "api_key=plain-secret"
        )

        redacted = self.setup.redact_text(text)

        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz", redacted)
        self.assertNotIn("abc.def.ghi", redacted)
        self.assertNotIn("user:password", redacted)
        self.assertNotIn("plain-secret", redacted)
        self.assertIn("[REDACTED]", redacted)


if __name__ == "__main__":
    unittest.main()

class MeshPayloadContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = load_setup_module()

    def test_validate_only_requires_extension_owned_mesh_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            payload_dir = Path(temp_dir)
            (payload_dir / 'manifest.json').write_text((EXT_DIR / 'manifest.json').read_text(encoding='utf-8'), encoding='utf-8')
            (payload_dir / 'generator.py').write_text('', encoding='utf-8')
            config = self.setup.SetupConfig(
                python_exe=sys.executable,
                ext_dir=payload_dir,
                gpu_sm=86,
                cuda_version=128,
                model_dir=payload_dir / 'models',
            )
            with mock.patch.object(self.setup, 'SCRIPT_DIR', payload_dir):
                with self.assertRaises(self.setup.SetupError) as ctx:
                    self.setup.validate_internal_config(config)
            self.assertEqual(ctx.exception.code, 'missing-extension-payload')
            self.assertIn('dreamcube_mesh.py', str(ctx.exception))
        self.assertIn('dreamcube_manual_cubemap.py', str(ctx.exception))

    def test_manifest_declares_mesh_primary_scene_nodes_and_size_caps(self):
        manifest = json.loads((EXT_DIR / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['source'], 'https://github.com/DrHepa/modly-dreamcube-extension')
        manifest_nodes = {node['id']: node for node in manifest['nodes']}
        nodes = {node['id']: {item['id']: item for item in node['params_schema']} for node in manifest['nodes']}
        scene = nodes['generate-scene']
        self.assertEqual(manifest_nodes['generate-scene']['output'], 'mesh')
        self.assertEqual(manifest_nodes['generate-scene-manual-cubemap']['output'], 'mesh')
        self.assertNotIn('output_format', scene)
        self.assertEqual(manifest_nodes['generate-panorama']['output'], 'image')
        self.assertNotIn('mesh_depth_jump_threshold', nodes['generate-panorama'])
        self.assertEqual(scene['mesh_depth_jump_threshold']['default'], 0.20)
        self.assertEqual(scene['mesh_depth_jump_threshold']['min'], 0)
        self.assertEqual(scene['mesh_depth_jump_threshold']['max'], 5)
        self.assertEqual(scene['mesh_depth_jump_threshold']['step'], 0.05)
        self.assertEqual(scene['mesh_footprint_ratio_threshold']['default'], 12)
        self.assertTrue(scene['mesh_footprint_ratio_threshold']['advanced'])
        self.assertEqual(scene['mesh_aspect_ratio_threshold']['default'], 10)
        self.assertTrue(scene['mesh_aspect_ratio_threshold']['advanced'])
        for node_id, params in nodes.items():
            self.assertEqual(params['max_cube_size']['max'], 512)
            if node_id != 'generate-scene-manual-cubemap':
                self.assertEqual(params['max_equi_size']['max'], 2048)
            else:
                self.assertEqual(params['max_cube_size']['min'], 256)
                self.assertNotIn('max_equi_size', params)
                self.assertNotIn('depth_mode', params)

    def test_setup_does_not_add_mesh_as_dependency_or_patch_upstream(self):
        source = SETUP_PATH.read_text(encoding='utf-8')
        self.assertIn('dreamcube_mesh.py', source)
        self.assertFalse(any('dreamcube_mesh' in package for package in self.setup.UPSTREAM_DEPENDENCIES))
        self.assertNotIn('patch_upstream', source)
