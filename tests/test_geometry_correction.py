from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dreamcube_mesh as mesh


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required")
class GeometryCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def front(self):
        points = self.torch.tensor([
            [-0.1, -0.1, 1.0],
            [0.1, -0.1, 1.0],
            [0.0, 0.1, 1.0],
        ])
        radii = self.torch.linalg.norm(points, dim=1)
        return points, points / radii[:, None], radii

    def two_faces(self, bad_rays, bad_radii, **kwargs):
        good_vertices, good_rays, good_radii = self.front()
        return mesh.filter_mesh_topology(
            vertices=self.torch.cat((good_vertices, bad_rays * bad_radii[:, None])),
            vertex_colors=self.torch.ones((6, 3)),
            distance=self.torch.cat((good_radii, bad_radii)),
            triangles=self.torch.tensor([[0, 1, 2], [3, 4, 5]]),
            rays=self.torch.cat((good_rays, bad_rays)),
            depth_jump_threshold=0,
            **kwargs,
        )

    def test_repaired_and_jump_faces_are_removed_without_private_vertices(self):
        _, good_rays, good_radii = self.front()
        rays = self.torch.cat((good_rays, good_rays, good_rays))
        distances = self.torch.cat((
            self.torch.tensor([float("inf"), 1.0, 1.0]),
            self.torch.tensor([1.0, 2.0, 1.0]),
            good_radii,
        ))
        result = mesh.filter_mesh_topology(
            vertices=distances[:, None] * rays,
            vertex_colors=self.torch.ones((9, 3)),
            distance=distances,
            triangles=self.torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            rays=rays,
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
            face_chunk_size=1,
        )
        triangles = result.stats["triangles"]
        self.assertEqual(triangles["removed_invalid_or_repaired"], 1)
        self.assertEqual(triangles["removed_depth_discontinuity"], 1)
        self.assertEqual(triangles["exported"], 1)
        self.assertEqual(triangles["caps"], 0)
        self.assertEqual(result.stats["vertices"]["duplicated_for_caps"], 0)
        self.assertFalse(hasattr(mesh, "_make_cap_vertices"))
        self.assertLessEqual(len(result.vertices), 9)

    def test_footprint_filter_removes_needle_and_zero_disables_it(self):
        _, rays, _ = self.front()
        radii = self.torch.tensor([1.0, 20.0, 1.0])
        filtered = self.two_faces(rays, radii, footprint_ratio_threshold=12, aspect_ratio_threshold=0)
        disabled = self.two_faces(rays, radii, footprint_ratio_threshold=0, aspect_ratio_threshold=0)
        self.assertEqual(filtered.stats["triangles"]["removed_footprint_ratio"], 1)
        self.assertEqual(filtered.stats["triangles"]["exported"], 1)
        self.assertEqual(disabled.stats["triangles"]["exported"], 2)

    def test_aspect_filter_removes_sliver_and_zero_disables_it(self):
        angles = self.torch.tensor([0.0, 0.01, 0.0201])
        rays = self.torch.stack((
            self.torch.sin(angles),
            self.torch.zeros_like(angles),
            self.torch.cos(angles),
        ), dim=1)
        radii = self.torch.ones(3)
        filtered = self.two_faces(rays, radii, footprint_ratio_threshold=0, aspect_ratio_threshold=10)
        disabled = self.two_faces(rays, radii, footprint_ratio_threshold=0, aspect_ratio_threshold=0)
        self.assertEqual(filtered.stats["triangles"]["removed_aspect_ratio"], 1)
        self.assertEqual(filtered.stats["triangles"]["exported"], 1)
        self.assertEqual(disabled.stats["triangles"]["exported"], 2)

    def test_benign_front_geometry_is_retained_by_defaults(self):
        vertices, rays, radii = self.front()
        result = mesh.filter_mesh_topology(
            vertices=vertices,
            vertex_colors=self.torch.ones((3, 3)),
            distance=radii,
            triangles=self.torch.tensor([[0, 1, 2]]),
            rays=rays,
        )
        self.assertEqual(result.stats["triangles"]["exported"], 1)
        self.assertEqual(result.stats["triangles"]["retention"], 1.0)


class SceneManifestCorrectionTests(unittest.TestCase):
    def test_generated_manifest_omits_collision_zones(self):
        import generator
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            run_dir = workspace / "Default" / "run"
            run_dir.mkdir(parents=True)
            glb = run_dir / "output_mesh.glb"
            glb.write_bytes(b"glb")
            instance = generator.DreamCubeGenerator(root / "models", workspace / "Default")
            with mock.patch.dict(os.environ, {"WORKSPACE_DIR": str(workspace)}):
                manifest_path = instance._write_scene_manifest(run_dir, glb)
            manifest = json.loads(manifest_path.read_text())
        self.assertNotIn("collisionZones", manifest)
        self.assertEqual(manifest["initialView"]["position"], [0, 0, 0])
        self.assertEqual(manifest["assets"][0]["kind"], "glb")


if __name__ == "__main__":
    unittest.main()
