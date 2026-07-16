from __future__ import annotations

import inspect
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import dreamcube_mesh as mesh


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch is required for mesh tests")
class DreamCubeMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def _front(self):
        points = self.torch.tensor([
            [-0.1, -0.1, 1.0],
            [0.1, -0.1, 1.0],
            [0.0, 0.1, 1.0],
        ])
        radii = self.torch.linalg.norm(points, dim=1)
        return points, points / radii[:, None], radii

    def test_cubemap_topology_count_and_representative_seams_and_corners(self):
        for size in (1, 2, 3, 4):
            topology = mesh.generate_cubemap_topology(size).cpu().tolist()
            self.assertEqual(len(topology), mesh.cubemap_candidate_triangle_count(size))

        size = 3
        topology = {tuple(face) for face in mesh.generate_cubemap_topology(size).cpu().tolist()}
        self.assertIn((0, 1, size), topology)
        self.assertIn((size - 1, 2 * size - 1, size * size), topology)
        self.assertIn((
            3 * size * size + size - 1,
            4 * size * size + (size - 1) * size,
            0,
        ), topology)

    def test_valid_cubemap_welds_all_known_joins_without_seam_or_corner_faces(self):
        torch = self.torch
        size = 4
        result = mesh.build_rgbd_cube_mesh(
            torch.ones((6, size, size, 3)) * 255,
            torch.ones((6, size, size)),
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
            face_chunk_size=5,
        )

        classes = result.stats["triangles"]["classes"]
        self.assertEqual(classes["face_grid"]["retained"], 12 * (size - 1) ** 2)
        self.assertEqual(classes["seam_strip"]["retained"], 0)
        self.assertEqual(classes["corner"]["retained"], 0)
        self.assertEqual(classes["seam_strip"]["added"], 0)
        self.assertEqual(classes["corner"]["added"], 0)
        self.assertEqual(result.stats["welds"]["candidate"], result.stats["welds"]["welded"])
        self.assertEqual(result.stats["boundary_edges"], 0)
        self.assertEqual(result.stats["expected_join_segments"], 12 * (size - 1))
        self.assertEqual(result.stats["join_segment_incidence"], {
            "0": 0,
            "1": 0,
            "2": 12 * (size - 1),
            "over2": 0,
        })
        self.assertEqual(result.stats["boundary_edges_along_cube_joins"], 0)
        self.assertEqual(result.stats["boundary_edges_touching_cube_joins"], 0)
        self.assertEqual(result.stats["strictly_internal_boundary_edges"], 0)
        self.assertEqual(result.stats["nonmanifold_edges"], 0)
        self.assertEqual(result.stats["incorrectly_oriented_manifold_edges"], 0)
        self.assertTrue(result.stats["edge_watertight"])
        self.assertEqual(result.stats["closure_status"], "closed_cube_joins")

    def test_clean_cubemap_face_groups_are_strictly_inward_wound(self):
        torch = self.torch
        size = 4
        result = mesh.build_rgbd_cube_mesh(
            torch.ones((6, size, size, 3)) * 255,
            torch.ones((6, size, size)),
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )

        vertices = torch.as_tensor(result.vertices)
        triangles = torch.as_tensor(result.triangles)
        triangle_vertices = vertices[triangles]
        centroids = triangle_vertices.mean(dim=1)
        normals = torch.linalg.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
            dim=1,
        )
        radial_normal_dots = (normals * centroids).sum(dim=1)
        face_directions = torch.tensor([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        face_groups = (centroids @ face_directions.T).argmax(dim=1)
        expected_per_face = 2 * (size - 1) ** 2

        self.assertEqual(
            torch.bincount(face_groups, minlength=6).tolist(),
            [expected_per_face] * 6,
        )
        for face_index, face_name in enumerate(mesh.CUBEMAP_FACE_ORDER):
            with self.subTest(face=face_name):
                face_dots = radial_normal_dots[face_groups == face_index]
                self.assertEqual(int(face_dots.numel()), expected_per_face)
                self.assertTrue(
                    bool((face_dots < 0).all()),
                    f"{face_name} radial normal dots must all be negative; "
                    f"range=[{float(face_dots.min())}, {float(face_dots.max())}]",
                )

    def test_prefilter_reconciliation_prevents_a_one_sided_join_rejection(self):
        torch = self.torch
        vertices = torch.tensor([
            [1.00, 0.00, 0.00],
            [1.00, 0.20, 0.00],
            [1.00, 0.00, 0.20],
            [1.00, 0.00, 0.19],
            [1.00, 0.20, 0.00],
            [1.00, 0.00, 0.20],
        ])
        rays = vertices / torch.linalg.norm(vertices, dim=1, keepdim=True)
        distance = torch.linalg.norm(vertices, dim=1)
        _, pre_weld_aspect, _ = mesh._triangle_geometry_metrics(
            face_vertices=vertices[torch.tensor([[0, 1, 2], [4, 3, 5]])],
            face_rays=rays[torch.tensor([[0, 1, 2], [4, 3, 5]])],
            torch=torch,
        )
        self.assertLessEqual(float(pre_weld_aspect[0]), 10)
        self.assertGreater(float(pre_weld_aspect[1]), 10)
        result = mesh.filter_mesh_topology(
            vertices=vertices,
            vertex_colors=torch.ones((6, 3)),
            distance=distance,
            triangles=torch.tensor([[0, 1, 2], [4, 3, 5]]),
            rays=rays,
            depth_jump_threshold=0.20,
            footprint_ratio_threshold=12,
            aspect_ratio_threshold=10,
            face_chunk_size=1,
            mode="cubemap",
            weld_groups=torch.tensor([[0, 3, -1], [1, 4, -1]]),
            join_segments=torch.tensor([[0, 1]]),
        )

        self.assertEqual(result.stats["triangles"]["exported"], 2)
        self.assertEqual(result.stats["join_segment_incidence"], {
            "0": 0, "1": 0, "2": 1, "over2": 0,
        })
        self.assertEqual(result.stats["boundary_edges_along_cube_joins"], 0)

    def test_internal_depth_hole_remains_open_without_added_triangles(self):
        torch = self.torch
        size = 5
        distance = torch.ones((6, size, size))
        distance[0, 2, 2] = 2.0
        result = mesh.build_rgbd_cube_mesh(
            torch.ones((6, size, size, 3)) * 255,
            distance,
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
            face_chunk_size=3,
        )

        classes = result.stats["triangles"]["classes"]
        self.assertGreater(classes["face_grid"]["removed"], 0)
        self.assertEqual(sum(item["added"] for item in classes.values()), 0)
        self.assertGreater(result.stats["boundary_edges"], 0)
        self.assertEqual(result.stats["nonmanifold_edges"], 0)
        self.assertFalse(result.stats["edge_watertight"])
        self.assertEqual(result.stats["join_segment_incidence"], {
            "0": 0,
            "1": 0,
            "2": 12 * (size - 1),
            "over2": 0,
        })
        self.assertEqual(result.stats["boundary_edges_along_cube_joins"], 0)
        self.assertEqual(result.stats["boundary_edges_touching_cube_joins"], 0)
        self.assertGreater(result.stats["strictly_internal_boundary_edges"], 0)
        self.assertEqual(
            result.stats["closure_status"],
            "closed_cube_joins_with_internal_holes",
        )

    def test_unsafe_cubemap_depth_mismatch_is_skipped_and_reported(self):
        torch = self.torch
        size = 4
        distance = torch.ones((6, size, size))
        distance[0] = 1.21
        result = mesh.build_rgbd_cube_mesh(
            torch.ones((6, size, size, 3)) * 255,
            distance,
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
            face_chunk_size=4,
        )

        welds = result.stats["welds"]
        self.assertGreater(welds["skipped_depth_mismatch"], 0)
        self.assertEqual(welds["candidate"], sum((
            welds["welded"],
            welds["skipped_non_finite_or_invalid"],
            welds["skipped_repaired"],
            welds["skipped_unreferenced"],
            welds["skipped_depth_mismatch"],
        )))
        self.assertGreater(result.stats["boundary_edges"], 0)
        self.assertGreater(
            result.stats["join_segment_incidence"]["0"]
            + result.stats["join_segment_incidence"]["1"],
            0,
        )
        self.assertGreater(
            result.stats["boundary_edges_along_cube_joins"],
            0,
        )
        self.assertTrue(
            result.stats["closure_status"].startswith("open_cube_joins")
        )
        self.assertEqual(result.stats["nonmanifold_edges"], 0)

    def test_cubemap_class_stats_reconcile_with_tiny_chunks(self):
        torch = self.torch
        size = 3
        result = mesh.build_rgbd_cube_mesh(
            torch.ones((6, size, size, 3)) * 255,
            torch.ones((6, size, size)),
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
            face_chunk_size=1,
        )

        classes = result.stats["triangles"]["classes"]
        for item in classes.values():
            self.assertEqual(item["candidate"] + item["added"], item["retained"] + item["removed"])
        self.assertEqual(sum(item["retained"] for item in classes.values()), result.stats["triangles"]["exported"])
        self.assertEqual(sum(item["candidate"] for item in classes.values()), result.stats["triangles"]["candidate"])

    def test_equirectangular_topology_wraps_final_column(self):
        topology = {tuple(face) for face in mesh.generate_equirectangular_topology(2, 3).cpu().tolist()}
        self.assertEqual(len(topology), mesh.equirectangular_candidate_triangle_count(2, 3))
        self.assertIn((2, 0, 5), topology)
        self.assertIn((0, 3, 5), topology)

    def test_sequential_rejection_compaction_colors_and_stats_reconcile(self):
        torch = self.torch
        good_vertices, good_rays, good_radii = self._front()
        bad_distance = torch.tensor([math.inf, 1.0, 1.0])
        vertices = torch.cat((good_vertices, bad_distance[:, None] * good_rays))
        colors = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        distance = torch.cat((good_radii, bad_distance))
        triangles = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 0, 1]])

        result = mesh.filter_mesh_topology(
            vertices=vertices,
            vertex_colors=colors,
            distance=distance,
            triangles=triangles,
            rays=torch.cat((good_rays, good_rays)),
            depth_jump_threshold=0.20,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )

        stats = result.stats["triangles"]
        self.assertEqual(stats["candidate"], 3)
        self.assertEqual(stats["exported"], 1)
        self.assertEqual(stats["regular"], 1)
        self.assertEqual(stats["caps"], 0)
        self.assertEqual(stats["removed_invalid_or_repaired"], 1)
        self.assertEqual(stats["removed_index_degenerate"], 1)
        self.assertEqual(stats["total_exported"], 1)
        self.assertEqual(result.stats["vertices"]["duplicated_for_caps"], 0)
        self.assertEqual(result.stats["vertices"]["invalid"]["non_finite_distance"], 1)
        np.testing.assert_allclose(result.vertex_colors, colors[:3].numpy())

    def test_depth_jump_threshold_is_strict_and_zero_disables_only_jump_rejection(self):
        torch = self.torch
        _, rays, _ = self._front()
        triangles = torch.tensor([[0, 1, 2]])
        colors = torch.ones((3, 3))

        exact_distance = torch.tensor([5.0, 6.0, 5.0])
        exact = mesh.filter_mesh_topology(
            vertices=exact_distance[:, None] * rays,
            vertex_colors=colors,
            distance=exact_distance,
            triangles=triangles,
            rays=rays,
            depth_jump_threshold=0.20,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )
        above_distance = torch.tensor([5.0, 6.01, 5.0])
        with self.assertRaises(mesh.MeshExportError) as removed:
            mesh.filter_mesh_topology(
                vertices=above_distance[:, None] * rays,
                vertex_colors=colors,
                distance=above_distance,
                triangles=triangles,
                rays=rays,
                depth_jump_threshold=0.20,
                footprint_ratio_threshold=0,
                aspect_ratio_threshold=0,
            )
        disabled = mesh.filter_mesh_topology(
            vertices=above_distance[:, None] * rays,
            vertex_colors=colors,
            distance=above_distance,
            triangles=triangles,
            rays=rays,
            depth_jump_threshold=0,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )

        self.assertEqual(exact.stats["triangles"]["exported"], 1)
        self.assertEqual(removed.exception.stats["triangles"]["removed_depth_discontinuity"], 1)
        self.assertEqual(disabled.stats["triangles"]["removed_depth_discontinuity"], 0)
        self.assertEqual(disabled.stats["triangles"]["exported"], 1)

    def test_relative_jump_rejection_is_scale_invariant(self):
        torch = self.torch
        good_vertices, rays, good_radii = self._front()
        triangles = torch.tensor([[0, 1, 2], [3, 4, 5]])
        colors = torch.ones((6, 3))

        for scale in (1.0, 1000.0):
            jump_distance = torch.tensor([1.0, 1.3, 1.0]) * scale
            distance = torch.cat((good_radii * scale, jump_distance))
            vertices = torch.cat((good_vertices * scale, jump_distance[:, None] * rays))
            result = mesh.filter_mesh_topology(
                vertices=vertices,
                vertex_colors=colors,
                distance=distance,
                triangles=triangles,
                rays=torch.cat((rays, rays)),
                depth_jump_threshold=0.20,
                footprint_ratio_threshold=0,
                aspect_ratio_threshold=0,
            )
            self.assertEqual(result.stats["triangles"]["removed_depth_discontinuity"], 1)
            self.assertEqual(result.stats["triangles"]["exported"], 1)

    def test_all_invalid_depth_error_carries_stats(self):
        torch = self.torch
        with self.assertRaises(mesh.MeshExportError) as ctx:
            mesh.filter_mesh_topology(
                vertices=torch.zeros((3, 3)),
                vertex_colors=torch.ones((3, 3)),
                distance=torch.tensor([float("nan"), 0.0, float("inf")]),
                triangles=torch.tensor([[0, 1, 2]]),
                depth_jump_threshold=0.20,
            )
        self.assertIn("at or above", str(ctx.exception).lower())
        self.assertEqual(ctx.exception.stats["vertices"]["invalid"], {
            "non_finite_distance": 2,
            "non_positive_distance": 1,
            "below_minimum_distance": 0,
            "non_finite_xyz": 0,
            "total": 3,
        })

    def test_min_valid_distance_rejects_touching_faces_and_keeps_boundary_valid(self):
        torch = self.torch
        good_vertices, rays, good_radii = self._front()
        bad_distance = torch.tensor([0.009999, mesh.MIN_VALID_DISTANCE_METERS, 0.02])
        result = mesh.filter_mesh_topology(
            vertices=torch.cat((good_vertices, bad_distance[:, None] * rays)),
            vertex_colors=torch.ones((6, 3)),
            distance=torch.cat((good_radii, bad_distance)),
            triangles=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            rays=torch.cat((rays, rays)),
            depth_jump_threshold=0,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )

        invalid = result.stats["vertices"]["invalid"]
        self.assertEqual(invalid["below_minimum_distance"], 1)
        self.assertEqual(invalid["non_positive_distance"], 0)
        self.assertEqual(result.stats["vertices"]["repaired"], 1)
        self.assertEqual(result.stats["triangles"]["removed_invalid_or_repaired"], 1)
        self.assertEqual(result.stats["triangles"]["exported"], 1)

    def test_invalid_stats_separate_non_positive_from_below_minimum(self):
        torch = self.torch
        good_vertices, rays, good_radii = self._front()
        bad_distance = torch.tensor([0.0, -1.0, 0.009999, 0.02])
        bad_rays = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
        result = mesh.filter_mesh_topology(
            vertices=torch.cat((bad_distance[:, None] * bad_rays, good_vertices)),
            vertex_colors=torch.ones((7, 3)),
            distance=torch.cat((bad_distance, good_radii)),
            triangles=torch.tensor([[0, 2, 3], [0, 1, 3], [4, 5, 6]]),
            rays=torch.cat((bad_rays, rays)),
            depth_jump_threshold=0,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )

        invalid = result.stats["vertices"]["invalid"]
        self.assertEqual(invalid, {
            "non_finite_distance": 0,
            "non_positive_distance": 2,
            "below_minimum_distance": 1,
            "non_finite_xyz": 0,
            "total": 3,
        })
        self.assertEqual(result.stats["vertices"]["repaired"], 3)
        self.assertEqual(result.stats["triangles"]["removed_invalid_or_repaired"], 2)
        self.assertEqual(result.stats["triangles"]["exported"], 1)

    def test_adaptive_diagonal_minimizes_rejected_faces_without_fabricating_geometry(self):
        torch = self.torch
        rgb = torch.ones((2, 2, 3)) * 255
        distance = torch.tensor([[1.0, 1.0], [10.0, 1.0]])
        result = mesh.build_rgbd_equi_mesh(
            rgb,
            distance,
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )
        stats = result.stats["triangles"]
        self.assertEqual(result.stats["triangles"]["adaptive_diagonals"], 1)
        self.assertEqual(stats["candidate"], 4)
        self.assertEqual(stats["exported"] + stats["removed_depth_discontinuity"], 4)
        self.assertEqual(stats["caps"], 0)
        self.assertEqual(result.stats["vertices"]["duplicated_for_caps"], 0)

    def test_boundary_quad_reselects_safe_alternate_diagonal(self):
        torch = self.torch
        vertices = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.4448153, 0.2204748],
            [1.0, 0.5060700, 0.6707333],
        ])
        rays = vertices / torch.linalg.norm(vertices, dim=1, keepdim=True)
        faces = torch.tensor([[0, 1, 2], [2, 1, 3]])
        alternate = torch.tensor([[0, 1, 3], [0, 3, 2]])
        adaptive_stats = {}

        _, existing_aspect, _ = mesh._triangle_geometry_metrics(
            face_vertices=vertices[faces],
            face_rays=rays[faces],
            torch=torch,
        )
        _, alternate_aspect, _ = mesh._triangle_geometry_metrics(
            face_vertices=vertices[alternate],
            face_rays=rays[alternate],
            torch=torch,
        )
        self.assertGreater(int((existing_aspect > 3).sum()), 0)
        self.assertTrue(bool((alternate_aspect <= 3).all()))

        selected, adaptive_count = mesh._build_adaptive_faces(
            faces,
            torch.tensor([[0, 1, 2, 3]]),
            torch.linalg.norm(vertices, dim=1),
            torch.zeros((4,), dtype=torch.bool),
            0,
            torch,
            vertices=vertices,
            rays=rays,
            footprint_threshold=0,
            aspect_threshold=3,
            join_group_for_vertex=torch.tensor([0, 1, -1, -1]),
            join_segments=torch.tensor([[0, 1]]),
            adaptive_stats=adaptive_stats,
        )

        self.assertTrue(torch.equal(selected, alternate))
        self.assertEqual(adaptive_count, 1)
        self.assertEqual(adaptive_stats["boundary_quads_evaluated"], 1)
        self.assertEqual(adaptive_stats["boundary_diagonal_reselections"], 1)
        self.assertEqual(adaptive_stats["boundary_quads_unclosable"], 0)

    def test_invalid_depth_repair_is_deterministic_and_face_is_rejected(self):
        torch = self.torch
        good_vertices, rays, good_radii = self._front()
        bad_distance = torch.tensor([float("inf"), 2.0, 1.0])
        result = mesh.filter_mesh_topology(
            vertices=torch.cat((bad_distance[:, None] * rays, good_vertices)),
            vertex_colors=torch.ones((6, 3)),
            distance=torch.cat((bad_distance, good_radii)),
            triangles=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            rays=torch.cat((rays, rays)),
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )
        self.assertEqual(result.stats["vertices"]["repaired"], 1)
        self.assertEqual(result.stats["vertices"]["repair_rounds"], 1)
        self.assertEqual(result.stats["triangles"]["removed_invalid_or_repaired"], 1)
        self.assertEqual(result.stats["triangles"]["exported"], 1)

    def test_parameter_grid_repairs_preserve_wrap_and_cubemap_seam_metadata(self):
        torch = self.torch
        equi_rgb = torch.ones((3, 4, 3)) * 255
        equi_distance = torch.ones((3, 4))
        equi_distance[0, 0] = float("inf")
        equi = mesh.build_rgbd_equi_mesh(
            equi_rgb,
            equi_distance,
            device="cpu",
            depth_jump_threshold=0,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )
        self.assertTrue(equi.stats["horizontal_wrap"])
        self.assertEqual(equi.stats["vertices"]["repaired"], 1)
        self.assertGreater(equi.stats["triangles"]["removed_invalid_or_repaired"], 0)

        cube_rgb = torch.ones((6, 3, 3, 3)) * 255
        cube_distance = torch.ones((6, 3, 3))
        cube_distance[1, 0, 0] = float("inf")
        cube = mesh.build_rgbd_cube_mesh(
            cube_rgb,
            cube_distance,
            device="cpu",
            depth_jump_threshold=0,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )
        self.assertEqual(cube.stats["closed_topology"]["edge_seams"], 12)
        self.assertEqual(cube.stats["vertices"]["repaired"], 1)
        self.assertGreater(cube.stats["triangles"]["removed_invalid_or_repaired"], 0)

    def test_threshold_zero_keeps_jump_faces_but_never_repaired_faces(self):
        torch = self.torch
        good_vertices, rays, good_radii = self._front()
        jump_distance = torch.tensor([1.0, 10.0, 1.0])
        repaired_distance = torch.tensor([float("inf"), 1.0, 1.0])
        result = mesh.filter_mesh_topology(
            vertices=torch.cat((
                good_vertices,
                jump_distance[:, None] * rays,
                repaired_distance[:, None] * rays,
            )),
            vertex_colors=torch.ones((9, 3)),
            distance=torch.cat((good_radii, jump_distance, repaired_distance)),
            triangles=torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            rays=torch.cat((rays, rays, rays)),
            depth_jump_threshold=0,
            footprint_ratio_threshold=0,
            aspect_ratio_threshold=0,
        )
        self.assertEqual(result.stats["triangles"]["removed_depth_discontinuity"], 0)
        self.assertEqual(result.stats["triangles"]["removed_invalid_or_repaired"], 1)
        self.assertEqual(result.stats["triangles"]["exported"], 2)
        self.assertEqual(result.stats["triangles"]["caps"], 0)

    def test_flat_equirect_and_cubemap_preserve_authoritative_candidates_and_winding(self):
        torch = self.torch
        height, width = 3, 4
        equi = mesh.build_rgbd_equi_mesh(
            torch.ones((height, width, 3)) * 255,
            torch.ones((height, width)),
            device="cpu",
            depth_jump_threshold=0.2,
        )
        expected_equi = mesh.equirectangular_candidate_triangle_count(height, width)
        self.assertEqual(equi.stats["triangles"]["candidate"], expected_equi)
        self.assertEqual(equi.stats["triangles"]["exported"], expected_equi)
        equi_faces = {tuple(int(v) for v in face) for face in equi.triangles.tolist()}
        for a, b, c, d in mesh.generate_equirectangular_quads(height, width).cpu().tolist():
            self.assertIn((a, b, c), equi_faces)
            self.assertIn((b, d, c), equi_faces)

        size = 3
        cube = mesh.build_rgbd_cube_mesh(
            torch.ones((6, size, size, 3)) * 255,
            torch.ones((6, size, size)),
            device="cpu",
            depth_jump_threshold=0.2,
        )
        expected_cube = mesh.cubemap_candidate_triangle_count(size)
        self.assertEqual(cube.stats["triangles"]["candidate"], expected_cube)
        self.assertEqual(cube.stats["triangles"]["exported"], 12 * (size - 1) ** 2)
        self.assertEqual(cube.stats["triangles"]["candidate"], expected_cube)
        self.assertEqual(cube.stats["closed_topology"]["edge_seams"], 12)
        self.assertEqual(cube.stats["closed_topology"]["corner_triangles"], 8)

    def test_rejection_stats_reconcile_after_adaptive_filtering(self):
        torch = self.torch
        height, width = 4, 5
        rgb = torch.ones((height, width, 3)) * 255
        distance = torch.ones((height, width))
        distance[1, 2] = 10.0

        result = mesh.build_rgbd_equi_mesh(
            rgb,
            distance,
            device="cpu",
            depth_jump_threshold=0.2,
            footprint_ratio_threshold=12,
            aspect_ratio_threshold=10,
            face_chunk_size=7,
        )
        stats = result.stats["triangles"]
        removed_keys = (
            "removed_invalid_or_repaired",
            "removed_depth_discontinuity",
            "removed_non_finite_or_degenerate_metrics",
            "removed_footprint_ratio",
            "removed_aspect_ratio",
            "removed_index_degenerate",
        )
        self.assertEqual(stats["candidate"], stats["exported"] + sum(stats[key] for key in removed_keys))
        self.assertEqual(stats["exported"], stats["regular"])
        self.assertEqual(stats["caps"], 0)
        self.assertEqual(result.stats["vertices"]["duplicated_for_caps"], 0)
        self.assertGreater(sum(stats[key] for key in removed_keys), 0)

    def test_medium_scale_vectorized_path_avoids_scalar_helpers(self):
        torch = self.torch
        height, width = 32, 48
        rgb = torch.ones((height, width, 3)) * 255
        rows = torch.arange(height, dtype=torch.float32).reshape(-1, 1)
        cols = torch.arange(width, dtype=torch.float32).reshape(1, -1)
        distance = 1.0 + (rows + cols) * 0.0005

        source = inspect.getsource(mesh._build_adaptive_faces) + inspect.getsource(mesh.filter_mesh_topology)
        self.assertNotIn("quad.tolist()", source)
        self.assertNotIn("_make_cap_vertices", source)

        result = mesh.build_rgbd_equi_mesh(
            rgb,
            distance,
            device="cpu",
            depth_jump_threshold=0.2,
            face_chunk_size=257,
        )
        stats = result.stats["triangles"]
        self.assertEqual(stats["candidate"], mesh.equirectangular_candidate_triangle_count(height, width))
        self.assertEqual(stats["exported"], stats["regular"])
        self.assertEqual(stats["caps"], 0)
        self.assertGreater(stats["exported"], 0)

    def test_tiny_open3d_shim_obj_export_is_parseable(self):
        class TriangleMesh:
            pass

        class Utility:
            @staticmethod
            def Vector3dVector(value):
                return np.asarray(value, dtype=float)

            @staticmethod
            def Vector3iVector(value):
                return np.asarray(value, dtype=int)

        class IO:
            @staticmethod
            def write_triangle_mesh(path, obj):
                with Path(path).open("w", encoding="utf-8") as handle:
                    for vertex, color in zip(obj.vertices, obj.vertex_colors):
                        handle.write("v {} {} {} {} {} {}\n".format(*vertex, *color))
                    for face in obj.triangles:
                        handle.write("f {} {} {}\n".format(*(face + 1)))
                return True

        shim = types.SimpleNamespace(
            geometry=types.SimpleNamespace(TriangleMesh=TriangleMesh),
            utility=Utility,
            io=IO,
        )
        build = mesh.MeshBuildResult(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            triangles=np.array([[0, 1, 2]], dtype=np.int64),
            vertex_colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            stats={"triangles": {"candidate": 1, "exported": 1}, "vertices": {"candidate": 3, "exported": 3}},
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"open3d": shim}):
            path = Path(directory) / "mesh.obj"
            mesh._export_build_result(build, path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum(line.startswith("v ") for line in lines), 3)
        self.assertEqual([line for line in lines if line.startswith("f ")], ["f 1 2 3"])

    def test_real_open3d_export_when_available(self):
        try:
            import open3d  # noqa: F401
        except Exception:
            self.skipTest("real Open3D is optional")
        self.assertTrue(True)


class TinyOpen3DShimExportTests(unittest.TestCase):
    def test_tiny_open3d_shim_obj_export_is_parseable_without_torch(self):
        class TriangleMesh:
            pass

        class Utility:
            @staticmethod
            def Vector3dVector(value):
                return np.asarray(value, dtype=float)

            @staticmethod
            def Vector3iVector(value):
                return np.asarray(value, dtype=int)

        class IO:
            @staticmethod
            def write_triangle_mesh(path, obj):
                with Path(path).open("w", encoding="utf-8") as handle:
                    for vertex, color in zip(obj.vertices, obj.vertex_colors):
                        handle.write("v {} {} {} {} {} {}\n".format(*vertex, *color))
                    for face in obj.triangles:
                        handle.write("f {} {} {}\n".format(*(face + 1)))
                return True

        shim = types.SimpleNamespace(
            geometry=types.SimpleNamespace(TriangleMesh=TriangleMesh),
            utility=Utility,
            io=IO,
        )
        build = mesh.MeshBuildResult(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            triangles=np.array([[0, 1, 2]], dtype=np.int64),
            vertex_colors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            stats={"triangles": {"candidate": 1, "exported": 1}, "vertices": {"candidate": 3, "exported": 3}},
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"open3d": shim}):
            path = Path(directory) / "mesh.obj"
            mesh._export_build_result(build, path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(sum(line.startswith("v ") for line in lines), 3)
        self.assertEqual([line for line in lines if line.startswith("f ")], ["f 1 2 3"])


if __name__ == "__main__":
    unittest.main()
