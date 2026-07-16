"""Extension-owned RGB-D mesh reconstruction for DreamCube.

The upstream project remains authoritative for 3D Gaussian splat export.  OBJ
mesh topology, filtering, compaction, and Open3D writing live here so the
extension does not depend on Open3D cleanup methods or upstream mesh helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_DEPTH_JUMP_THRESHOLD = 0.20
DEFAULT_FOOTPRINT_RATIO_THRESHOLD = 12.0
DEFAULT_ASPECT_RATIO_THRESHOLD = 10.0
DEFAULT_FACE_CHUNK_SIZE = 262_144
METRIC_EPSILON = 1e-8
# DreamCube distances are meters; interior RGB-D scene samples below 1 cm are physically invalid.
MIN_VALID_DISTANCE_METERS = 0.01
CUBEMAP_FACE_ORDER = ("+Z", "-X", "-Z", "+X", "+Y", "-Y")


class MeshExportError(RuntimeError):
    """Mesh export failure with any statistics collected before the failure."""

    def __init__(self, message: str, *, stats: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stats = dict(stats) if stats is not None else None


@dataclass(frozen=True)
class MeshBuildResult:
    """Compacted CPU arrays and structured filtering statistics."""

    vertices: Any
    triangles: Any
    vertex_colors: Any
    stats: dict[str, Any]


@dataclass(frozen=True)
class MeshExportResult:
    """Open3D-compatible mesh and structured filtering statistics."""

    mesh: Any
    stats: dict[str, Any]


@dataclass(frozen=True)
class CubemapTopologyPlan:
    """Candidate topology plus the known duplicate cubemap border groups."""

    triangles: Any
    triangle_classes: Any
    weld_groups: Any
    join_segments: Any | None = None


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime setup supplies torch
        raise RuntimeError("DreamCube mesh reconstruction requires torch.") from exc
    return torch


def cubemap_candidate_triangle_count(face_size: int) -> int:
    """Return the closed six-face cubemap triangle count."""

    size = int(face_size)
    if size < 1:
        raise ValueError("Cubemap face size must be at least 1.")
    return 12 * size * size - 4


def equirectangular_candidate_triangle_count(height: int, width: int) -> int:
    """Return the horizontal-wrap equirectangular triangle count."""

    h = int(height)
    w = int(width)
    if h < 1 or w < 1:
        raise ValueError("Equirectangular dimensions must be at least 1.")
    return 2 * max(h - 1, 0) * w


def generate_equirectangular_topology(height: int, width: int, *, device: Any = "cpu") -> Any:
    """Build a row grid whose last column wraps back to the first column."""

    torch = _require_torch()
    h = int(height)
    w = int(width)
    if h < 1 or w < 1:
        raise ValueError("Equirectangular dimensions must be at least 1.")

    rows = torch.arange(0, h - 1, device=device, dtype=torch.long).repeat(w)
    cols = torch.arange(0, w, device=device, dtype=torch.long).repeat_interleave(max(h - 1, 0))
    top_left = rows * w + cols
    top_right = rows * w + (cols + 1) % w
    bottom_left = top_left + w
    bottom_right = (rows + 1) * w + (cols + 1) % w

    first = torch.stack((top_left, top_right, bottom_left), dim=1)
    second = torch.stack((top_right, bottom_right, bottom_left), dim=1)
    topology = torch.cat((first, second), dim=0)
    expected = equirectangular_candidate_triangle_count(h, w)
    if int(topology.shape[0]) != expected:  # defensive invariant, no data transfer
        raise RuntimeError(f"Equirectangular topology mismatch: expected {expected}, got {topology.shape[0]}.")
    return topology


def generate_cubemap_topology_plan(face_size: int, *, device: Any = "cpu") -> CubemapTopologyPlan:
    """Build classified candidates and deterministic cubemap border weld groups."""

    torch = _require_torch()
    size = int(face_size)
    if size < 1:
        raise ValueError("Cubemap face size must be at least 1.")

    all_triangles: list[Any] = []
    triangle_classes: list[Any] = []
    grid_y, grid_x = torch.meshgrid(
        torch.arange(size - 1, device=device, dtype=torch.long),
        torch.arange(size - 1, device=device, dtype=torch.long),
        indexing="ij",
    )
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)
    v0 = grid_y * size + grid_x
    v1 = v0 + 1
    v2 = v0 + size
    v3 = v2 + 1
    face_triangles = torch.cat((torch.stack((v0, v1, v2), dim=1), torch.stack((v2, v1, v3), dim=1)), dim=0)
    vertices_per_face = size * size
    for face_index in range(6):
        all_triangles.append(face_triangles + face_index * vertices_per_face)
        triangle_classes.append(torch.zeros((int(face_triangles.shape[0]),), device=device, dtype=torch.uint8))

    def edge(face: int, name: str) -> Any:
        offset = face * vertices_per_face
        if name == "top":
            return torch.arange(size, device=device, dtype=torch.long) + offset
        if name == "bottom":
            return torch.arange(size, device=device, dtype=torch.long) + offset + (size - 1) * size
        if name == "left":
            return torch.arange(size, device=device, dtype=torch.long) * size + offset
        if name == "right":
            return torch.arange(size, device=device, dtype=torch.long) * size + offset + size - 1
        raise ValueError(f"Unknown cubemap edge {name!r}.")

    edge_pairs = (
        (0, "right", 1, "left", False), (1, "right", 2, "left", False),
        (2, "right", 3, "left", False), (3, "right", 0, "left", False),
        (0, "top", 4, "bottom", False), (1, "top", 4, "right", True),
        (2, "top", 4, "top", True), (3, "top", 4, "left", False),
        (0, "bottom", 5, "top", False), (1, "bottom", 5, "right", False),
        (2, "bottom", 5, "bottom", True), (3, "bottom", 5, "left", True),
    )
    paired_edges: list[tuple[Any, Any]] = []
    for face_a, edge_a_name, face_b, edge_b_name, reverse_b in edge_pairs:
        edge_a = edge(face_a, edge_a_name)
        edge_b = edge(face_b, edge_b_name)
        if reverse_b:
            edge_b = edge_b.flip(0)
        paired_edges.append((edge_a, edge_b))
        a0, a1 = edge_a[:-1], edge_a[1:]
        b0, b1 = edge_b[:-1], edge_b[1:]
        all_triangles.extend((torch.stack((a0, a1, b0), dim=1), torch.stack((b0, a1, b1), dim=1)))
        triangle_classes.extend((
            torch.ones((max(size - 1, 0),), device=device, dtype=torch.uint8),
            torch.ones((max(size - 1, 0),), device=device, dtype=torch.uint8),
        ))

    def vertex_id(face: int, row: int, col: int) -> int:
        return face * vertices_per_face + row * size + col

    corners = (
        (vertex_id(3, 0, size - 1), vertex_id(4, size - 1, 0), vertex_id(0, 0, 0)),
        (vertex_id(0, 0, size - 1), vertex_id(4, size - 1, size - 1), vertex_id(1, 0, 0)),
        (vertex_id(1, 0, size - 1), vertex_id(4, 0, size - 1), vertex_id(2, 0, 0)),
        (vertex_id(2, 0, size - 1), vertex_id(4, 0, 0), vertex_id(3, 0, 0)),
        (vertex_id(3, size - 1, 0), vertex_id(5, size - 1, 0), vertex_id(2, size - 1, size - 1)),
        (vertex_id(2, size - 1, 0), vertex_id(5, size - 1, size - 1), vertex_id(1, size - 1, size - 1)),
        (vertex_id(1, size - 1, 0), vertex_id(5, 0, size - 1), vertex_id(0, size - 1, size - 1)),
        (vertex_id(0, size - 1, 0), vertex_id(5, 0, 0), vertex_id(3, size - 1, size - 1)),
    )
    all_triangles.append(torch.tensor(corners, device=device, dtype=torch.long))
    triangle_classes.append(torch.full((8,), 2, device=device, dtype=torch.uint8))
    topology = torch.cat(all_triangles, dim=0)
    classes = torch.cat(triangle_classes, dim=0)
    expected = cubemap_candidate_triangle_count(size)
    if int(topology.shape[0]) != expected:
        raise RuntimeError(f"Cubemap topology mismatch: expected {expected}, got {topology.shape[0]}.")

    parent: dict[int, int] = {}
    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)
    for edge_a, edge_b in paired_edges:
        for first, second in zip(edge_a.detach().cpu().tolist(), edge_b.detach().cpu().tolist()):
            union(int(first), int(second))
    grouped: dict[int, list[int]] = {}
    for value in sorted(parent):
        grouped.setdefault(find(value), []).append(value)
    groups = [values for values in grouped.values() if 2 <= len(values) <= 3]
    padded = [values + [-1] * (3 - len(values)) for values in groups]
    weld_groups = torch.tensor(padded, device=device, dtype=torch.long) if padded else torch.empty((0, 3), device=device, dtype=torch.long)
    group_by_vertex = {
        vertex: group_index
        for group_index, values in enumerate(groups)
        for vertex in values
    }
    join_segment_groups = [
        (group_by_vertex[int(first)], group_by_vertex[int(second)])
        for edge_a, _ in paired_edges
        for first, second in zip(
            edge_a[:-1].detach().cpu().tolist(),
            edge_a[1:].detach().cpu().tolist(),
        )
    ]
    join_segments = (
        torch.tensor(join_segment_groups, device=device, dtype=torch.long)
        if join_segment_groups
        else torch.empty((0, 2), device=device, dtype=torch.long)
    )
    return CubemapTopologyPlan(topology, classes, weld_groups, join_segments)


def generate_cubemap_topology(face_size: int, *, device: Any = "cpu") -> Any:
    """Build six face grids, twelve edge seams, and eight corner triangles."""

    return generate_cubemap_topology_plan(face_size, device=device).triangles

def _validated_threshold(value: Any, *, label: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number.") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number.")
    return threshold


def _shape_list(shape: Any) -> list[int]:
    return [int(value) for value in shape]



def _relative_jump(depths: Any, torch: Any) -> Any:
    minimum_depth = depths.amin(dim=1)
    maximum_depth = depths.amax(dim=1)
    return (maximum_depth - minimum_depth) / torch.clamp(minimum_depth, min=1e-6)


def _repair_invalid_depths(*, distance: Any, vertices: Any, rays: Any, triangles: Any, torch: Any) -> tuple[Any, Any, Any, int]:
    valid = (
        torch.isfinite(distance)
        & (distance >= MIN_VALID_DISTANCE_METERS)
        & torch.isfinite(vertices).all(dim=1)
    )
    repaired = ~valid
    if not bool(valid.any().item()):
        raise MeshExportError(
            "DreamCube mesh export cannot repair depth: no finite depth samples at or above "
            f"{MIN_VALID_DISTANCE_METERS:g} meters exist.",
            stats={"vertices": {"candidate": int(distance.numel()), "invalid": {"total": int(repaired.sum().item())}}},
        )
    if not bool(repaired.any().item()):
        return distance, vertices, repaired, 0

    repaired_distance = distance.clone()
    frontier = valid.clone()
    unresolved = ~valid
    rounds = 0
    while bool(unresolved.any().item()):
        rounds += 1
        candidate_depth = torch.full_like(repaired_distance, float("inf"))
        # Deterministic by sorted triangle order; same-round ties keep the minimum propagated depth.
        for edge in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)):
            src = triangles[:, edge[0]]
            dst = triangles[:, edge[1]]
            mask = frontier[src] & unresolved[dst]
            if bool(mask.any().item()):
                dst_sel = dst[mask]
                src_depth = repaired_distance[src[mask]]
                candidate_depth.scatter_reduce_(0, dst_sel, src_depth, reduce="amin", include_self=True)
        newly_resolved = unresolved & torch.isfinite(candidate_depth)
        if not bool(newly_resolved.any().item()):
            raise MeshExportError(
                "DreamCube mesh export cannot repair depth: invalid samples are disconnected from valid depth "
                f"at or above {MIN_VALID_DISTANCE_METERS:g} meters.",
                stats={"vertices": {"candidate": int(distance.numel()), "invalid": {"total": int(repaired.sum().item())}}},
            )
        repaired_distance[newly_resolved] = candidate_depth[newly_resolved]
        unresolved = unresolved & ~newly_resolved
        frontier = newly_resolved

    repaired_vertices = repaired_distance.reshape(-1, 1) * rays
    return repaired_distance, repaired_vertices, repaired, rounds


def _face_keys(faces: Any, vertex_count: int) -> Any:
    return (faces[:, 0] * vertex_count + faces[:, 1]) * vertex_count + faces[:, 2]


def _keys_present(haystack: Any, needles: Any, torch: Any) -> Any:
    if int(needles.numel()) == 0:
        return torch.zeros_like(needles, dtype=torch.bool)
    if int(haystack.numel()) == 0:
        return torch.zeros_like(needles, dtype=torch.bool)
    sorted_haystack = torch.sort(haystack).values
    positions = torch.searchsorted(sorted_haystack, needles)
    positions = torch.clamp(positions, max=int(sorted_haystack.numel()) - 1)
    return sorted_haystack[positions] == needles


def _build_adaptive_faces(
    base_faces: Any,
    quads: Any,
    distance: Any,
    repaired: Any,
    threshold: float,
    torch: Any,
    *,
    vertices: Any | None = None,
    rays: Any | None = None,
    footprint_threshold: float = 0.0,
    aspect_threshold: float = 0.0,
    join_group_for_vertex: Any | None = None,
    join_segments: Any | None = None,
    safe_join_groups: Any | None = None,
    adaptive_stats: dict[str, Any] | None = None,
) -> tuple[Any, int]:
    if quads is None or int(quads.numel()) == 0:
        return base_faces, 0

    device = distance.device
    vertex_count = int(distance.numel())
    quads_tensor = quads.detach().to(device=device, dtype=torch.long).reshape(-1, 4)
    base_faces_tensor = base_faces.detach().to(device=device, dtype=torch.long).reshape(-1, 3)

    a = quads_tensor[:, 0]
    b = quads_tensor[:, 1]
    c = quads_tensor[:, 2]
    d = quads_tensor[:, 3]
    first = torch.stack((a, b, c), dim=1)
    second_cubemap = torch.stack((c, b, d), dim=1)
    second_equirect = torch.stack((b, d, c), dim=1)
    alt_first = torch.stack((a, b, d), dim=1)
    alt_second = torch.stack((a, d, c), dim=1)

    base_keys = _face_keys(base_faces_tensor, vertex_count)
    equirect_second_keys = _face_keys(second_equirect, vertex_count)
    use_equirect_winding = _keys_present(base_keys, equirect_second_keys, torch)
    second = torch.where(use_equirect_winding.reshape(-1, 1), second_equirect, second_cubemap)

    existing_faces = torch.stack((first, second), dim=1)
    alternate_faces = torch.stack((alt_first, alt_second), dim=1)

    expected_join_keys = None
    join_group_count = 0
    if join_group_for_vertex is not None and join_segments is not None:
        join_group_count = int(
            torch.as_tensor(
                join_group_for_vertex,
                device=device,
                dtype=torch.long,
            ).amax().item()
        ) + 1
        expected_join_pairs = torch.as_tensor(
            join_segments,
            device=device,
            dtype=torch.long,
        ).reshape(-1, 2)
        expected_join_keys = torch.sort(
            torch.minimum(
                expected_join_pairs[:, 0],
                expected_join_pairs[:, 1],
            ) * max(join_group_count, 1)
            + torch.maximum(
                expected_join_pairs[:, 0],
                expected_join_pairs[:, 1],
            )
        ).values

    def evaluate(face_pairs: Any) -> dict[str, Any]:
        flat_faces = face_pairs.reshape(-1, 3)
        depths = distance[flat_faces]
        jumps = _relative_jump(depths, torch).reshape(-1, 2)
        jump_rejections = (jumps > threshold) if threshold > 0.0 else torch.zeros_like(jumps, dtype=torch.bool)
        repair_rejections = repaired[face_pairs].any(dim=2)
        rejections = jump_rejections | repair_rejections
        footprint = torch.zeros_like(jumps)
        aspect = torch.zeros_like(jumps)
        invalid_metrics = torch.zeros_like(jumps, dtype=torch.bool)
        if vertices is not None and rays is not None:
            flat_footprint, flat_aspect, flat_invalid = _triangle_geometry_metrics(
                face_vertices=vertices[flat_faces],
                face_rays=rays[flat_faces],
                torch=torch,
            )
            footprint = flat_footprint.reshape(-1, 2)
            aspect = flat_aspect.reshape(-1, 2)
            invalid_metrics = flat_invalid.reshape(-1, 2)
            metric_rejections = invalid_metrics.clone()
            if footprint_threshold > 0.0:
                metric_rejections |= footprint > footprint_threshold
            if aspect_threshold > 0.0:
                metric_rejections |= aspect > aspect_threshold
            rejections |= metric_rejections
        contains_join = torch.zeros_like(rejections)
        if expected_join_keys is not None and int(expected_join_keys.numel()) > 0:
            provenance = join_group_for_vertex[flat_faces]
            edge_first = provenance[:, (0, 1, 2)]
            edge_second = provenance[:, (1, 2, 0)]
            both_join = (edge_first >= 0) & (edge_second >= 0)
            if safe_join_groups is not None:
                safe_groups_tensor = torch.as_tensor(
                    safe_join_groups,
                    device=device,
                    dtype=torch.bool,
                ).reshape(-1)
                both_join &= (
                    safe_groups_tensor[edge_first.clamp(min=0)]
                    & safe_groups_tensor[edge_second.clamp(min=0)]
                )
            logical_keys = (
                torch.minimum(edge_first, edge_second)
                * max(join_group_count, 1)
                + torch.maximum(edge_first, edge_second)
            )
            positions = torch.searchsorted(expected_join_keys, logical_keys)
            positions = torch.clamp(
                positions,
                max=int(expected_join_keys.numel()) - 1,
            )
            contains_join = (
                both_join
                & (expected_join_keys[positions] == logical_keys)
            ).any(dim=1).reshape(-1, 2)
        return {
            "rejections": rejections,
            "rejection_count": rejections.sum(dim=1),
            "join_rejections": (rejections & contains_join).sum(dim=1),
            "contains_join": contains_join,
            "jumps": jumps,
            "footprint": footprint,
            "aspect": aspect,
            "invalid_metrics": invalid_metrics,
            "repaired": repair_rejections,
            "max_jump": jumps.amax(dim=1),
        }

    existing = evaluate(existing_faces)
    alternate = evaluate(alternate_faces)
    existing_choice = (
        alternate["rejection_count"] < existing["rejection_count"]
    ) | (
        (alternate["rejection_count"] == existing["rejection_count"])
        & (
            (alternate["join_rejections"] < existing["join_rejections"])
            | (
                (alternate["join_rejections"] == existing["join_rejections"])
                & (alternate["max_jump"] < existing["max_jump"])
            )
        )
    )
    boundary_quads = existing["contains_join"].any(dim=1)
    boundary_choose_alternate = (
        alternate["rejection_count"] < existing["rejection_count"]
    ) | (
        (alternate["rejection_count"] == existing["rejection_count"])
        & existing_choice
    )
    choose_alternate = torch.where(
        boundary_quads,
        boundary_choose_alternate,
        existing_choice,
    )

    if adaptive_stats is not None:
        selected_rejections = torch.where(
            choose_alternate.reshape(-1, 1),
            alternate["rejections"],
            existing["rejections"],
        )
        unclosable = boundary_quads & selected_rejections.any(dim=1)
        diagnostic_rows: list[dict[str, Any]] = []
        for quad_index in unclosable.nonzero(
            as_tuple=False
        ).reshape(-1).detach().cpu().tolist():
            def triangle_metrics(values: dict[str, Any]) -> list[dict[str, Any]]:
                metrics: list[dict[str, Any]] = []
                for triangle_index in range(2):
                    depth_jump = float(
                        values["jumps"][quad_index, triangle_index].item()
                    )
                    footprint_ratio = float(
                        values["footprint"][quad_index, triangle_index].item()
                    )
                    aspect_ratio = float(
                        values["aspect"][quad_index, triangle_index].item()
                    )
                    invalid_metric = bool(
                        values["invalid_metrics"][
                            quad_index, triangle_index
                        ].item()
                    )
                    repaired_triangle = bool(
                        values["repaired"][
                            quad_index, triangle_index
                        ].item()
                    )
                    rejection_reasons = []
                    if repaired_triangle:
                        rejection_reasons.append("repaired")
                    if threshold > 0.0 and depth_jump > threshold:
                        rejection_reasons.append("depth_jump")
                    if invalid_metric:
                        rejection_reasons.append("invalid_metric")
                    if (
                        footprint_threshold > 0.0
                        and footprint_ratio > footprint_threshold
                    ):
                        rejection_reasons.append("footprint_ratio")
                    if (
                        aspect_threshold > 0.0
                        and aspect_ratio > aspect_threshold
                    ):
                        rejection_reasons.append("aspect_ratio")
                    metrics.append({
                        "passes": not bool(rejection_reasons),
                        "contains_cube_join": bool(
                            values["contains_join"][
                                quad_index, triangle_index
                            ].item()
                        ),
                        "depth_jump": depth_jump,
                        "footprint_ratio": footprint_ratio,
                        "aspect_ratio": aspect_ratio,
                        "invalid_metric": invalid_metric,
                        "repaired": repaired_triangle,
                        "rejection_reasons": rejection_reasons,
                    })
                return metrics

            diagnostic_rows.append({
                "quad_index": int(quad_index),
                "selected_diagonal": (
                    "alternate"
                    if bool(choose_alternate[quad_index].item())
                    else "existing"
                ),
                "existing": triangle_metrics(existing),
                "alternate": triangle_metrics(alternate),
            })
        adaptive_stats.update({
            "boundary_quads_evaluated": int(boundary_quads.sum().item()),
            "boundary_diagonal_reselections": int(
                (boundary_quads & choose_alternate).sum().item()
            ),
            "boundary_quads_unclosable": int(unclosable.sum().item()),
            "boundary_unclosable_metrics": diagnostic_rows,
        })
    selected_pairs = torch.where(choose_alternate.reshape(-1, 1, 1), alternate_faces, existing_faces).reshape(-1, 3)

    replaced_pairs = existing_faces.reshape(-1, 3)
    replaced_keys = _face_keys(replaced_pairs, vertex_count)
    extras = base_faces_tensor[~_keys_present(replaced_keys, base_keys, torch)]
    adaptive_faces = torch.cat((selected_pairs, extras), dim=0)
    expected = int(base_faces_tensor.shape[0])
    if int(adaptive_faces.shape[0]) != expected:
        raise RuntimeError(f"Adaptive topology mismatch: expected {expected}, got {adaptive_faces.shape[0]}.")
    return adaptive_faces, int(choose_alternate.sum().item())


def generate_equirectangular_quads(height: int, width: int, *, device: Any = "cpu") -> Any:
    torch = _require_torch()
    h = int(height)
    w = int(width)
    rows = torch.arange(0, h - 1, device=device, dtype=torch.long).repeat(w)
    cols = torch.arange(0, w, device=device, dtype=torch.long).repeat_interleave(max(h - 1, 0))
    top_left = rows * w + cols
    top_right = rows * w + (cols + 1) % w
    bottom_left = top_left + w
    bottom_right = (rows + 1) * w + (cols + 1) % w
    return torch.stack((top_left, top_right, bottom_left, bottom_right), dim=1)


def generate_cubemap_quads(face_size: int, *, device: Any = "cpu") -> Any:
    torch = _require_torch()
    size = int(face_size)
    if size < 1:
        raise ValueError("Cubemap face size must be at least 1.")
    quads: list[Any] = []
    if size > 1:
        gy, gx = torch.meshgrid(
            torch.arange(size - 1, device=device, dtype=torch.long),
            torch.arange(size - 1, device=device, dtype=torch.long),
            indexing="ij",
        )
        gy = gy.reshape(-1); gx = gx.reshape(-1)
        base = gy * size + gx
        face_quads = torch.stack((base, base + 1, base + size, base + size + 1), dim=1)
        for face_index in range(6):
            quads.append(face_quads + face_index * size * size)
    return torch.cat(quads, dim=0) if quads else torch.empty((0, 4), device=device, dtype=torch.long)

def _triangle_geometry_metrics(
    *,
    face_vertices: Any,
    face_rays: Any,
    torch: Any,
) -> tuple[Any, Any, Any]:
    """Return footprint ratio, normalized aspect, and invalid-metric mask."""

    edge_vectors = torch.stack(
        (
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 1],
            face_vertices[:, 0] - face_vertices[:, 2],
        ),
        dim=1,
    )
    actual_lengths = torch.linalg.norm(edge_vectors, dim=2)
    radii = torch.linalg.norm(face_vertices, dim=2)

    ray_norms = torch.linalg.norm(face_rays, dim=2)
    unit_rays = face_rays / torch.clamp(
        ray_norms,
        min=METRIC_EPSILON,
    ).unsqueeze(2)
    first_rays = unit_rays[:, (0, 1, 2)]
    second_rays = unit_rays[:, (1, 2, 0)]
    cosine = torch.sum(first_rays * second_rays, dim=2)
    sine = torch.linalg.norm(torch.cross(first_rays, second_rays, dim=2), dim=2)
    angles = torch.atan2(sine, cosine)
    minimum_radii = torch.minimum(
        radii[:, (0, 1, 2)],
        radii[:, (1, 2, 0)],
    )
    expected_lengths = 2.0 * minimum_radii * torch.sin(angles / 2.0)
    footprint_ratio = (
        actual_lengths / torch.clamp(expected_lengths, min=METRIC_EPSILON)
    ).amax(dim=1)

    longest_edge = actual_lengths.amax(dim=1)
    double_area = torch.linalg.norm(
        torch.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
            dim=1,
        ),
        dim=1,
    )
    area = double_area * 0.5
    aspect_ratio = torch.where(
        area > 0.0,
        math.sqrt(3.0) * longest_edge.square() / (4.0 * area),
        torch.full_like(area, float("inf")),
    )

    invalid_metrics = (
        ~torch.isfinite(face_vertices).all(dim=2).all(dim=1)
        | ~torch.isfinite(face_rays).all(dim=2).all(dim=1)
        | ~torch.isfinite(actual_lengths).all(dim=1)
        | ~torch.isfinite(expected_lengths).all(dim=1)
        | ~torch.isfinite(footprint_ratio)
        | ~torch.isfinite(aspect_ratio)
        | ~torch.isfinite(radii).all(dim=1)
        | (radii <= 0.0).any(dim=1)
        | ~torch.isfinite(ray_norms).all(dim=1)
        | (ray_norms <= METRIC_EPSILON).any(dim=1)
        | (longest_edge <= 0.0)
        | (area <= 0.0)
    )
    return footprint_ratio, aspect_ratio, invalid_metrics


def _final_edge_stats(
    *,
    faces: Any,
    vertex_count: int,
    torch: Any,
    join_group_for_vertex: Any | None = None,
    join_segments: Any | None = None,
    safe_join_groups: Any | None = None,
) -> dict[str, Any]:
    """Measure final topology, using cubemap provenance for logical joins."""

    empty = {
        "boundary_edges": 0,
        "nonmanifold_edges": 0,
        "incorrectly_oriented_manifold_edges": 0,
        "edge_watertight": False,
        "closure_status": "empty",
    }
    if int(faces.numel()) == 0:
        return empty

    first, second, third = faces.unbind(dim=1)
    directed_first = torch.cat((first, second, third))
    directed_second = torch.cat((second, third, first))
    edge_low = torch.minimum(directed_first, directed_second)
    edge_high = torch.maximum(directed_first, directed_second)
    edge_keys = edge_low * vertex_count + edge_high
    unique_edge_keys, inverse, edge_counts = torch.unique(
        edge_keys,
        return_inverse=True,
        return_counts=True,
    )
    orientation = torch.where(
        directed_first < directed_second,
        torch.ones_like(directed_first),
        -torch.ones_like(directed_first),
    )
    orientation_sums = torch.zeros(
        int(unique_edge_keys.numel()),
        device=faces.device,
        dtype=torch.long,
    )
    orientation_sums.scatter_add_(0, inverse, orientation)

    boundary_mask = edge_counts == 1
    boundary_edges = int(boundary_mask.sum().item())
    nonmanifold_edges = int((edge_counts > 2).sum().item())
    incorrectly_oriented = int(
        ((edge_counts == 2) & (orientation_sums.abs() == 2)).sum().item()
    )
    watertight = boundary_edges == 0 and nonmanifold_edges == 0
    stats: dict[str, Any] = {
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "incorrectly_oriented_manifold_edges": incorrectly_oriented,
        "edge_watertight": watertight,
        "closure_status": (
            "watertight"
            if watertight
            else ("nonmanifold" if nonmanifold_edges else "open_boundaries")
        ),
    }

    if join_group_for_vertex is None or join_segments is None or safe_join_groups is None:
        return stats

    provenance = torch.as_tensor(
        join_group_for_vertex,
        device=faces.device,
        dtype=torch.long,
    ).reshape(-1)
    expected = torch.as_tensor(
        join_segments,
        device=faces.device,
        dtype=torch.long,
    ).reshape(-1, 2)
    safe_groups = torch.as_tensor(
        safe_join_groups,
        device=faces.device,
        dtype=torch.bool,
    ).reshape(-1)
    expected_count = int(expected.shape[0])
    group_count = max(int(safe_groups.numel()), 1)
    if expected_count == 0:
        stats.update({
            "expected_join_segments": 0,
            "join_segment_incidence": {
                "0": 0,
                "1": 0,
                "2": 0,
                "over2": 0,
            },
            "boundary_edges_along_cube_joins": 0,
            "boundary_edges_touching_cube_joins": 0,
            "strictly_internal_boundary_edges": boundary_edges,
            "cube_joins_closed": True,
            "join_diagnostics_basis": "topology_provenance",
            "closure_status": (
                "closed_cube_joins"
                if boundary_edges == 0
                else "closed_cube_joins_with_internal_holes"
            ),
        })
        return stats

    expected_low = torch.minimum(expected[:, 0], expected[:, 1])
    expected_high = torch.maximum(expected[:, 0], expected[:, 1])
    expected_keys = expected_low * group_count + expected_high
    sorted_expected_keys, expected_order = torch.sort(expected_keys)

    first_groups = provenance[directed_first]
    second_groups = provenance[directed_second]
    logical_present = (first_groups >= 0) & (second_groups >= 0)
    logical_low = torch.minimum(first_groups, second_groups)
    logical_high = torch.maximum(first_groups, second_groups)
    logical_keys = logical_low * group_count + logical_high
    logical_positions = torch.searchsorted(sorted_expected_keys, logical_keys)
    logical_positions_clamped = torch.clamp(
        logical_positions,
        max=max(expected_count - 1, 0),
    )
    logical_matches = (
        logical_present
        & (expected_count > 0)
        & (sorted_expected_keys[logical_positions_clamped] == logical_keys)
    )
    sorted_incidence = torch.zeros(
        expected_count,
        device=faces.device,
        dtype=torch.long,
    )
    if bool(logical_matches.any().item()):
        sorted_incidence.scatter_add_(
            0,
            logical_positions_clamped[logical_matches],
            torch.ones_like(logical_positions_clamped[logical_matches]),
        )
    incidence = torch.zeros_like(sorted_incidence)
    incidence[expected_order] = sorted_incidence
    expected_safe = safe_groups[expected[:, 0]] & safe_groups[expected[:, 1]]
    incidence = torch.where(expected_safe, incidence, torch.zeros_like(incidence))
    incidence_histogram = {
        "0": int((incidence == 0).sum().item()),
        "1": int((incidence == 1).sum().item()),
        "2": int((incidence == 2).sum().item()),
        "over2": int((incidence > 2).sum().item()),
    }

    unique_edge_low = torch.div(
        unique_edge_keys,
        vertex_count,
        rounding_mode="floor",
    )
    unique_edge_high = unique_edge_keys % vertex_count
    boundary_low_groups = provenance[unique_edge_low]
    boundary_high_groups = provenance[unique_edge_high]
    boundary_has_join_vertex = (
        (boundary_low_groups >= 0) | (boundary_high_groups >= 0)
    )
    boundary_both_join_vertices = (
        (boundary_low_groups >= 0) & (boundary_high_groups >= 0)
    )
    boundary_logical_low = torch.minimum(
        boundary_low_groups,
        boundary_high_groups,
    )
    boundary_logical_high = torch.maximum(
        boundary_low_groups,
        boundary_high_groups,
    )
    boundary_logical_keys = (
        boundary_logical_low * group_count + boundary_logical_high
    )
    boundary_positions = torch.searchsorted(
        sorted_expected_keys,
        boundary_logical_keys,
    )
    boundary_positions_clamped = torch.clamp(
        boundary_positions,
        max=max(expected_count - 1, 0),
    )
    boundary_along_join = (
        boundary_mask
        & boundary_both_join_vertices
        & (expected_count > 0)
        & (
            sorted_expected_keys[boundary_positions_clamped]
            == boundary_logical_keys
        )
    )
    boundary_touching_join = boundary_mask & boundary_has_join_vertex
    along_join_count = int(boundary_along_join.sum().item())
    touching_join_count = int(boundary_touching_join.sum().item())
    internal_boundary_count = int(
        (boundary_mask & ~boundary_has_join_vertex).sum().item()
    )
    joins_open = (
        incidence_histogram["0"] > 0
        or incidence_histogram["1"] > 0
        or incidence_histogram["over2"] > 0
        or touching_join_count > 0
    )
    internal_holes = internal_boundary_count > 0
    closure_status = "open_cube_joins" if joins_open else "closed_cube_joins"
    if internal_holes:
        closure_status += "_with_internal_holes"
    if nonmanifold_edges:
        closure_status += "_nonmanifold"

    stats.update({
        "expected_join_segments": expected_count,
        "join_segment_incidence": incidence_histogram,
        "boundary_edges_along_cube_joins": along_join_count,
        "boundary_edges_touching_cube_joins": touching_join_count,
        "strictly_internal_boundary_edges": internal_boundary_count,
        "cube_joins_closed": not joins_open,
        "join_diagnostics_basis": "topology_provenance",
        "closure_status": closure_status,
    })
    return stats


def filter_mesh_topology(
    *,
    vertices: Any,
    vertex_colors: Any,
    distance: Any,
    triangles: Any,
    depth_jump_threshold: float = DEFAULT_DEPTH_JUMP_THRESHOLD,
    footprint_ratio_threshold: float = DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
    aspect_ratio_threshold: float = DEFAULT_ASPECT_RATIO_THRESHOLD,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
    mode: str = "synthetic",
    source_shape: Any | None = None,
    resized_shape: Any | None = None,
    max_size: int | None = None,
    extra_stats: Mapping[str, Any] | None = None,
    rays: Any | None = None,
    quads: Any | None = None,
    topology_classes: Any | None = None,
    weld_groups: Any | None = None,
    join_segments: Any | None = None,
) -> MeshBuildResult:
    """Repair invalid depth, reject unsafe triangles, and compact geometry."""

    torch = _require_torch()
    depth_threshold = _validated_threshold(
        depth_jump_threshold,
        label="Mesh depth-jump threshold",
    )
    footprint_threshold = _validated_threshold(
        footprint_ratio_threshold,
        label="Mesh footprint-ratio threshold",
    )
    aspect_threshold = _validated_threshold(
        aspect_ratio_threshold,
        label="Mesh aspect-ratio threshold",
    )
    chunk_size = int(face_chunk_size)
    if chunk_size < 1:
        raise ValueError("Mesh face chunk size must be at least 1.")

    vertices_tensor = torch.as_tensor(vertices).detach()
    if not torch.is_floating_point(vertices_tensor):
        vertices_tensor = vertices_tensor.float()
    device = vertices_tensor.device
    colors_tensor = torch.as_tensor(vertex_colors, device=device).detach()
    if not torch.is_floating_point(colors_tensor):
        colors_tensor = colors_tensor.float()
    distance_tensor = torch.as_tensor(distance, device=device).detach().reshape(-1)
    if not torch.is_floating_point(distance_tensor):
        distance_tensor = distance_tensor.float()
    faces_tensor = torch.as_tensor(
        triangles,
        device=device,
        dtype=torch.long,
    ).detach().reshape(-1, 3)

    if vertices_tensor.ndim != 2 or int(vertices_tensor.shape[1]) != 3:
        raise ValueError("Mesh vertices must have shape (N, 3).")
    if colors_tensor.ndim != 2 or int(colors_tensor.shape[1]) != 3:
        raise ValueError("Mesh vertex colors must have shape (N, 3).")
    vertex_count = int(vertices_tensor.shape[0])
    if int(colors_tensor.shape[0]) != vertex_count or int(distance_tensor.numel()) != vertex_count:
        raise ValueError("Vertices, colors, and distances must have the same vertex count.")
    if faces_tensor.numel() > 0:
        minimum_index = int(faces_tensor.min().item())
        maximum_index = int(faces_tensor.max().item())
        if minimum_index < 0 or maximum_index >= vertex_count:
            raise ValueError("Mesh topology contains an out-of-range vertex index.")

    if rays is None:
        finite_distance = torch.isfinite(distance_tensor) & (
            torch.abs(distance_tensor) > METRIC_EPSILON
        )
        inferred = torch.zeros_like(vertices_tensor)
        inferred[finite_distance] = (
            vertices_tensor[finite_distance]
            / distance_tensor[finite_distance].reshape(-1, 1)
        )
        norms = torch.linalg.norm(inferred, dim=1, keepdim=True)
        fallback = torch.tensor(
            [1.0, 0.0, 0.0],
            device=device,
            dtype=vertices_tensor.dtype,
        ).reshape(1, 3).repeat(vertex_count, 1)
        rays_tensor = torch.where(
            norms > METRIC_EPSILON,
            inferred / torch.clamp(norms, min=METRIC_EPSILON),
            fallback,
        )
    else:
        rays_tensor = torch.as_tensor(rays, device=device).detach().reshape(-1, 3)
        if int(rays_tensor.shape[0]) != vertex_count:
            raise ValueError("Mesh rays must have the same vertex count as vertices.")
        if not torch.is_floating_point(rays_tensor):
            rays_tensor = rays_tensor.float()
        ray_norms = torch.linalg.norm(rays_tensor, dim=1, keepdim=True)
        rays_tensor = rays_tensor / torch.clamp(
            ray_norms,
            min=METRIC_EPSILON,
        )

    finite_distance = torch.isfinite(distance_tensor)
    non_finite_distance = ~finite_distance
    non_positive_distance = finite_distance & (distance_tensor <= 0)
    below_minimum_distance = (
        finite_distance
        & (distance_tensor > 0)
        & (distance_tensor < MIN_VALID_DISTANCE_METERS)
    )
    generated_xyz_is_finite = torch.isfinite(vertices_tensor).all(dim=1)
    non_finite_xyz = (
        finite_distance
        & (distance_tensor >= MIN_VALID_DISTANCE_METERS)
        & (~generated_xyz_is_finite)
    )
    invalid_counts = {
        "non_finite_distance": int(non_finite_distance.sum().item()),
        "non_positive_distance": int(non_positive_distance.sum().item()),
        "below_minimum_distance": int(below_minimum_distance.sum().item()),
        "non_finite_xyz": int(non_finite_xyz.sum().item()),
    }
    invalid_counts["total"] = sum(invalid_counts.values())

    try:
        repaired_distance, repaired_vertices, repaired_mask, repair_rounds = _repair_invalid_depths(
            distance=distance_tensor,
            vertices=vertices_tensor,
            rays=rays_tensor,
            triangles=faces_tensor,
            torch=torch,
        )
    except MeshExportError as exc:
        stats = dict(exc.stats or {})
        vertices_stats = dict(stats.get("vertices", {}))
        vertices_stats.update(
            {
                "candidate": vertex_count,
                "invalid": invalid_counts,
                "repaired": 0,
                "duplicated_for_caps": 0,
                "exported": 0,
            }
        )
        stats.update(
            {
                "mode": str(mode),
                "source_shape": _shape_list(source_shape) if source_shape is not None else None,
                "resized_shape": _shape_list(resized_shape) if resized_shape is not None else None,
                "max_size": int(max_size) if max_size is not None else None,
                "depth_jump_threshold": depth_threshold,
                "footprint_ratio_threshold": footprint_threshold,
                "aspect_ratio_threshold": aspect_threshold,
                "min_valid_distance_meters": MIN_VALID_DISTANCE_METERS,
                "face_chunk_size": chunk_size,
                "vertices": vertices_stats,
                "triangles": {
                    "candidate": int(faces_tensor.shape[0]),
                    "caps": 0,
                    "jump_caps": 0,
                    "repaired_caps": 0,
                    "exported": 0,
                    "retention": 0.0,
                    "retention_percent": 0.0,
                },
            }
        )
        raise MeshExportError(str(exc), stats=stats) from exc

    classes_tensor = None
    topology_class_candidate_counts: list[int] | None = None
    if topology_classes is not None:
        classes_tensor = torch.as_tensor(topology_classes, device=device, dtype=torch.uint8).reshape(-1)
        if int(classes_tensor.numel()) != int(faces_tensor.shape[0]):
            raise ValueError("Topology classes must match the candidate triangle count.")
        topology_class_candidate_counts = [
            int((classes_tensor == class_id).sum().item())
            for class_id in range(3)
        ]

    weld_stats = {
        "candidate": 0,
        "welded": 0,
        "skipped": 0,
        "skipped_non_finite_or_invalid": 0,
        "skipped_repaired": 0,
        "skipped_unreferenced": 0,
        "skipped_depth_mismatch": 0,
    }
    weld_map = torch.arange(vertex_count, device=device, dtype=torch.long)
    join_group_for_vertex = None
    safe_join_groups = None
    if weld_groups is not None:
        groups = torch.as_tensor(
            weld_groups,
            device=device,
            dtype=torch.long,
        ).reshape(-1, 3)
        group_count = int(groups.shape[0])
        weld_stats["candidate"] = group_count
        present = groups >= 0
        safe_indices = groups.clamp(min=0)
        candidate_referenced = torch.zeros(
            vertex_count,
            device=device,
            dtype=torch.bool,
        )
        if faces_tensor.numel() > 0:
            candidate_referenced[faces_tensor.reshape(-1)] = True
        finite_valid = (
            torch.isfinite(distance_tensor[safe_indices])
            & (distance_tensor[safe_indices] >= MIN_VALID_DISTANCE_METERS)
            & torch.isfinite(vertices_tensor[safe_indices]).all(dim=2)
        ) | ~present
        non_finite_or_invalid = ~finite_valid.all(dim=1)
        repaired_group = (repaired_mask[safe_indices] & present).any(dim=1)
        unreferenced_group = (
            (~candidate_referenced[safe_indices]) & present
        ).any(dim=1)
        group_depths = torch.where(
            present,
            repaired_distance[safe_indices],
            torch.full_like(repaired_distance[safe_indices], float("inf")),
        )
        min_depth = group_depths.amin(dim=1)
        max_depth = torch.where(
            present,
            group_depths,
            torch.full_like(group_depths, float("-inf")),
        ).amax(dim=1)
        mismatch = (
            (max_depth - min_depth)
            / torch.clamp(min_depth, min=1e-6)
        ) > depth_threshold
        eligible = ~non_finite_or_invalid
        weld_stats["skipped_non_finite_or_invalid"] = int(
            non_finite_or_invalid.sum().item()
        )
        skipped = eligible & repaired_group
        weld_stats["skipped_repaired"] = int(skipped.sum().item())
        eligible &= ~repaired_group
        skipped = eligible & unreferenced_group
        weld_stats["skipped_unreferenced"] = int(skipped.sum().item())
        eligible &= ~unreferenced_group
        skipped = eligible & mismatch
        weld_stats["skipped_depth_mismatch"] = int(skipped.sum().item())
        safe_join_groups = eligible & ~mismatch
        weld_stats["welded"] = int(safe_join_groups.sum().item())
        weld_stats["skipped"] = (
            weld_stats["candidate"] - weld_stats["welded"]
        )

        group_rows = torch.arange(
            group_count,
            device=device,
            dtype=torch.long,
        ).reshape(-1, 1).expand_as(groups)
        join_group_for_vertex = torch.full(
            (vertex_count,),
            -1,
            device=device,
            dtype=torch.long,
        )
        join_group_for_vertex[safe_indices[present]] = group_rows[present]

        member_count = present.sum(dim=1).clamp(min=1).reshape(-1, 1)
        position_sum = (
            repaired_vertices[safe_indices]
            * present.unsqueeze(2)
        ).sum(dim=1)
        color_sum = (
            colors_tensor[safe_indices]
            * present.unsqueeze(2)
        ).sum(dim=1)
        reconciled_vertices = position_sum / member_count
        reconciled_colors = color_sum / member_count
        reconciled_distances = torch.linalg.norm(
            reconciled_vertices,
            dim=1,
        )
        reconciled_rays = reconciled_vertices / torch.clamp(
            reconciled_distances,
            min=METRIC_EPSILON,
        ).reshape(-1, 1)

        safe_members = present & safe_join_groups.reshape(-1, 1)
        member_indices = safe_indices[safe_members]
        member_rows = group_rows[safe_members]
        if member_indices.numel() > 0:
            repaired_vertices[member_indices] = reconciled_vertices[member_rows]
            colors_tensor[member_indices] = reconciled_colors[member_rows]
            repaired_distance[member_indices] = reconciled_distances[member_rows]
            rays_tensor[member_indices] = reconciled_rays[member_rows]
            representative_candidates = torch.where(
                present,
                groups,
                torch.full_like(groups, vertex_count),
            )
            representatives = representative_candidates.amin(dim=1)
            weld_map[member_indices] = representatives[member_rows]
            faces_tensor = weld_map[faces_tensor]

    adaptive_diagonals = 0
    adaptive_selection_stats: dict[str, Any] = {
        "boundary_quads_evaluated": 0,
        "boundary_diagonal_reselections": 0,
        "boundary_quads_unclosable": 0,
        "boundary_unclosable_metrics": [],
    }
    candidate_face_count = int(faces_tensor.shape[0])
    suppressed_seam_count = 0
    if quads is not None:
        quads_tensor = torch.as_tensor(quads, device=device, dtype=torch.long).reshape(-1, 4)
        quads_tensor = weld_map[quads_tensor]
        if classes_tensor is None:
            faces_tensor, adaptive_diagonals = _build_adaptive_faces(
                faces_tensor,
                quads_tensor,
                repaired_distance,
                repaired_mask,
                depth_threshold,
                torch,
                vertices=repaired_vertices,
                rays=rays_tensor,
                footprint_threshold=footprint_threshold,
                aspect_threshold=aspect_threshold,
                join_group_for_vertex=join_group_for_vertex,
                join_segments=join_segments,
                safe_join_groups=safe_join_groups,
                adaptive_stats=adaptive_selection_stats,
            )
        else:
            face_grid = classes_tensor == 0
            faces_tensor, adaptive_diagonals = _build_adaptive_faces(
                faces_tensor[face_grid],
                quads_tensor,
                repaired_distance,
                repaired_mask,
                depth_threshold,
                torch,
                vertices=repaired_vertices,
                rays=rays_tensor,
                footprint_threshold=footprint_threshold,
                aspect_threshold=aspect_threshold,
                join_group_for_vertex=join_group_for_vertex,
                join_segments=join_segments,
                safe_join_groups=safe_join_groups,
                adaptive_stats=adaptive_selection_stats,
            )
            suppressed_seam_count = int((~face_grid).sum().item())
            classes_tensor = torch.zeros(
                int(faces_tensor.shape[0]),
                device=device,
                dtype=torch.uint8,
            )
    elif classes_tensor is not None:
        face_grid = classes_tensor == 0
        faces_tensor = faces_tensor[face_grid]
        suppressed_seam_count = int((~face_grid).sum().item())
        classes_tensor = torch.zeros(
            int(faces_tensor.shape[0]),
            device=device,
            dtype=torch.uint8,
        )

    retained_chunks: list[Any] = []
    retained_class_chunks: list[Any] = []
    removed_index_degenerate_count = 0
    removed_invalid_or_repaired_count = 0
    removed_depth_discontinuity_count = 0
    removed_invalid_metric_count = 0
    removed_footprint_ratio_count = 0
    removed_aspect_ratio_count = 0
    join_failure_keys = (
        "index_degenerate",
        "invalid_or_repaired",
        "depth_discontinuity",
        "invalid_metric",
        "footprint_ratio",
        "aspect_ratio",
    )
    safe_join_failure_stats = {
        "along_cube_joins": {key: 0 for key in join_failure_keys},
        "touching_cube_joins": {key: 0 for key in join_failure_keys},
    }
    sorted_join_keys = None
    join_group_count = 0
    if (
        join_group_for_vertex is not None
        and join_segments is not None
        and safe_join_groups is not None
    ):
        join_group_count = max(int(safe_join_groups.numel()), 1)
        expected_join_pairs = torch.as_tensor(
            join_segments,
            device=device,
            dtype=torch.long,
        ).reshape(-1, 2)
        sorted_join_keys = torch.sort(
            torch.minimum(
                expected_join_pairs[:, 0],
                expected_join_pairs[:, 1],
            ) * join_group_count
            + torch.maximum(
                expected_join_pairs[:, 0],
                expected_join_pairs[:, 1],
            )
        ).values

    def safe_join_adjacency(chunk: Any) -> tuple[Any, Any]:
        if (
            sorted_join_keys is None
            or int(sorted_join_keys.numel()) == 0
            or int(chunk.numel()) == 0
        ):
            empty_mask = torch.zeros(
                int(chunk.shape[0]),
                device=device,
                dtype=torch.bool,
            )
            return empty_mask, empty_mask
        provenance = join_group_for_vertex[chunk]
        on_join = provenance >= 0
        safe_provenance = safe_join_groups[provenance.clamp(min=0)]
        safely_touching = (
            on_join.any(dim=1)
            & (safe_provenance | ~on_join).all(dim=1)
        )
        edge_first = provenance[:, (0, 1, 2)]
        edge_second = provenance[:, (1, 2, 0)]
        edge_present = (edge_first >= 0) & (edge_second >= 0)
        edge_safe = (
            safe_join_groups[edge_first.clamp(min=0)]
            & safe_join_groups[edge_second.clamp(min=0)]
        )
        logical_keys = (
            torch.minimum(edge_first, edge_second) * join_group_count
            + torch.maximum(edge_first, edge_second)
        )
        positions = torch.searchsorted(sorted_join_keys, logical_keys)
        positions = torch.clamp(
            positions,
            max=int(sorted_join_keys.numel()) - 1,
        )
        safely_along = (
            edge_present
            & edge_safe
            & (sorted_join_keys[positions] == logical_keys)
        ).any(dim=1)
        return safely_touching, safely_along

    def record_join_failures(
        key: str,
        rejected: Any,
        touching: Any,
        along: Any,
    ) -> None:
        safe_join_failure_stats["touching_cube_joins"][key] += int(
            (rejected & touching).sum().item()
        )
        safe_join_failure_stats["along_cube_joins"][key] += int(
            (rejected & along).sum().item()
        )

    for start in range(0, candidate_face_count, chunk_size):
        chunk = faces_tensor[start : start + chunk_size]
        chunk_classes = classes_tensor[start : start + chunk_size] if classes_tensor is not None else None
        if chunk.numel() == 0:
            continue
        chunk_touches_safe_join, chunk_along_safe_join = safe_join_adjacency(
            chunk
        )

        index_degenerate_mask = (
            (chunk[:, 0] == chunk[:, 1])
            | (chunk[:, 1] == chunk[:, 2])
            | (chunk[:, 0] == chunk[:, 2])
        )
        removed_index_degenerate_count += int(index_degenerate_mask.sum().item())
        record_join_failures(
            "index_degenerate",
            index_degenerate_mask,
            chunk_touches_safe_join,
            chunk_along_safe_join,
        )
        chunk = chunk[~index_degenerate_mask]
        chunk_touches_safe_join = chunk_touches_safe_join[
            ~index_degenerate_mask
        ]
        chunk_along_safe_join = chunk_along_safe_join[
            ~index_degenerate_mask
        ]
        if chunk_classes is not None:
            chunk_classes = chunk_classes[~index_degenerate_mask]
        if chunk.numel() == 0:
            continue

        repaired_face_mask = repaired_mask[chunk].any(dim=1)
        removed_invalid_or_repaired_count += int(repaired_face_mask.sum().item())
        record_join_failures(
            "invalid_or_repaired",
            repaired_face_mask,
            chunk_touches_safe_join,
            chunk_along_safe_join,
        )
        chunk = chunk[~repaired_face_mask]
        chunk_touches_safe_join = chunk_touches_safe_join[
            ~repaired_face_mask
        ]
        chunk_along_safe_join = chunk_along_safe_join[
            ~repaired_face_mask
        ]
        if chunk_classes is not None:
            chunk_classes = chunk_classes[~repaired_face_mask]
        if chunk.numel() == 0:
            continue

        jumps = _relative_jump(repaired_distance[chunk], torch)
        depth_discontinuity_mask = (
            (jumps > depth_threshold)
            if depth_threshold > 0.0
            else torch.zeros_like(jumps, dtype=torch.bool)
        )
        removed_depth_discontinuity_count += int(
            depth_discontinuity_mask.sum().item()
        )
        record_join_failures(
            "depth_discontinuity",
            depth_discontinuity_mask,
            chunk_touches_safe_join,
            chunk_along_safe_join,
        )
        chunk = chunk[~depth_discontinuity_mask]
        chunk_touches_safe_join = chunk_touches_safe_join[
            ~depth_discontinuity_mask
        ]
        chunk_along_safe_join = chunk_along_safe_join[
            ~depth_discontinuity_mask
        ]
        if chunk_classes is not None:
            chunk_classes = chunk_classes[~depth_discontinuity_mask]
        if chunk.numel() == 0:
            continue

        footprint_ratio, aspect_ratio, invalid_metric_mask = _triangle_geometry_metrics(
            face_vertices=repaired_vertices[chunk],
            face_rays=rays_tensor[chunk],
            torch=torch,
        )
        removed_invalid_metric_count += int(invalid_metric_mask.sum().item())
        record_join_failures(
            "invalid_metric",
            invalid_metric_mask,
            chunk_touches_safe_join,
            chunk_along_safe_join,
        )
        chunk = chunk[~invalid_metric_mask]
        chunk_touches_safe_join = chunk_touches_safe_join[
            ~invalid_metric_mask
        ]
        chunk_along_safe_join = chunk_along_safe_join[
            ~invalid_metric_mask
        ]
        if chunk_classes is not None:
            chunk_classes = chunk_classes[~invalid_metric_mask]
        footprint_ratio = footprint_ratio[~invalid_metric_mask]
        aspect_ratio = aspect_ratio[~invalid_metric_mask]
        if chunk.numel() == 0:
            continue

        footprint_mask = (
            (footprint_ratio > footprint_threshold)
            if footprint_threshold > 0.0
            else torch.zeros_like(footprint_ratio, dtype=torch.bool)
        )
        removed_footprint_ratio_count += int(footprint_mask.sum().item())
        record_join_failures(
            "footprint_ratio",
            footprint_mask,
            chunk_touches_safe_join,
            chunk_along_safe_join,
        )
        chunk = chunk[~footprint_mask]
        chunk_touches_safe_join = chunk_touches_safe_join[
            ~footprint_mask
        ]
        chunk_along_safe_join = chunk_along_safe_join[
            ~footprint_mask
        ]
        if chunk_classes is not None:
            chunk_classes = chunk_classes[~footprint_mask]
        aspect_ratio = aspect_ratio[~footprint_mask]
        if chunk.numel() == 0:
            continue

        aspect_mask = (
            (aspect_ratio > aspect_threshold)
            if aspect_threshold > 0.0
            else torch.zeros_like(aspect_ratio, dtype=torch.bool)
        )
        removed_aspect_ratio_count += int(aspect_mask.sum().item())
        record_join_failures(
            "aspect_ratio",
            aspect_mask,
            chunk_touches_safe_join,
            chunk_along_safe_join,
        )
        retained = chunk[~aspect_mask]
        if chunk_classes is not None:
            chunk_classes = chunk_classes[~aspect_mask]
        if retained.numel() > 0:
            retained_chunks.append(retained)
            if chunk_classes is not None:
                retained_class_chunks.append(chunk_classes)

    retained_faces = (
        torch.cat(retained_chunks, dim=0)
        if retained_chunks
        else torch.empty((0, 3), device=device, dtype=torch.long)
    )
    class_names = ("face_grid", "seam_strip", "corner")
    class_stats: dict[str, dict[str, int]] = {}
    if topology_class_candidate_counts is not None:
        retained_face_grid = int(retained_faces.shape[0])
        for class_id, class_name in enumerate(class_names):
            candidate_count = topology_class_candidate_counts[class_id]
            final_retained = retained_face_grid if class_id == 0 else 0
            filtered = (
                candidate_count - final_retained
                if class_id == 0
                else 0
            )
            class_stats[class_name] = {
                "candidate": candidate_count,
                "retained": final_retained,
                "removed": candidate_count - final_retained,
                "added": 0,
                "filtered": filtered,
                "suppressed_after_weld": candidate_count if class_id else 0,
            }

    referenced = torch.zeros(vertex_count, device=device, dtype=torch.bool)
    if retained_faces.numel() > 0:
        referenced[retained_faces.reshape(-1)] = True

    edge_stats = _final_edge_stats(
        faces=retained_faces,
        vertex_count=vertex_count,
        torch=torch,
        join_group_for_vertex=join_group_for_vertex,
        join_segments=join_segments,
        safe_join_groups=safe_join_groups,
    )

    referenced_indices = referenced.nonzero(as_tuple=False).reshape(-1)
    exported_vertex_count = int(referenced_indices.numel())
    remap = torch.full((vertex_count,), -1, device=device, dtype=torch.long)
    if exported_vertex_count:
        remap[referenced_indices] = torch.arange(
            exported_vertex_count,
            device=device,
            dtype=torch.long,
        )

    output_faces = remap[retained_faces]
    if mode == "cubemap":
        output_faces = output_faces[:, [0, 2, 1]]
    output_vertices = repaired_vertices[referenced_indices]
    output_colors = colors_tensor[referenced_indices]

    exported_face_count = int(output_faces.shape[0])
    removed_total = (
        removed_index_degenerate_count
        + removed_invalid_or_repaired_count
        + removed_depth_discontinuity_count
        + removed_invalid_metric_count
        + removed_footprint_ratio_count
        + removed_aspect_ratio_count
        + suppressed_seam_count
    )
    retention = (
        exported_face_count / candidate_face_count
        if candidate_face_count
        else 0.0
    )
    retention_percent = 100.0 * retention
    for failures in safe_join_failure_stats.values():
        failures["total"] = sum(failures.values())
    triangle_stats = {
        "candidate": candidate_face_count,
        "removed_invalid_or_repaired": removed_invalid_or_repaired_count,
        "removed_invalid_vertices": removed_invalid_or_repaired_count,
        "removed_depth_discontinuity": removed_depth_discontinuity_count,
        "removed_non_finite_or_degenerate_metrics": removed_invalid_metric_count,
        "removed_footprint_ratio": removed_footprint_ratio_count,
        "removed_aspect_ratio": removed_aspect_ratio_count,
        "removed_index_degenerate": removed_index_degenerate_count,
        "regular": exported_face_count,
        "added": 0,
        "caps": 0,
        "cap_triangles": 0,
        "bridge_triangles": 0,
        "jump_caps": 0,
        "repaired_caps": 0,
        "adaptive_diagonals": adaptive_diagonals,
        "boundary_diagonal_reselections": adaptive_selection_stats[
            "boundary_diagonal_reselections"
        ],
        "boundary_adaptive": adaptive_selection_stats,
        "safe_join_adjacent_failures": safe_join_failure_stats,
        "classes": class_stats,
        "removed_seam_or_corner_candidates": suppressed_seam_count,
        "exported": exported_face_count,
        "total_exported": exported_face_count,
        "retention": retention,
        "retention_percent": retention_percent,
        "retained_percent": retention_percent,
        "estimated_angular_coverage_percent": retention_percent,
    }
    repaired_vertex_count = int(repaired_mask.sum().item())
    vertex_stats = {
        "candidate": vertex_count,
        "invalid": invalid_counts,
        "repaired": repaired_vertex_count,
        "repair_rounds": int(repair_rounds),
        "removed_unreferenced_valid": max(
            0,
            vertex_count - repaired_vertex_count - exported_vertex_count,
        ),
        "duplicated_for_caps": 0,
        "exported": exported_vertex_count,
    }
    stats: dict[str, Any] = {
        "mode": str(mode),
        "source_shape": _shape_list(source_shape) if source_shape is not None else None,
        "resized_shape": _shape_list(resized_shape) if resized_shape is not None else None,
        "max_size": int(max_size) if max_size is not None else None,
        "depth_jump_threshold": depth_threshold,
        "footprint_ratio_threshold": footprint_threshold,
        "aspect_ratio_threshold": aspect_threshold,
        "min_valid_distance_meters": MIN_VALID_DISTANCE_METERS,
        "face_chunk_size": chunk_size,
        "vertices": vertex_stats,
        "triangles": triangle_stats,
        "welds": weld_stats,
        **edge_stats,
    }
    if extra_stats:
        stats.update(dict(extra_stats))

    if removed_total + exported_face_count != candidate_face_count:
        raise RuntimeError("DreamCube mesh reconstruction statistics failed to reconcile.")
    if exported_face_count == 0:
        raise MeshExportError(
            "DreamCube mesh export produced no triangles after rejection "
            f"({candidate_face_count} candidate, {removed_total} removed).",
            stats=stats,
        )

    vertices_cpu = output_vertices.to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous().numpy()
    colors_cpu = output_colors.to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous().numpy()
    faces_cpu = output_faces.to(
        device="cpu",
        dtype=torch.int64,
    ).contiguous().numpy()
    return MeshBuildResult(vertices_cpu, faces_cpu, colors_cpu, stats)

def _target_device(torch: Any, device: Any | None) -> Any:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resize_scale(height: int, width: int, max_size: int | None) -> float:
    if max_size is None:
        return 1.0
    limit = int(max_size)
    if limit < 1:
        raise ValueError("Mesh max_size must be positive or None.")
    largest = max(int(height), int(width))
    return limit / largest if largest > limit else 1.0


def equi_unit_rays(height: int, width: int, *, device: Any) -> Any:
    """DreamCube right-handed rays: X left, Y up, Z forward."""

    torch = _require_torch()
    u = (torch.arange(width, device=device).float() + 0.5) / width
    v = (torch.arange(height, device=device).float() + 0.5) / height
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    phi = uu * 2 * torch.pi - torch.pi
    theta = torch.pi / 2 - vv * torch.pi
    x = -torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)


def cube_unit_rays(face_size: int, *, device: Any) -> Any:
    """DreamCube cubemap rays in +Z,-X,-Z,+X,+Y,-Y order."""

    torch = _require_torch()
    size = int(face_size)
    u = (torch.arange(size, device=device).float() + 0.5) / size
    v = (torch.arange(size, device=device).float() + 0.5) / size
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    rays = (
        torch.stack((1 - uu * 2, 1 - vv * 2, torch.ones_like(vv)), dim=-1),
        torch.stack((-torch.ones_like(vv), 1 - vv * 2, 1 - uu * 2), dim=-1),
        torch.stack((uu * 2 - 1, 1 - vv * 2, -torch.ones_like(vv)), dim=-1),
        torch.stack((torch.ones_like(vv), 1 - vv * 2, uu * 2 - 1), dim=-1),
        torch.stack((1 - uu * 2, torch.ones_like(vv), vv * 2 - 1), dim=-1),
        torch.stack((1 - uu * 2, -torch.ones_like(vv), -vv * 2 + 1), dim=-1),
    )
    stacked = torch.stack(rays, dim=0)
    return stacked / torch.linalg.norm(stacked, dim=-1, keepdim=True)


def build_rgbd_equi_mesh(
    rgb: Any,
    distance: Any,
    *,
    rays: Any | None = None,
    max_size: int | None = None,
    device: Any | None = None,
    depth_jump_threshold: float = DEFAULT_DEPTH_JUMP_THRESHOLD,
    footprint_ratio_threshold: float = DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
    aspect_ratio_threshold: float = DEFAULT_ASPECT_RATIO_THRESHOLD,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
) -> MeshBuildResult:
    """Reconstruct and filter an equirectangular RGB-D mesh."""

    torch = _require_torch()
    import torch.nn.functional as functional

    rgb_tensor = torch.as_tensor(rgb)
    distance_tensor = torch.as_tensor(distance)
    if rgb_tensor.ndim != 3 or int(rgb_tensor.shape[-1]) != 3:
        raise ValueError("Equirectangular RGB must have shape (H, W, 3).")
    if distance_tensor.ndim != 2 or tuple(rgb_tensor.shape[:2]) != tuple(distance_tensor.shape):
        raise ValueError("Equirectangular RGB and distance shapes must match.")
    source_shape = tuple(int(value) for value in distance_tensor.shape)
    target_device = _target_device(torch, device)
    rgb_tensor = rgb_tensor.to(target_device)
    distance_tensor = distance_tensor.to(target_device)
    if not torch.is_floating_point(distance_tensor):
        distance_tensor = distance_tensor.float()

    if rays is None:
        rays_tensor = equi_unit_rays(source_shape[0], source_shape[1], device=target_device)
    else:
        rays_tensor = torch.as_tensor(rays, device=target_device)
        if rays_tensor.ndim != 3 or tuple(rays_tensor.shape) != source_shape + (3,):
            raise ValueError("Equirectangular rays must have shape (H, W, 3).")
        if not torch.is_floating_point(rays_tensor):
            rays_tensor = rays_tensor.float()

    scale = _resize_scale(source_shape[0], source_shape[1], max_size)
    rgb_resized = functional.interpolate(
        (rgb_tensor / 255.0).permute(2, 0, 1).unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    ).squeeze(0).permute(1, 2, 0)
    distance_resized = functional.interpolate(
        distance_tensor.unsqueeze(0).unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    ).squeeze(0).squeeze(0)
    rays_resized = functional.interpolate(
        rays_tensor.permute(2, 0, 1).unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    ).squeeze(0).permute(1, 2, 0)
    rays_resized = rays_resized / (torch.linalg.norm(rays_resized, dim=-1, keepdim=True) + 1e-8)

    resized_shape = tuple(int(value) for value in distance_resized.shape)
    vertices = distance_resized.reshape(-1, 1) * rays_resized.reshape(-1, 3)
    colors = rgb_resized.reshape(-1, 3)
    topology = generate_equirectangular_topology(*resized_shape, device=target_device)
    quads = generate_equirectangular_quads(*resized_shape, device=target_device)
    return filter_mesh_topology(
        vertices=vertices,
        vertex_colors=colors,
        distance=distance_resized,
        triangles=topology,
        depth_jump_threshold=depth_jump_threshold,
        footprint_ratio_threshold=footprint_ratio_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        face_chunk_size=face_chunk_size,
        mode="equirectangular",
        source_shape=source_shape,
        resized_shape=resized_shape,
        max_size=max_size,
        extra_stats={"horizontal_wrap": True},
        rays=rays_resized.reshape(-1, 3),
        quads=quads,
    )


def build_rgbd_cube_mesh(
    rgb: Any,
    distance: Any,
    *,
    rays: Any | None = None,
    max_size: int | None = None,
    device: Any | None = None,
    depth_jump_threshold: float = DEFAULT_DEPTH_JUMP_THRESHOLD,
    footprint_ratio_threshold: float = DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
    aspect_ratio_threshold: float = DEFAULT_ASPECT_RATIO_THRESHOLD,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
) -> MeshBuildResult:
    """Reconstruct and filter a closed DreamCube cubemap RGB-D mesh."""

    torch = _require_torch()
    import torch.nn.functional as functional

    rgb_tensor = torch.as_tensor(rgb)
    distance_tensor = torch.as_tensor(distance)
    if rgb_tensor.ndim != 4 or int(rgb_tensor.shape[-1]) != 3:
        raise ValueError("Cubemap RGB must have shape (6, H, W, 3).")
    if distance_tensor.ndim != 3 or tuple(rgb_tensor.shape[:3]) != tuple(distance_tensor.shape):
        raise ValueError("Cubemap RGB and distance shapes must match.")
    if int(rgb_tensor.shape[0]) != 6:
        raise ValueError("Cubemap input must contain six faces in +Z,-X,-Z,+X,+Y,-Y order.")
    if int(rgb_tensor.shape[1]) != int(rgb_tensor.shape[2]):
        raise ValueError("Cubemap faces must be square.")

    source_shape = tuple(int(value) for value in distance_tensor.shape)
    target_device = _target_device(torch, device)
    rgb_tensor = rgb_tensor.to(target_device)
    distance_tensor = distance_tensor.to(target_device)
    if not torch.is_floating_point(distance_tensor):
        distance_tensor = distance_tensor.float()
    scale = _resize_scale(source_shape[1], source_shape[2], max_size)
    rgb_resized = functional.interpolate(
        (rgb_tensor / 255.0).permute(0, 3, 1, 2),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    ).permute(0, 2, 3, 1)
    distance_resized = functional.interpolate(
        distance_tensor.unsqueeze(1),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=False,
    ).squeeze(1)
    resized_shape = tuple(int(value) for value in distance_resized.shape)

    if rays is None:
        rays_resized = cube_unit_rays(resized_shape[1], device=target_device)
    else:
        rays_tensor = torch.as_tensor(rays, device=target_device)
        if rays_tensor.ndim != 4 or tuple(rays_tensor.shape) != source_shape + (3,):
            raise ValueError("Cubemap rays must have shape (6, H, W, 3).")
        if not torch.is_floating_point(rays_tensor):
            rays_tensor = rays_tensor.float()
        rays_resized = functional.interpolate(
            rays_tensor.permute(0, 3, 1, 2),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ).permute(0, 2, 3, 1)
        rays_resized = rays_resized / (torch.linalg.norm(rays_resized, dim=-1, keepdim=True) + 1e-8)

    vertices = distance_resized.reshape(-1, 1) * rays_resized.reshape(-1, 3)
    colors = rgb_resized.reshape(-1, 3)
    topology_plan = generate_cubemap_topology_plan(resized_shape[1], device=target_device)
    topology = topology_plan.triangles
    quads = generate_cubemap_quads(resized_shape[1], device=target_device)
    return filter_mesh_topology(
        vertices=vertices,
        vertex_colors=colors,
        distance=distance_resized,
        triangles=topology,
        depth_jump_threshold=depth_jump_threshold,
        footprint_ratio_threshold=footprint_ratio_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        face_chunk_size=face_chunk_size,
        mode="cubemap",
        source_shape=source_shape,
        resized_shape=resized_shape,
        max_size=max_size,
        extra_stats={
            "face_order": list(CUBEMAP_FACE_ORDER),
            "closed_topology": {
                "face_grids": 6,
                "edge_seams": 12,
                "corner_triangles": 8,
            },
        },
        rays=rays_resized.reshape(-1, 3),
        quads=quads,
        topology_classes=topology_plan.triangle_classes,
        weld_groups=topology_plan.weld_groups,
        join_segments=topology_plan.join_segments,
    )


def _write_open3d_mesh(
    vertices: Any,
    triangles: Any,
    vertex_colors: Any,
    *,
    save_path: str | Path | None = None,
) -> Any:
    """Create and optionally write a mesh using real Open3D or the setup shim."""

    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - setup installs real provider or shim
        raise RuntimeError("DreamCube mesh export requires the Open3D provider or setup shim.") from exc

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = o3d.io.write_triangle_mesh(str(path), mesh)
        if written is False or not path.is_file():
            raise RuntimeError(f"Open3D did not write the DreamCube mesh to {path}.")
    return mesh


def _export_build_result(build: MeshBuildResult, save_path: str | Path | None) -> MeshExportResult:
    try:
        mesh = _write_open3d_mesh(
            build.vertices,
            build.triangles,
            build.vertex_colors,
            save_path=save_path,
        )
    except MeshExportError:
        raise
    except Exception as exc:
        raise MeshExportError(f"DreamCube mesh writing failed: {exc}", stats=build.stats) from exc
    return MeshExportResult(mesh=mesh, stats=build.stats)


def convert_rgbd_equi_to_mesh(
    rgb: Any,
    distance: Any,
    *,
    rays: Any | None = None,
    max_size: int | None = None,
    device: Any | None = None,
    save_path: str | Path | None = None,
    depth_jump_threshold: float = DEFAULT_DEPTH_JUMP_THRESHOLD,
    footprint_ratio_threshold: float = DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
    aspect_ratio_threshold: float = DEFAULT_ASPECT_RATIO_THRESHOLD,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
) -> MeshExportResult:
    """Build, filter, compact, and optionally write an equirectangular mesh."""

    return _export_build_result(
        build_rgbd_equi_mesh(
            rgb,
            distance,
            rays=rays,
            max_size=max_size,
            device=device,
            depth_jump_threshold=depth_jump_threshold,
            footprint_ratio_threshold=footprint_ratio_threshold,
            aspect_ratio_threshold=aspect_ratio_threshold,
            face_chunk_size=face_chunk_size,
        ),
        save_path,
    )


def convert_rgbd_cube_to_mesh(
    rgb: Any,
    distance: Any,
    *,
    rays: Any | None = None,
    max_size: int | None = None,
    device: Any | None = None,
    save_path: str | Path | None = None,
    depth_jump_threshold: float = DEFAULT_DEPTH_JUMP_THRESHOLD,
    footprint_ratio_threshold: float = DEFAULT_FOOTPRINT_RATIO_THRESHOLD,
    aspect_ratio_threshold: float = DEFAULT_ASPECT_RATIO_THRESHOLD,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
) -> MeshExportResult:
    """Build, filter, compact, and optionally write a closed cubemap mesh."""

    return _export_build_result(
        build_rgbd_cube_mesh(
            rgb,
            distance,
            rays=rays,
            max_size=max_size,
            device=device,
            depth_jump_threshold=depth_jump_threshold,
            footprint_ratio_threshold=footprint_ratio_threshold,
            aspect_ratio_threshold=aspect_ratio_threshold,
            face_chunk_size=face_chunk_size,
        ),
        save_path,
    )


__all__ = [
    "CUBEMAP_FACE_ORDER",
    "DEFAULT_ASPECT_RATIO_THRESHOLD",
    "DEFAULT_DEPTH_JUMP_THRESHOLD",
    "DEFAULT_FOOTPRINT_RATIO_THRESHOLD",
    "MIN_VALID_DISTANCE_METERS",
    "MeshBuildResult",
    "MeshExportError",
    "MeshExportResult",
    "build_rgbd_cube_mesh",
    "build_rgbd_equi_mesh",
    "convert_rgbd_cube_to_mesh",
    "convert_rgbd_equi_to_mesh",
    "cube_unit_rays",
    "cubemap_candidate_triangle_count",
    "equi_unit_rays",
    "equirectangular_candidate_triangle_count",
    "filter_mesh_topology",
    "generate_cubemap_topology",
    "generate_equirectangular_topology",
    "generate_equirectangular_quads",
    "generate_cubemap_quads",
]
