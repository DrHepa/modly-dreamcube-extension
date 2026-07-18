"""Joint post-processing for sparse cubemap relative-depth predictions.

The input predictions are independent Depth Anything V2 face estimates.  This
module aligns only faces that share an observed cube edge, maps the resulting
relative z estimates to an explicitly non-metric millimetre range, converts
z-distance to radial distance for 90-degree faces, and reconciles only observed
shared borders and corners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dreamcube_manual_cubemap import EDGE_PAIRS, FACE_ORDER


ESTIMATED_Z_RANGE_MM = (1000.0, 5000.0)
PERCENTILE_RANGE = (2.0, 98.0)
_EPSILON = 1e-8


@dataclass(frozen=True)
class CubemapDepthResult:
    """Sparse radial-depth outputs and JSON-safe post-process diagnostics."""

    depths_mm: dict[str, np.ndarray]
    metadata: dict[str, Any]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}

    def find(self, item: tuple[str, int, int]) -> tuple[str, int, int]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(
        self,
        first: tuple[str, int, int],
        second: tuple[str, int, int],
    ) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _edge(array: np.ndarray, edge_name: str) -> np.ndarray:
    if edge_name == "left":
        return array[:, 0]
    if edge_name == "right":
        return array[:, -1]
    if edge_name == "top":
        return array[0, :]
    if edge_name == "bottom":
        return array[-1, :]
    raise ValueError(f"Unknown cubemap edge: {edge_name!r}")


def _edge_coordinate(edge_name: str, index: int, size: int) -> tuple[int, int]:
    if edge_name == "left":
        return index, 0
    if edge_name == "right":
        return index, size - 1
    if edge_name == "top":
        return 0, index
    if edge_name == "bottom":
        return size - 1, index
    raise ValueError(f"Unknown cubemap edge: {edge_name!r}")


def _observed_edge_pairs(faces: set[str]) -> tuple[tuple[str, str, str, str, bool], ...]:
    return tuple(
        pair
        for pair in EDGE_PAIRS
        if pair[0] in faces and pair[2] in faces
    )


def _connected_components(faces: set[str]) -> list[tuple[str, ...]]:
    adjacency = {face: set() for face in faces}
    for first_face, _first_edge, second_face, _second_edge, _reverse in _observed_edge_pairs(faces):
        adjacency[first_face].add(second_face)
        adjacency[second_face].add(first_face)

    components: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for root in FACE_ORDER:
        if root not in faces or root in visited:
            continue
        pending = [root]
        component: set[str] = set()
        while pending:
            face = pending.pop()
            if face in component:
                continue
            component.add(face)
            visited.add(face)
            pending.extend(adjacency[face] - component)
        components.append(tuple(face for face in FACE_ORDER if face in component))
    return components


def _sanitize_predictions(
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    unknown = sorted(set(predictions).difference(FACE_ORDER))
    if unknown:
        raise ValueError(f"Unknown cubemap face prediction(s): {', '.join(unknown)}")
    if "front" not in predictions:
        raise ValueError("Cubemap depth estimation requires the front face.")

    sanitized: dict[str, np.ndarray] = {}
    finite_metadata: dict[str, dict[str, Any]] = {}
    expected_shape: tuple[int, int] | None = None
    for face in FACE_ORDER:
        if face not in predictions:
            continue
        array = np.asarray(predictions[face], dtype=np.float64)
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"Cubemap depth prediction for {face} must be a non-empty 2D array.")
        if array.shape[0] != array.shape[1]:
            raise ValueError(f"Cubemap depth prediction for {face} must be square; got {array.shape}.")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError(
                "All supplied cubemap depth predictions must share one square size; "
                f"expected {expected_shape}, got {array.shape} for {face}."
            )

        finite = np.isfinite(array)
        finite_values = array[finite]
        fill_value = float(np.median(finite_values)) if finite_values.size else 0.0
        clean = array.copy()
        clean[~finite] = fill_value
        sanitized[face] = clean
        finite_metadata[face] = {
            "replaced_non_finite": int(clean.size - finite_values.size),
            "fill_value": fill_value,
            "raw_min": float(clean.min()),
            "raw_max": float(clean.max()),
        }
    return sanitized, finite_metadata


def _joint_normalize(
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values = np.concatenate([predictions[face].reshape(-1) for face in FACE_ORDER if face in predictions])
    low, high = np.percentile(values, PERCENTILE_RANGE)
    span = float(high - low)
    constant = not np.isfinite(span) or span < _EPSILON
    if constant:
        normalized = {
            face: np.full_like(predictions[face], 0.5, dtype=np.float64)
            for face in FACE_ORDER
            if face in predictions
        }
        span = 0.0
    else:
        normalized = {
            face: (predictions[face] - float(low)) / span
            for face in FACE_ORDER
            if face in predictions
        }
    return normalized, {
        "percentiles": [float(low), float(high)],
        "percentile_range": list(PERCENTILE_RANGE),
        "span": span,
        "constant_mapping": constant,
    }


def _fit_component(
    normalized: Mapping[str, np.ndarray],
    component: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    gauge = "front" if "front" in component else component[0]
    transforms = {gauge: {"scale": 1.0, "shift": 0.0}}
    if len(component) == 1:
        return {gauge: normalized[gauge].copy()}, {
            "faces": list(component),
            "gauge": gauge,
            "singleton": True,
            "observed_edge_count": 0,
            "fit_equations": 0,
            "transforms": transforms,
        }

    component_set = set(component)
    observed = _observed_edge_pairs(component_set)
    unknown_faces = [face for face in component if face != gauge]
    offsets = {face: index * 2 for index, face in enumerate(unknown_faces)}
    rows: list[np.ndarray] = []
    targets: list[float] = []

    for first_face, first_edge, second_face, second_edge, reverse_second in observed:
        first_values = _edge(normalized[first_face], first_edge)
        second_values = _edge(normalized[second_face], second_edge)
        if reverse_second:
            second_values = second_values[::-1]
        for first_value, second_value in zip(first_values, second_values):
            row = np.zeros(len(unknown_faces) * 2, dtype=np.float64)
            target = 0.0
            if first_face == gauge:
                target -= float(first_value)
            else:
                offset = offsets[first_face]
                row[offset] += float(first_value)
                row[offset + 1] += 1.0
            if second_face == gauge:
                target += float(second_value)
            else:
                offset = offsets[second_face]
                row[offset] -= float(second_value)
                row[offset + 1] -= 1.0
            rows.append(row)
            targets.append(target)

    regularization = 0.25
    for face in unknown_faces:
        offset = offsets[face]
        scale_row = np.zeros(len(unknown_faces) * 2, dtype=np.float64)
        scale_row[offset] = regularization
        rows.append(scale_row)
        targets.append(regularization)
        shift_row = np.zeros(len(unknown_faces) * 2, dtype=np.float64)
        shift_row[offset + 1] = regularization
        rows.append(shift_row)
        targets.append(0.0)

    solution, *_ = np.linalg.lstsq(
        np.stack(rows),
        np.asarray(targets, dtype=np.float64),
        rcond=None,
    )
    aligned = {gauge: normalized[gauge].copy()}
    for face in unknown_faces:
        offset = offsets[face]
        scale = float(solution[offset])
        shift = float(solution[offset + 1])
        if (
            not np.isfinite(scale)
            or scale < 0.1
            or scale > 10.0
            or not np.isfinite(shift)
        ):
            scale, shift = 1.0, 0.0
        transforms[face] = {"scale": scale, "shift": shift}
        aligned[face] = normalized[face] * scale + shift

    return aligned, {
        "faces": list(component),
        "gauge": gauge,
        "singleton": False,
        "observed_edge_count": len(observed),
        "fit_equations": len(rows),
        "transforms": transforms,
    }


def _reconcile_observed_boundaries(
    radial_float: dict[str, np.ndarray],
    observed: tuple[tuple[str, str, str, str, bool], ...],
) -> int:
    if not observed:
        return 0
    size = next(iter(radial_float.values())).shape[0]
    groups = _UnionFind()
    for first_face, first_edge, second_face, second_edge, reverse_second in observed:
        for index in range(size):
            second_index = size - 1 - index if reverse_second else index
            first_row, first_column = _edge_coordinate(first_edge, index, size)
            second_row, second_column = _edge_coordinate(second_edge, second_index, size)
            groups.union(
                (first_face, first_row, first_column),
                (second_face, second_row, second_column),
            )

    members: dict[tuple[str, int, int], list[tuple[str, int, int]]] = {}
    for item in groups.parent:
        members.setdefault(groups.find(item), []).append(item)
    for items in members.values():
        shared_value = float(np.mean([
            radial_float[face][row, column]
            for face, row, column in items
        ]))
        for face, row, column in items:
            radial_float[face][row, column] = shared_value
    return len(members)


def postprocess_cubemap_depths(
    predictions: Mapping[str, np.ndarray],
) -> CubemapDepthResult:
    """Convert a valid front-containing sparse face set to radial uint16 PNG data."""

    sanitized, finite_metadata = _sanitize_predictions(predictions)
    normalized, normalization_metadata = _joint_normalize(sanitized)
    supplied_faces = {face for face in FACE_ORDER if face in normalized}
    components = _connected_components(supplied_faces)

    aligned: dict[str, np.ndarray] = {}
    component_metadata: list[dict[str, Any]] = []
    for component in components:
        component_aligned, diagnostics = _fit_component(normalized, component)
        aligned.update(component_aligned)
        component_metadata.append(diagnostics)

    aligned_values = np.concatenate([aligned[face].reshape(-1) for face in FACE_ORDER if face in aligned])
    aligned_low, aligned_high = np.percentile(aligned_values, PERCENTILE_RANGE)
    aligned_span = float(aligned_high - aligned_low)
    constant_mapping = not np.isfinite(aligned_span) or aligned_span < _EPSILON

    size = next(iter(aligned.values())).shape[0]
    coordinates = ((np.arange(size, dtype=np.float64) + 0.5) / size) * 2.0 - 1.0
    xx, yy = np.meshgrid(coordinates, coordinates)
    radial_factor = np.sqrt(1.0 + xx * xx + yy * yy)
    minimum_z, maximum_z = ESTIMATED_Z_RANGE_MM

    radial_float: dict[str, np.ndarray] = {}
    for face in FACE_ORDER:
        if face not in aligned:
            continue
        if constant_mapping:
            near = np.full_like(aligned[face], 0.5, dtype=np.float64)
        else:
            near = np.clip((aligned[face] - float(aligned_low)) / aligned_span, 0.0, 1.0)
        z_mm = maximum_z - near * (maximum_z - minimum_z)
        radial_float[face] = z_mm * radial_factor

    observed = _observed_edge_pairs(supplied_faces)
    boundary_group_count = _reconcile_observed_boundaries(radial_float, observed)
    depths = {
        face: np.rint(radial_float[face]).clip(1, 65535).astype(np.uint16)
        for face in FACE_ORDER
        if face in radial_float
    }

    degraded = len(components) > 1
    warning = None
    if degraded:
        warning = (
            "Supplied cubemap faces form disconnected adjacency components; "
            "relative affine alignment cannot cross the missing-face gaps."
        )

    metadata: dict[str, Any] = {
        "face_order": [face for face in FACE_ORDER if face in depths],
        "semantics": "estimated relative depth; radial millimetre encoding is non-metric",
        "metric": False,
        "finite_handling": finite_metadata,
        "joint_normalization": normalization_metadata,
        "components": component_metadata,
        "component_count": len(components),
        "quality_status": "degraded" if degraded else "estimated",
        "degraded": degraded,
        "warning": warning,
        "global_z_mapping": {
            "aligned_percentiles": [float(aligned_low), float(aligned_high)],
            "percentile_range": list(PERCENTILE_RANGE),
            "z_range_mm": [int(minimum_z), int(maximum_z)],
            "constant_mapping": constant_mapping,
        },
        "radial_conversion": "z_mm * sqrt(1 + x^2 + y^2) for 90-degree perspective faces",
        "observed_shared_edge_count": len(observed),
        "boundary_equivalence_groups": boundary_group_count,
        "missing_faces_fabricated": False,
    }
    return CubemapDepthResult(depths_mm=depths, metadata=metadata)
