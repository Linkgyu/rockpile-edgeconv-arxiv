"""Exterior-only scan construction from labelled synthetic point clouds."""

from __future__ import annotations

import numpy as np


def keep_xy_height_envelope(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray | None = None,
    grid_resolution_m: float = 0.035,
    z_tolerance_m: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the top exterior surface envelope in each XY grid cell."""

    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(labels)
    if source_indices is None:
        source_indices = np.arange(len(points), dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)

    if len(points) == 0:
        return points, labels, source_indices

    origin_xy = points[:, :2].min(axis=0)
    cells = np.floor((points[:, :2] - origin_xy) / float(grid_resolution_m)).astype(np.int64)
    keys = cells[:, 0] * 10_000_000 + cells[:, 1]
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]

    keep = np.zeros(len(points), dtype=bool)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_keys[end] == sorted_keys[start]:
            end += 1
        idx = order[start:end]
        z_max = float(points[idx, 2].max())
        keep[idx] = points[idx, 2] >= z_max - float(z_tolerance_m)
        start = end

    kept = np.flatnonzero(keep)
    return points[kept], labels[kept], source_indices[kept]


def exterior_points_from_viewpoints(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    viewpoint_xyz_list: np.ndarray,
    angular_resolution_deg: float = 0.22,
    height_envelope_grid_m: float | None = 0.035,
    height_envelope_tolerance_m: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep points visible from at least one exterior viewpoint.

    For each viewpoint, points are assigned to angular azimuth/elevation bins.
    The closest point in each bin is retained. The union over viewpoints is then
    optionally filtered to the upper XY height envelope so interior/contact
    samples do not survive as pseudo-visible exterior points.
    """

    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(labels)
    viewpoints = np.asarray(viewpoint_xyz_list, dtype=float)
    visible = set()

    for vp in viewpoints:
        vectors = points - vp[None, :]
        ranges = np.linalg.norm(vectors, axis=1)
        safe = np.clip(ranges, 1e-12, None)
        azimuth = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
        elevation = np.degrees(np.arcsin(np.clip(vectors[:, 2] / safe, -1.0, 1.0)))
        az_bin = np.floor((azimuth + 180.0) / angular_resolution_deg).astype(np.int64)
        el_bin = np.floor((elevation + 90.0) / angular_resolution_deg).astype(np.int64)
        key = az_bin * 1_000_000 + el_bin
        order = np.lexsort((ranges, key))
        sorted_key = key[order]
        first = np.r_[True, sorted_key[1:] != sorted_key[:-1]]
        visible.update(order[first].tolist())

    visible_idx = np.array(sorted(visible), dtype=np.int64)
    exterior_points = points[visible_idx]
    exterior_labels = labels[visible_idx]
    if height_envelope_grid_m is not None:
        return keep_xy_height_envelope(
            exterior_points,
            exterior_labels,
            source_indices=visible_idx,
            grid_resolution_m=height_envelope_grid_m,
            z_tolerance_m=height_envelope_tolerance_m,
        )
    return exterior_points, exterior_labels, visible_idx


def default_viewpoints(points_xyz: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """Create four side viewpoints and one overhead viewpoint around a pile."""

    points = np.asarray(points_xyz, dtype=float)
    center = points.mean(axis=0)
    span = points.max(axis=0) - points.min(axis=0)
    r = float(max(span[0], span[1]) * 1.8 + margin)
    z_mid = float(center[2] + 0.35 * span[2])
    z_top = float(points[:, 2].max() + r)
    return np.array(
        [
            [center[0] + r, center[1], z_mid],
            [center[0] - r, center[1], z_mid],
            [center[0], center[1] + r, z_mid],
            [center[0], center[1] - r, z_mid],
            [center[0], center[1], z_top],
        ],
        dtype=float,
    )
