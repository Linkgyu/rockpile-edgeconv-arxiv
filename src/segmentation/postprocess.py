"""Post-processing utilities for oversized predicted rockpile clusters."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree


def _relabel_nonnegative(labels: np.ndarray) -> np.ndarray:
    out = np.full(len(labels), -1, dtype=int)
    valid = labels >= 0
    if np.any(valid):
        _, inv = np.unique(labels[valid], return_inverse=True)
        out[valid] = inv
    return out


def _height_markers_xy(
    points: np.ndarray,
    grid_resolution_m: float,
    min_peak_distance_m: float,
    peak_prominence_m: float,
) -> np.ndarray:
    xy = points[:, :2]
    z = points[:, 2]
    xy_min = xy.min(axis=0)
    ij = np.floor((xy - xy_min) / grid_resolution_m).astype(int)
    shape = ij.max(axis=0) + 1
    if shape[0] < 3 or shape[1] < 3:
        return np.empty((0, 2), dtype=float)

    height = np.full((shape[0], shape[1]), np.nan, dtype=float)
    for (i, j), value in zip(ij, z):
        if np.isnan(height[i, j]) or value > height[i, j]:
            height[i, j] = value
    finite = np.isfinite(height)
    if not np.any(finite):
        return np.empty((0, 2), dtype=float)

    fill = np.nanmin(height[finite])
    height = np.where(finite, height, fill)
    smooth = gaussian_filter(height, sigma=1.0)
    footprint = max(3, int(round(min_peak_distance_m / grid_resolution_m)))
    if footprint % 2 == 0:
        footprint += 1
    local_max = smooth == maximum_filter(smooth, size=footprint, mode="nearest")
    threshold = np.nanpercentile(smooth[finite], 55.0) + peak_prominence_m
    peak_cells = np.argwhere(local_max & finite & (smooth >= threshold))
    if len(peak_cells) <= 1:
        return np.empty((0, 2), dtype=float)

    peak_values = smooth[peak_cells[:, 0], peak_cells[:, 1]]
    order = np.argsort(peak_values)[::-1]
    selected = []
    min_sep_cells = max(2.0, min_peak_distance_m / grid_resolution_m)
    for cell in peak_cells[order]:
        if all(np.linalg.norm(cell - prev) >= min_sep_cells for prev in selected):
            selected.append(cell)
    if len(selected) <= 1:
        return np.empty((0, 2), dtype=float)
    selected = np.asarray(selected, dtype=float)
    return xy_min + (selected[:, [0, 1]] + 0.5) * grid_resolution_m


def split_oversized_clusters_by_height_markers(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    max_cluster_points: int = 260,
    grid_resolution_m: float = 0.030,
    min_peak_distance_m: float = 0.11,
    peak_prominence_m: float = 0.015,
    min_child_points: int = 18,
    max_markers_per_cluster: int = 10,
) -> np.ndarray:
    """Split large connected components using local height maxima in XY."""

    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(labels, dtype=int)
    out = labels.copy()
    next_label = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0

    for label in sorted(set(labels)):
        if label < 0:
            continue
        idx = np.flatnonzero(labels == label)
        if len(idx) < max_cluster_points:
            continue
        cluster = points[idx]
        markers = _height_markers_xy(cluster, grid_resolution_m, min_peak_distance_m, peak_prominence_m)
        if len(markers) <= 1:
            continue
        if len(markers) > max_markers_per_cluster:
            markers = markers[:max_markers_per_cluster]
        _, child = cKDTree(markers).query(cluster[:, :2], k=1)
        counts = np.bincount(child, minlength=len(markers))
        if np.any(counts < min_child_points):
            continue
        out[idx] = -1
        for child_id in range(len(markers)):
            child_idx = idx[child == child_id]
            if len(child_idx) >= min_child_points:
                out[child_idx] = next_label
                next_label += 1

    return _relabel_nonnegative(out)


def absorb_unlabelled_points_by_edge_affinity(
    labels: np.ndarray,
    edges: np.ndarray,
    probabilities: np.ndarray,
    absorb_threshold: float = 0.90,
    max_passes: int = 2,
) -> np.ndarray:
    """Attach noise points to adjacent labelled components when affinity is high.

    This is useful for domains where a strict connected-component threshold
    fragments true rocks and leaves many points labelled as noise.  Only
    currently unlabelled points are absorbed; existing component labels are not
    merged with each other.
    """

    out = np.asarray(labels, dtype=int).copy()
    edges = np.asarray(edges, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    candidate_edges = edges[probabilities >= absorb_threshold]
    if len(candidate_edges) == 0:
        return _relabel_nonnegative(out)

    for _ in range(max_passes):
        changed = 0
        votes: dict[int, list[int]] = {}
        for a, b in candidate_edges:
            la = int(out[a])
            lb = int(out[b])
            if la < 0 <= lb:
                votes.setdefault(int(a), []).append(lb)
            elif lb < 0 <= la:
                votes.setdefault(int(b), []).append(la)
        for point_idx, labels_for_point in votes.items():
            if out[point_idx] >= 0 or not labels_for_point:
                continue
            counts = np.bincount(np.asarray(labels_for_point, dtype=int))
            out[point_idx] = int(np.argmax(counts))
            changed += 1
        if changed == 0:
            break
    return _relabel_nonnegative(out)
