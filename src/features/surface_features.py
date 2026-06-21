"""Surface features and graph construction for exterior point clouds."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def estimate_normals_curvature(points_xyz: np.ndarray, k_neighbors: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local PCA normals and curvature."""

    points = np.asarray(points_xyz, dtype=float)
    k = int(np.clip(k_neighbors, 4, max(4, len(points) - 1)))
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1, workers=-1)
    idx = idx[:, 1:]
    nb = points[idx]
    centered = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
    normals[normals[:, 2] < 0] *= -1
    curvature = eigvals[:, 0] / np.clip(eigvals.sum(axis=1), 1e-12, None)
    return normals.astype(np.float32), curvature.astype(np.float32)


def knn_edges(points_xyz: np.ndarray, k: int = 12) -> np.ndarray:
    """Return undirected unique kNN edges as an array of shape (E, 2)."""

    points = np.asarray(points_xyz, dtype=float)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1, workers=-1)
    src = np.repeat(np.arange(len(points)), k)
    dst = idx[:, 1:].reshape(-1)
    edges = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int64)


def edge_features(points_xyz: np.ndarray, normals: np.ndarray, curvature: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Build neural edge features for same-fragment affinity prediction."""

    p = np.asarray(points_xyz, dtype=np.float32)
    n = np.asarray(normals, dtype=np.float32)
    c = np.asarray(curvature, dtype=np.float32)
    i = edges[:, 0]
    j = edges[:, 1]
    delta = p[j] - p[i]
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    normal_dot = np.sum(n[i] * n[j], axis=1, keepdims=True)
    normal_angle = np.arccos(np.clip(np.abs(normal_dot), -1.0, 1.0)) / np.pi
    curv_i = c[i, None]
    curv_j = c[j, None]
    curv_delta = np.abs(curv_i - curv_j)
    mid = 0.5 * (p[i] + p[j])
    z_delta = np.abs(delta[:, 2:3])
    xy_dist = np.linalg.norm(delta[:, :2], axis=1, keepdims=True)
    return np.hstack(
        [
            delta,
            np.abs(delta),
            dist,
            xy_dist,
            z_delta,
            normal_dot,
            normal_angle,
            curv_i,
            curv_j,
            curv_delta,
            mid[:, 2:3],
        ]
    ).astype(np.float32)
