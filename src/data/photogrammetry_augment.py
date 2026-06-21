"""Photogrammetry-like synthetic point-cloud augmentation.

The synthetic scenes are metrically clean compared with real SfM/MVS point
clouds. These helpers perturb only the model inputs and keep the saved graph
and fragment labels unchanged, so they can be used during training without
regenerating the whole benchmark.
"""

from __future__ import annotations

import numpy as np


def _renormalize_normals(normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float32)
    n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-8, None)
    n[n[:, 2] < 0] *= -1
    return n.astype(np.float32)


def apply_photogrammetry_realism(scene: dict, seed: int, strength: float = 1.0) -> dict:
    """Return a copy of a scene with SfM/MVS-style measurement perturbations.

    The transformation mimics the most common synthetic-to-real gaps:
    nonuniform density, millimetre-scale coordinate jitter, softened normals,
    local curvature inflation, and a small fraction of background-like noisy
    points. Graph topology and labels are preserved, making this a conservative
    augmentation for edge-affinity training.
    """

    rng = np.random.default_rng(seed)
    out = {key: value.copy() if hasattr(value, "copy") else value for key, value in scene.items()}

    points = np.asarray(out["points_xyz"], dtype=np.float32).copy()
    normals = np.asarray(out["normals"], dtype=np.float32).copy()
    curvature = np.asarray(out["curvature"], dtype=np.float32).copy()

    z = points[:, 2]
    z_span = max(float(z.max() - z.min()), 1e-6)
    z_norm = (z - z.min()) / z_span
    radial = np.linalg.norm(points[:, :2] - points[:, :2].mean(axis=0, keepdims=True), axis=1)
    radial /= max(float(np.percentile(radial, 98)), 1e-6)

    density_weight = 0.55 + 0.45 * z_norm
    density_weight *= np.clip(1.10 - 0.35 * radial, 0.45, 1.10)
    density_weight *= rng.uniform(0.80, 1.20, size=len(points))

    jitter_sigma = (0.0035 + 0.0050 * radial + 0.0020 * (1.0 - z_norm)) * strength
    points += rng.normal(0.0, jitter_sigma[:, None], size=points.shape).astype(np.float32)
    points[:, 2] += rng.normal(0.0, 0.0025 * strength, size=len(points)).astype(np.float32)
    points[:, 2] -= points[:, 2].min()

    normal_noise = rng.normal(0.0, 0.035 * strength, size=normals.shape).astype(np.float32)
    normals = _renormalize_normals(normals + normal_noise)

    curvature_scale = rng.lognormal(mean=0.0, sigma=0.22 * strength, size=len(curvature)).astype(np.float32)
    curvature = np.clip(curvature * curvature_scale + rng.uniform(0.0, 0.006 * strength, len(curvature)), 0.0, None)

    edge_features = np.asarray(out["edge_features"], dtype=np.float32).copy()
    edges = out["edges"].astype(np.int64)
    i = edges[:, 0]
    j = edges[:, 1]
    delta = points[j] - points[i]
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    xy_dist = np.linalg.norm(delta[:, :2], axis=1, keepdims=True)
    z_delta = np.abs(delta[:, 2:3])
    normal_dot = np.sum(normals[i] * normals[j], axis=1, keepdims=True)
    normal_angle = np.arccos(np.clip(np.abs(normal_dot), -1.0, 1.0)) / np.pi
    curv_i = curvature[i, None]
    curv_j = curvature[j, None]
    curv_delta = np.abs(curv_i - curv_j)
    mid_z = 0.5 * (points[i, 2:3] + points[j, 2:3])
    edge_features[:, :] = np.hstack(
        [delta, np.abs(delta), dist, xy_dist, z_delta, normal_dot, normal_angle, curv_i, curv_j, curv_delta, mid_z]
    ).astype(np.float32)

    # Low-confidence regions in real photogrammetry tend to be sparse/noisy.
    # Encode this by slightly depressing geometric consistency on affected edges.
    sparse_point = density_weight < np.percentile(density_weight, 18)
    sparse_edge = sparse_point[i] | sparse_point[j]
    edge_features[sparse_edge, 9:11] *= rng.uniform(0.70, 0.95, size=(sparse_edge.sum(), 2)).astype(np.float32)
    edge_features[sparse_edge, -1] += rng.normal(0.0, 0.01 * strength, size=sparse_edge.sum()).astype(np.float32)

    out["points_xyz"] = points.astype(np.float32)
    out["normals"] = normals.astype(np.float32)
    out["curvature"] = curvature.astype(np.float32)
    out["edge_features"] = edge_features.astype(np.float32)
    out["photogrammetry_realism_strength"] = np.array([strength], dtype=np.float32)
    return out
