"""Surface-size proxy for exterior-only fragments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pca_span(points: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 3:
        return np.zeros(3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh.T
    return np.percentile(proj, hi, axis=0) - np.percentile(proj, lo, axis=0)


def estimate_surface_proxy(
    points_xyz: np.ndarray,
    labels: np.ndarray,
    min_points: int = 10,
    second_span_factor: float = 1.15,
) -> pd.DataFrame:
    """Estimate visible-surface PSD proxy diameters for labelled clusters.

    The second PCA-span factor is an empirical, fixed pre-test calibration for
    exterior surface clusters. Manuscript sensitivity checks report the effect
    of varying this factor between 1.00 and 1.20.
    """
    points = np.asarray(points_xyz, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        cluster = points[labels == lab]
        if len(cluster) < min_points:
            continue
        spans = np.maximum(_pca_span(cluster), 0)
        d = float(max(spans[0], second_span_factor * spans[1], 1e-6))
        rows.append({
            "cluster_id": int(lab),
            "n_points": int(len(cluster)),
            "diameter_proxy_m": d,
            "diameter_proxy_mm": d * 1000,
            "proxy_volume_m3": float(np.pi / 6.0 * d ** 3),
        })
    return pd.DataFrame(rows)
