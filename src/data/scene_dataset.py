"""Scene loading utilities for the 100-pile Synthetic_Rockpile-style dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_INDEX = REPO_ROOT / "results" / "tables" / "dem_noboundary_relax150_100scene_index.csv"


def load_scene_index(path: Path | str | None = None) -> pd.DataFrame:
    """Load the 100-scene train/validation/test index."""

    index_path = SCENE_INDEX if path is None else Path(path)
    df = pd.read_csv(index_path)
    def resolve_scene_path(value: str) -> str:
        original = Path(value)
        if original.exists():
            return str(original)
        local = REPO_ROOT / "data" / "processed" / original.name
        return str(local if local.exists() else original)

    df["path"] = df["path"].astype(str).map(resolve_scene_path)
    return df


def load_npz_scene(row_or_path) -> dict:
    """Load one scene npz as a plain dictionary."""

    path = Path(row_or_path["path"] if hasattr(row_or_path, "__getitem__") and "path" in row_or_path else row_or_path)
    data = np.load(path)
    return {key: data[key] for key in data.files}


def point_features(points_xyz: np.ndarray, normals: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """Build normalized point features for EdgeConv."""

    points = np.asarray(points_xyz, dtype=np.float32)
    center = points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points - center, axis=1))
    points_norm = (points - center) / max(float(scale), 1e-6)
    normals = np.asarray(normals, dtype=np.float32)
    curvature = np.asarray(curvature, dtype=np.float32).reshape(-1, 1)
    curv_scale = np.percentile(curvature, 95) if np.any(curvature > 0) else 1.0
    curvature = np.clip(curvature / max(float(curv_scale), 1e-6), 0, 3)
    return np.hstack([points_norm, normals, curvature]).astype(np.float32)


def directed_edges(edges: np.ndarray) -> np.ndarray:
    """Convert undirected edge pairs to directed edge_index shape (2, 2E)."""

    e = np.asarray(edges, dtype=np.int64)
    rev = e[:, [1, 0]]
    both = np.vstack([e, rev])
    return both.T.astype(np.int64)


def edge_geom_features(edge_features: np.ndarray) -> np.ndarray:
    """Normalize saved geometric edge features for the classifier head."""

    x = np.asarray(edge_features, dtype=np.float32).copy()
    # Distance-like columns in the saved feature vector are metres; scale to a
    # convenient range while leaving dot/angle/curvature cues intact.
    x[:, :9] *= 10.0
    x[:, -1:] *= 2.0
    return x.astype(np.float32)


def scene_to_torch(scene: dict, device: torch.device | str = "cpu") -> dict:
    """Convert one scene dictionary to torch tensors."""

    x = point_features(scene["points_xyz"], scene["normals"], scene["curvature"])
    edge_index = directed_edges(scene["edges"])
    edge_attr = edge_geom_features(scene["edge_features"])
    out = {
        "x": torch.from_numpy(x).to(device),
        "edge_index": torch.from_numpy(edge_index).long().to(device),
        "edges": torch.from_numpy(scene["edges"].astype(np.int64)).long().to(device),
        "edge_attr": torch.from_numpy(edge_attr).float().to(device),
        "edge_y": torch.from_numpy(scene["edge_same_fragment"].astype(np.float32)).float().to(device),
        "points_xyz": scene["points_xyz"],
        "instance_labels": scene["instance_labels"],
        "ground_truth_P80_mm": float(scene["ground_truth_P80_mm"][0]),
    }
    return out


def balanced_edge_indices(y: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    """Balanced positive/negative edge sample indices."""

    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(bool)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    n_each = min(len(pos), len(neg), max_edges // 2)
    if n_each == 0:
        raise ValueError("Scene has no positive or negative edges")
    idx = np.r_[rng.choice(pos, n_each, replace=False), rng.choice(neg, n_each, replace=False)]
    rng.shuffle(idx)
    return idx.astype(np.int64)
