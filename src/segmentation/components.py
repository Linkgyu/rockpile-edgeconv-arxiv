"""Connected components from predicted edge probabilities."""

from __future__ import annotations

import numpy as np


def components_from_edge_probabilities(n_points: int, edges: np.ndarray, probabilities: np.ndarray, threshold: float, min_cluster_points: int = 10) -> np.ndarray:
    parent = np.arange(n_points, dtype=np.int64)
    rank = np.zeros(n_points, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, b in edges[np.asarray(probabilities) >= threshold]:
        union(int(a), int(b))
    roots = np.array([find(i) for i in range(n_points)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    counts = np.bincount(labels)
    labels[counts[labels] < min_cluster_points] = -1
    out = np.full(n_points, -1, dtype=int)
    valid = labels >= 0
    if np.any(valid):
        _, inv = np.unique(labels[valid], return_inverse=True)
        out[valid] = inv
    return out
