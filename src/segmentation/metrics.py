"""Segmentation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_scores(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, float]:
    true = np.asarray(true_labels)
    pred = np.asarray(predicted_labels)
    return {
        "adjusted_rand_index": float(adjusted_rand_score(true, pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true, pred)),
        "n_true_fragments": int(len(np.unique(true))),
        "n_predicted_clusters": int(len(np.unique(pred[pred >= 0]))),
        "noise_fraction": float(np.mean(pred < 0)),
    }
