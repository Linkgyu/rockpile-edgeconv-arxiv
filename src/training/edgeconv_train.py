"""Training and inference helpers for EdgeConv affinity models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.photogrammetry_augment import apply_photogrammetry_realism
from src.data.scene_dataset import balanced_edge_indices, load_npz_scene, scene_to_torch


def train_one_scene_step(
    model,
    optimizer,
    criterion,
    row,
    device,
    max_edges: int,
    seed: int,
    realism_strength: float = 0.0,
) -> float:
    scene = load_npz_scene(row)
    if realism_strength > 0:
        scene = apply_photogrammetry_realism(scene, seed=seed, strength=realism_strength)
    batch = scene_to_torch(scene, device=device)
    idx_np = balanced_edge_indices(scene["edge_same_fragment"], max_edges=max_edges, seed=seed)
    idx = torch.from_numpy(idx_np).long().to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["x"], batch["edge_index"], batch["edges"][idx], batch["edge_attr"][idx])
    loss = criterion(logits, batch["edge_y"][idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def scene_edge_probabilities(model, row, device, chunk_edges: int = 65536) -> tuple[dict, np.ndarray]:
    scene = load_npz_scene(row)
    batch = scene_to_torch(scene, device=device)
    model.eval()
    h = model.encode(batch["x"], batch["edge_index"])
    probs = []
    n_edges = batch["edges"].shape[0]
    for start in range(0, n_edges, chunk_edges):
        end = min(start + chunk_edges, n_edges)
        e = batch["edges"][start:end]
        attr = batch["edge_attr"][start:end]
        src = e[:, 0]
        dst = e[:, 1]
        pair = torch.cat([h[src], h[dst], torch.abs(h[src] - h[dst]), attr], dim=1)
        logits = model.edge_head(pair).squeeze(1)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return scene, np.concatenate(probs)


@torch.no_grad()
def edge_metrics_on_scene(model, row, device, max_edges: int = 40000, seed: int = 0) -> dict[str, float]:
    scene = load_npz_scene(row)
    batch = scene_to_torch(scene, device=device)
    idx_np = balanced_edge_indices(scene["edge_same_fragment"], max_edges=max_edges, seed=seed)
    idx = torch.from_numpy(idx_np).long().to(device)
    model.eval()
    logits = model(batch["x"], batch["edge_index"], batch["edges"][idx], batch["edge_attr"][idx])
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    y = scene["edge_same_fragment"][idx_np].astype(bool)
    out = {"average_precision": float(average_precision_score(y, prob))}
    try:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out
