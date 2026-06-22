from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.scene_dataset import load_scene_index  # noqa: E402
from src.fragmentation.psd import cumulative_psd, percentile_size  # noqa: E402
from src.fragmentation.surface_proxy import estimate_surface_proxy  # noqa: E402
from src.models.edgeconv import EdgeAffinityDGCNN  # noqa: E402
from src.segmentation.components import components_from_edge_mask, components_from_edge_probabilities  # noqa: E402
from src.segmentation.metrics import clustering_scores  # noqa: E402
from src.segmentation.postprocess import absorb_unlabelled_points_by_edge_affinity, split_oversized_clusters_by_height_markers  # noqa: E402
from src.training.edgeconv_train import edge_metrics_on_scene, scene_edge_probabilities, train_one_scene_step  # noqa: E402


OUT_MODELS = ROOT / "outputs" / "models"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"


POSTPROCESS_KWARGS = {
    "max_cluster_points": 240,
    "grid_resolution_m": 0.030,
    "min_peak_distance_m": 0.10,
    "peak_prominence_m": 0.012,
    "min_child_points": 14,
    "max_markers_per_cluster": 12,
}

ABSORB_KWARGS = {
    "threshold_offset": 0.035,
    "min_absorb_threshold": 0.88,
    "max_passes": 3,
}

HYBRID_BRIDGE_DEFAULTS = {
    "dist_scale": 0.070,
    "z_scale": 0.040,
    "curv_scale": 0.050,
    "graph_threshold": 0.85,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def p80_from_labels(points_xyz: np.ndarray, labels: np.ndarray) -> float:
    sizes = estimate_surface_proxy(points_xyz, labels, min_points=10)
    if sizes.empty:
        return float("nan")
    psd = cumulative_psd(sizes["diameter_proxy_m"].to_numpy(), sizes["proxy_volume_m3"].to_numpy())
    return percentile_size(psd, 80.0)


def graph_geometry_score(scene: dict, dist_scale: float, z_scale: float, curv_scale: float) -> np.ndarray:
    """Geometry-only continuity score used as a conservative bridge signal."""

    features = np.asarray(scene["edge_features"], dtype=float)
    dist = features[:, 6]
    z_jump = features[:, 8]
    normal_agreement = np.abs(features[:, 9])
    normal_angle = features[:, 10]
    curv_delta = features[:, 13]
    return (
        0.30 * np.exp(-dist / dist_scale)
        + 0.25 * normal_agreement
        + 0.20 * np.exp(-z_jump / z_scale)
        + 0.15 * np.exp(-curv_delta / curv_scale)
        + 0.10 * np.exp(-normal_angle / 0.18)
    )


def hybrid_bridge_labels(
    scene: dict,
    probabilities: np.ndarray,
    threshold: float,
    bridge_probability: float,
    graph_threshold: float,
    dist_scale: float = HYBRID_BRIDGE_DEFAULTS["dist_scale"],
    z_scale: float = HYBRID_BRIDGE_DEFAULTS["z_scale"],
    curv_scale: float = HYBRID_BRIDGE_DEFAULTS["curv_scale"],
    min_cluster_points: int = 10,
) -> np.ndarray:
    """Join high-probability EdgeConv edges plus conservative geometry bridges.

    The bridge is deliberately restricted to edges that are both plausible under
    the learned model and highly continuous under the graph baseline geometry
    score.  It targets the observed failure mode where a strict EdgeConv
    threshold breaks true fragments into many small components.
    """

    graph_score = graph_geometry_score(scene, dist_scale, z_scale, curv_scale)
    keep = (np.asarray(probabilities) >= threshold) | (
        (np.asarray(probabilities) >= bridge_probability) & (graph_score >= graph_threshold)
    )
    return components_from_edge_mask(
        len(scene["points_xyz"]),
        scene["edges"],
        keep,
        min_cluster_points=min_cluster_points,
    )


def evaluate_scene_predictions(
    scene: dict,
    probabilities: np.ndarray,
    threshold: float,
    bridge_probability: float | None = None,
    graph_threshold: float | None = None,
) -> list[dict]:
    true_labels = scene["instance_labels"]
    points = scene["points_xyz"]
    true_p80 = float(scene["ground_truth_P80_mm"][0])
    edges = scene["edges"]

    raw_labels = components_from_edge_probabilities(len(points), edges, probabilities, threshold=threshold, min_cluster_points=10)
    absorb_threshold = max(ABSORB_KWARGS["min_absorb_threshold"], threshold - ABSORB_KWARGS["threshold_offset"])
    absorbed_labels = absorb_unlabelled_points_by_edge_affinity(
        raw_labels,
        edges,
        probabilities,
        absorb_threshold=absorb_threshold,
        max_passes=ABSORB_KWARGS["max_passes"],
    )
    post_labels = split_oversized_clusters_by_height_markers(points, raw_labels, **POSTPROCESS_KWARGS)
    absorbed_post_labels = split_oversized_clusters_by_height_markers(points, absorbed_labels, **POSTPROCESS_KWARGS)
    variants = [
        ("edgeconv", raw_labels),
        ("edgeconv_absorb", absorbed_labels),
        ("edgeconv_post_split", post_labels),
        ("edgeconv_absorb_post_split", absorbed_post_labels),
    ]
    if bridge_probability is not None and graph_threshold is not None:
        bridge_labels = hybrid_bridge_labels(
            scene,
            probabilities,
            threshold=threshold,
            bridge_probability=bridge_probability,
            graph_threshold=graph_threshold,
        )
        bridge_absorb_threshold = max(ABSORB_KWARGS["min_absorb_threshold"], bridge_probability)
        bridge_absorbed_labels = absorb_unlabelled_points_by_edge_affinity(
            bridge_labels,
            edges,
            probabilities,
            absorb_threshold=bridge_absorb_threshold,
            max_passes=ABSORB_KWARGS["max_passes"],
        )
        bridge_post_labels = split_oversized_clusters_by_height_markers(points, bridge_labels, **POSTPROCESS_KWARGS)
        bridge_absorbed_post_labels = split_oversized_clusters_by_height_markers(points, bridge_absorbed_labels, **POSTPROCESS_KWARGS)
        variants.extend(
            [
                ("edgeconv_hybrid_bridge", bridge_labels),
                ("edgeconv_hybrid_bridge_absorb", bridge_absorbed_labels),
                ("edgeconv_hybrid_bridge_post_split", bridge_post_labels),
                ("edgeconv_hybrid_bridge_absorb_post_split", bridge_absorbed_post_labels),
            ]
        )

    rows = []
    for variant, labels in variants:
        scores = clustering_scores(true_labels, labels)
        pred_p80 = p80_from_labels(points, labels)
        rows.append(
            {
                "variant": variant,
                "threshold": float(threshold),
                "bridge_probability": float(bridge_probability) if bridge_probability is not None else float("nan"),
                "graph_threshold": float(graph_threshold) if graph_threshold is not None else float("nan"),
                "predicted_P80_mm": pred_p80,
                "ground_truth_P80_mm": true_p80,
                "abs_P80_error_mm": abs(pred_p80 - true_p80) if np.isfinite(pred_p80) else float("nan"),
                "abs_P80_error_pct": abs(pred_p80 - true_p80) / true_p80 * 100.0 if np.isfinite(pred_p80) else float("nan"),
                **scores,
            }
        )
    return rows


def train_model(args: argparse.Namespace, device: torch.device, train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> pd.DataFrame:
    model = EdgeAffinityDGCNN(point_channels=7, edge_attr_channels=15, hidden=48, emb=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = []
    best_val_ap = -np.inf
    best_epoch = -1
    best_state = None
    stale_epochs = 0

    for epoch in range(1, args.max_epochs + 1):
        train_losses = []
        shuffled = train_rows.sample(frac=1.0, random_state=args.seed + epoch).reset_index(drop=True)
        for row_idx, row in shuffled.iterrows():
            loss = train_one_scene_step(
                model,
                optimizer,
                criterion,
                row,
                device=device,
                max_edges=args.max_train_edges,
                seed=args.seed * 1000 + epoch * 100 + row_idx,
                realism_strength=args.photogrammetry_realism,
            )
            train_losses.append(loss)

        val_metrics = []
        for row_idx, row in val_rows.head(args.val_metric_scenes).reset_index(drop=True).iterrows():
            val_metrics.append(
                edge_metrics_on_scene(
                    model,
                    row,
                    device=device,
                    max_edges=args.max_val_edges,
                    seed=args.seed * 2000 + epoch * 100 + row_idx,
                )
            )
        val_ap = float(np.mean([m["average_precision"] for m in val_metrics]))
        val_auc = float(np.mean([m["roc_auc"] for m in val_metrics]))
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_average_precision": val_ap, "val_roc_auc": val_auc})
        print(f"epoch {epoch:02d}: train_loss={train_loss:.4f}, val_AP={val_ap:.4f}, val_AUC={val_auc:.4f}", flush=True)

        if val_ap > best_val_ap + args.min_delta:
            best_val_ap = val_ap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}; best epoch={best_epoch}, best val_AP={best_val_ap:.4f}", flush=True)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = int(history[-1]["epoch"])
        best_val_ap = float(history[-1]["val_average_precision"])

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    training_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_ap": best_val_ap,
            "scan_filter_version": "angular_nearest_plus_xy_height_envelope",
            "training_args": training_args,
        },
        OUT_MODELS / "edgeconv_affinity.pt",
    )
    model.load_state_dict(best_state)
    args._trained_model = model
    return pd.DataFrame(history)


def validation_threshold_sweep(model: EdgeAffinityDGCNN, rows: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    thresholds = np.unique(
        np.r_[
            np.linspace(0.50, 0.95, 16),
            np.linspace(0.96, 0.995, 8),
            np.linspace(0.996, 0.999, 4),
            [0.9992, 0.9995, 0.9997, 0.9999],
        ]
    )
    all_rows = []
    for scene_idx, row in rows.reset_index(drop=True).iterrows():
        scene, probs = scene_edge_probabilities(model, row, device=device)
        for threshold in thresholds:
            for result in evaluate_scene_predictions(scene, probs, float(threshold)):
                result["scene_id"] = int(row["scene_id"])
                result["split"] = row["split"]
                all_rows.append(result)
        bridge_thresholds = np.unique(np.r_[np.linspace(0.985, 0.999, 8), [0.997, 0.9992, 0.9995, 0.9997, 0.9999]])
        bridge_probabilities = [0.50, 0.70, 0.85, 0.95]
        graph_thresholds = [0.80, 0.85, 0.90]
        for threshold in bridge_thresholds:
            for bridge_probability in bridge_probabilities:
                if bridge_probability >= threshold:
                    continue
                for graph_threshold in graph_thresholds:
                    for result in evaluate_scene_predictions(
                        scene,
                        probs,
                        float(threshold),
                        bridge_probability=float(bridge_probability),
                        graph_threshold=float(graph_threshold),
                    ):
                        if not str(result["variant"]).startswith("edgeconv_hybrid_bridge"):
                            continue
                        result["scene_id"] = int(row["scene_id"])
                        result["split"] = row["split"]
                        all_rows.append(result)
        print(f"validated scene {scene_idx + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(all_rows)


def choose_validation_setting(sweep: pd.DataFrame, noise_penalty: float = 0.0, max_noise_fraction: float = 1.0) -> tuple[dict, pd.DataFrame]:
    summary = (
        sweep.groupby(["variant", "threshold", "bridge_probability", "graph_threshold"], as_index=False, dropna=False)
        .agg(
            mean_abs_P80_error_pct=("abs_P80_error_pct", "mean"),
            median_abs_P80_error_pct=("abs_P80_error_pct", "median"),
            mean_abs_P80_error_mm=("abs_P80_error_mm", "mean"),
            mean_NMI=("normalized_mutual_info", "mean"),
            mean_ARI=("adjusted_rand_index", "mean"),
            mean_noise_fraction=("noise_fraction", "mean"),
        )
        .sort_values(["mean_abs_P80_error_pct", "median_abs_P80_error_pct", "mean_noise_fraction"])
        .reset_index(drop=True)
    )
    summary["selection_score"] = summary["mean_abs_P80_error_pct"] + noise_penalty * summary["mean_noise_fraction"] * 100.0
    eligible = summary[summary["mean_noise_fraction"] <= max_noise_fraction].copy()
    if eligible.empty:
        eligible = summary.copy()
    eligible = eligible.sort_values(["selection_score", "mean_abs_P80_error_pct", "median_abs_P80_error_pct"]).reset_index(drop=True)
    summary = summary.sort_values(["selection_score", "mean_abs_P80_error_pct", "median_abs_P80_error_pct"]).reset_index(drop=True)
    best = eligible.iloc[0]
    return best.to_dict(), summary


def evaluate_test_split(
    model: EdgeAffinityDGCNN,
    rows: pd.DataFrame,
    device: torch.device,
    selected_variant: str,
    selected_threshold: float,
    selected_bridge_probability: float | None = None,
    selected_graph_threshold: float | None = None,
) -> pd.DataFrame:
    out_rows = []
    for scene_idx, row in rows.reset_index(drop=True).iterrows():
        scene, probs = scene_edge_probabilities(model, row, device=device)
        evaluated = evaluate_scene_predictions(
            scene,
            probs,
            selected_threshold,
            bridge_probability=selected_bridge_probability,
            graph_threshold=selected_graph_threshold,
        )
        for result in evaluated:
            result["scene_id"] = int(row["scene_id"])
            result["split"] = row["split"]
            out_rows.append(result)

            if result["variant"] == selected_variant:
                labels = labels_for_selected_variant(scene, probs, selected_variant, selected_threshold, selected_bridge_probability, selected_graph_threshold)
                np.savez_compressed(
                    OUT_MODELS / f"scene_{int(row['scene_id']):03d}_{selected_variant}_predictions.npz",
                    edge_probabilities=probs.astype(np.float32),
                    predicted_labels=labels.astype(np.int32),
                    threshold=np.array([selected_threshold], dtype=np.float32),
                    ground_truth_labels=scene["instance_labels"].astype(np.int32),
                    points_xyz=scene["points_xyz"].astype(np.float32),
                )
        print(f"tested scene {scene_idx + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(out_rows)


def labels_for_selected_variant(
    scene: dict,
    probs: np.ndarray,
    selected_variant: str,
    selected_threshold: float,
    selected_bridge_probability: float | None = None,
    selected_graph_threshold: float | None = None,
) -> np.ndarray:
    if selected_variant.startswith("edgeconv_hybrid_bridge"):
        if selected_bridge_probability is None or selected_graph_threshold is None:
            raise ValueError("Hybrid bridge variant requires bridge_probability and graph_threshold")
        labels = hybrid_bridge_labels(
            scene,
            probs,
            threshold=selected_threshold,
            bridge_probability=selected_bridge_probability,
            graph_threshold=selected_graph_threshold,
        )
        if selected_variant in {"edgeconv_hybrid_bridge_absorb", "edgeconv_hybrid_bridge_absorb_post_split"}:
            labels = absorb_unlabelled_points_by_edge_affinity(
                labels,
                scene["edges"],
                probs,
                absorb_threshold=max(ABSORB_KWARGS["min_absorb_threshold"], selected_bridge_probability),
                max_passes=ABSORB_KWARGS["max_passes"],
            )
        if selected_variant in {"edgeconv_hybrid_bridge_post_split", "edgeconv_hybrid_bridge_absorb_post_split"}:
            labels = split_oversized_clusters_by_height_markers(scene["points_xyz"], labels, **POSTPROCESS_KWARGS)
        return labels

    raw_labels = components_from_edge_probabilities(
        len(scene["points_xyz"]),
        scene["edges"],
        probs,
        threshold=selected_threshold,
        min_cluster_points=10,
    )
    labels = raw_labels
    if selected_variant in {"edgeconv_absorb", "edgeconv_absorb_post_split"}:
        labels = absorb_unlabelled_points_by_edge_affinity(
            raw_labels,
            scene["edges"],
            probs,
            absorb_threshold=max(ABSORB_KWARGS["min_absorb_threshold"], selected_threshold - ABSORB_KWARGS["threshold_offset"]),
            max_passes=ABSORB_KWARGS["max_passes"],
        )
    if selected_variant == "edgeconv_post_split":
        labels = split_oversized_clusters_by_height_markers(scene["points_xyz"], raw_labels, **POSTPROCESS_KWARGS)
    elif selected_variant == "edgeconv_absorb_post_split":
        labels = split_oversized_clusters_by_height_markers(scene["points_xyz"], labels, **POSTPROCESS_KWARGS)
    return labels


def write_training_curve(history: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax1.plot(history["epoch"], history["train_loss"], marker="o", color="#244C72", label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE loss")
    ax1.grid(True, color="#D8DEE8", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_average_precision"], marker="s", color="#C8643B", label="Validation AP")
    ax2.set_ylabel("Validation AP")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "02_edgeconv_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def write_test_error_histogram(test_results: pd.DataFrame, selected_variant: str) -> None:
    selected = test_results[test_results["variant"] == selected_variant]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    ax.hist(selected["abs_P80_error_pct"].dropna(), bins=np.linspace(0, max(5.0, selected["abs_P80_error_pct"].max() + 2.0), 9), color="#2F6B7A", edgecolor="white")
    ax.set_xlabel("Absolute P80 error [%]")
    ax.set_ylabel("Number of test scenes")
    ax.grid(True, axis="y", color="#D8DEE8", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "03_edgeconv_test_p80_error_histogram.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain EdgeConv on surface-only synthetic rockpile scans and evaluate PSD.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1.0e-3)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-edges", type=int, default=22000)
    parser.add_argument("--max-val-edges", type=int, default=36000)
    parser.add_argument("--val-metric-scenes", type=int, default=12)
    parser.add_argument(
        "--scene-index",
        type=Path,
        default=None,
        help="Optional scene index CSV. Defaults to the baseline multi_scene_index.csv in dnn-rockpile-affinity-psd.",
    )
    parser.add_argument(
        "--noise-penalty",
        type=float,
        default=0.0,
        help="Validation selection penalty added as penalty * mean_noise_fraction * 100.",
    )
    parser.add_argument(
        "--max-noise-fraction",
        type=float,
        default=1.0,
        help="Prefer validation settings at or below this mean noise fraction when possible.",
    )
    parser.add_argument(
        "--photogrammetry-realism",
        type=float,
        default=0.75,
        help="Training-time SfM/MVS realism augmentation strength. Set 0 to disable.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional output namespace under outputs/runs/<run-name> so ablations do not overwrite the main benchmark.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.run_name:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.run_name).strip("_")
        if not safe_name:
            raise SystemExit("--run-name did not contain any usable characters")
        global OUT_MODELS, OUT_TABLES, OUT_FIGURES
        run_root = ROOT / "outputs" / "runs" / safe_name
        OUT_MODELS = run_root / "models"
        OUT_TABLES = run_root / "tables"
        OUT_FIGURES = run_root / "figures"
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    index = load_scene_index(args.scene_index)
    if "scan_filter_version" not in index.columns or not index["scan_filter_version"].eq("angular_nearest_plus_xy_height_envelope").all():
        raise SystemExit("Scene index is not the regenerated surface-envelope dataset. Regenerate it before retraining.")

    train_rows = index[index["split"] == "train"].reset_index(drop=True)
    val_rows = index[index["split"] == "val"].reset_index(drop=True)
    test_rows = index[index["split"] == "test"].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}", flush=True)

    history = train_model(args, device, train_rows, val_rows)
    history.to_csv(OUT_TABLES / "edgeconv_training_history.csv", index=False)
    write_training_curve(history)

    model = args._trained_model
    sweep = validation_threshold_sweep(model, val_rows, device)
    sweep.to_csv(OUT_TABLES / "edgeconv_validation_threshold_sweep.csv", index=False)
    selected_setting, validation_summary = choose_validation_setting(
        sweep,
        noise_penalty=args.noise_penalty,
        max_noise_fraction=args.max_noise_fraction,
    )
    selected_variant = str(selected_setting["variant"])
    selected_threshold = float(selected_setting["threshold"])
    selected_bridge_probability = selected_setting.get("bridge_probability")
    selected_graph_threshold = selected_setting.get("graph_threshold")
    selected_bridge_probability = None if pd.isna(selected_bridge_probability) else float(selected_bridge_probability)
    selected_graph_threshold = None if pd.isna(selected_graph_threshold) else float(selected_graph_threshold)
    validation_summary.to_csv(OUT_TABLES / "edgeconv_validation_threshold_summary.csv", index=False)
    print(
        "selected validation setting: "
        f"variant={selected_variant}, threshold={selected_threshold:.4f}, "
        f"bridge_probability={selected_bridge_probability}, graph_threshold={selected_graph_threshold}",
        flush=True,
    )

    test_results = evaluate_test_split(
        model,
        test_rows,
        device,
        selected_variant,
        selected_threshold,
        selected_bridge_probability=selected_bridge_probability,
        selected_graph_threshold=selected_graph_threshold,
    )
    test_results.to_csv(OUT_TABLES / "edgeconv_test_results.csv", index=False)

    test_summary = (
        test_results.groupby("variant", as_index=False)
        .agg(
            n_scenes=("scene_id", "count"),
            threshold=("threshold", "first"),
            bridge_probability=("bridge_probability", "first"),
            graph_threshold=("graph_threshold", "first"),
            mean_abs_P80_error_pct=("abs_P80_error_pct", "mean"),
            median_abs_P80_error_pct=("abs_P80_error_pct", "median"),
            mean_abs_P80_error_mm=("abs_P80_error_mm", "mean"),
            mean_NMI=("normalized_mutual_info", "mean"),
            mean_ARI=("adjusted_rand_index", "mean"),
            mean_noise_fraction=("noise_fraction", "mean"),
        )
        .sort_values("mean_abs_P80_error_pct")
        .reset_index(drop=True)
    )
    test_summary["selected_for_transfer"] = test_summary["variant"].eq(selected_variant)
    test_summary.to_csv(OUT_TABLES / "edgeconv_test_summary.csv", index=False)
    test_summary.to_csv(OUT_TABLES / "edgeconv_raw_vs_postprocess_test_summary.csv", index=False)
    write_test_error_histogram(test_results, selected_variant)

    metadata = {
        "scan_filter_version": "angular_nearest_plus_xy_height_envelope",
        "selected_variant": selected_variant,
        "selected_threshold": selected_threshold,
        "selected_bridge_probability": selected_bridge_probability,
        "selected_graph_threshold": selected_graph_threshold,
        "best_epoch": int(history.loc[history["val_average_precision"].idxmax(), "epoch"]),
        "best_val_average_precision": float(history["val_average_precision"].max()),
        "postprocess_kwargs": POSTPROCESS_KWARGS,
        "absorb_kwargs": ABSORB_KWARGS,
        "photogrammetry_realism_strength": float(args.photogrammetry_realism),
        "scene_index": str(args.scene_index) if args.scene_index else "default",
        "noise_penalty": float(args.noise_penalty),
        "max_noise_fraction": float(args.max_noise_fraction),
    }
    (OUT_TABLES / "edgeconv_retraining_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(test_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
