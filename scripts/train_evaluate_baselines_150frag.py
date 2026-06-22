from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.photogrammetry_augment import apply_photogrammetry_realism  # noqa: E402
from src.data.scene_dataset import balanced_edge_indices, edge_geom_features, load_npz_scene, load_scene_index  # noqa: E402
from src.fragmentation.psd import cumulative_psd, percentile_size  # noqa: E402
from src.fragmentation.surface_proxy import estimate_surface_proxy  # noqa: E402
from src.segmentation.components import components_from_edge_probabilities  # noqa: E402
from src.segmentation.metrics import clustering_scores  # noqa: E402


OUT_TABLES = ROOT / "results" / "tables"
OUT_FIGURES = ROOT / "results" / "figures"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def p80_from_labels(points_xyz: np.ndarray, labels: np.ndarray) -> float:
    sizes = estimate_surface_proxy(points_xyz, labels, min_points=10)
    if sizes.empty:
        return float("nan")
    psd = cumulative_psd(sizes["diameter_proxy_m"].to_numpy(), sizes["proxy_volume_m3"].to_numpy())
    return percentile_size(psd, 80.0)


def evaluate_labels(scene: dict, labels: np.ndarray, method: str, setting: str, split: str) -> dict:
    true_p80 = float(scene["ground_truth_P80_mm"][0])
    pred_p80 = p80_from_labels(scene["points_xyz"], labels)
    out = {
        "method": method,
        "setting": setting,
        "split": split,
        "scene_id": int(scene["scene_id"][0]),
        "predicted_P80_mm": pred_p80,
        "ground_truth_P80_mm": true_p80,
        "signed_P80_error_pct": (pred_p80 - true_p80) / true_p80 * 100.0 if np.isfinite(pred_p80) else float("nan"),
        "abs_P80_error_pct": abs(pred_p80 - true_p80) / true_p80 * 100.0 if np.isfinite(pred_p80) else float("nan"),
        "abs_P80_error_mm": abs(pred_p80 - true_p80) if np.isfinite(pred_p80) else float("nan"),
    }
    out.update(clustering_scores(scene["instance_labels"], labels))
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["method", "setting"], as_index=False)
        .agg(
            n_scenes=("scene_id", "count"),
            bias_P80_error_pct=("signed_P80_error_pct", "mean"),
            mean_abs_P80_error_pct=("abs_P80_error_pct", "mean"),
            std_abs_P80_error_pct=("abs_P80_error_pct", "std"),
            median_abs_P80_error_pct=("abs_P80_error_pct", "median"),
            max_abs_P80_error_pct=("abs_P80_error_pct", "max"),
            mean_abs_P80_error_mm=("abs_P80_error_mm", "mean"),
            mean_NMI=("normalized_mutual_info", "mean"),
            mean_ARI=("adjusted_rand_index", "mean"),
            mean_noise_fraction=("noise_fraction", "mean"),
            mean_predicted_clusters=("n_predicted_clusters", "mean"),
        )
        .sort_values(["mean_abs_P80_error_pct", "mean_noise_fraction"])
        .reset_index(drop=True)
    )


def choose_best(validation_rows: pd.DataFrame, noise_penalty: float) -> pd.Series:
    summary = summarize(validation_rows)
    summary["selection_score"] = summary["mean_abs_P80_error_pct"] + noise_penalty * 100.0 * summary["mean_noise_fraction"]
    return summary.sort_values(["selection_score", "mean_abs_P80_error_pct", "median_abs_P80_error_pct"]).iloc[0]


def mlp_edge_features(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    return edge_geom_features(scene["edge_features"]), scene["edge_same_fragment"].astype(int)


def train_mlp_affinity(train_rows: pd.DataFrame, val_rows: pd.DataFrame, args: argparse.Namespace) -> tuple[MLPClassifier, pd.DataFrame]:
    scaler = StandardScaler()
    rng = np.random.default_rng(args.seed)
    print("fitting MLP scaler", flush=True)
    sample_parts = []
    for row_idx, row in train_rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        idx = balanced_edge_indices(scene["edge_same_fragment"], max_edges=min(args.max_train_edges, 9000), seed=args.seed + row_idx)
        x, _ = mlp_edge_features(scene)
        sample_parts.append(x[idx])
    scaler.fit(np.vstack(sample_parts))

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=args.weight_decay,
        batch_size=2048,
        learning_rate_init=args.lr,
        max_iter=1,
        warm_start=True,
        random_state=args.seed,
        shuffle=True,
    )

    history = []
    classes = np.array([0, 1], dtype=int)
    for epoch in range(1, args.epochs + 1):
        losses = []
        shuffled = train_rows.sample(frac=1.0, random_state=args.seed + epoch).reset_index(drop=True)
        for row_idx, row in shuffled.iterrows():
            scene = load_npz_scene(row)
            if args.photogrammetry_realism > 0:
                scene = apply_photogrammetry_realism(scene, seed=args.seed * 1000 + epoch * 100 + row_idx, strength=args.photogrammetry_realism)
            idx = balanced_edge_indices(scene["edge_same_fragment"], max_edges=args.max_train_edges, seed=args.seed * 2000 + epoch * 100 + row_idx)
            x, y = mlp_edge_features(scene)
            clf.partial_fit(scaler.transform(x[idx]), y[idx], classes=classes)
            if hasattr(clf, "loss_"):
                losses.append(float(clf.loss_))

        val_scores = []
        for row_idx, row in val_rows.head(args.val_metric_scenes).reset_index(drop=True).iterrows():
            scene = load_npz_scene(row)
            idx = balanced_edge_indices(scene["edge_same_fragment"], max_edges=args.max_val_edges, seed=args.seed * 3000 + epoch * 100 + row_idx)
            x, y = mlp_edge_features(scene)
            prob = clf.predict_proba(scaler.transform(x[idx]))[:, 1]
            # Average precision without importing another metrics module; rank-based approximation is not needed here.
            val_scores.append(float(np.mean((prob >= 0.5) == y[idx])))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_balanced_accuracy_at_0p5": float(np.mean(val_scores)),
        }
        history.append(row)
        print(f"mlp epoch {epoch:02d}: loss={row['train_loss']:.4f}, val_bal_acc={row['val_balanced_accuracy_at_0p5']:.4f}", flush=True)

    clf._rockpile_scaler = scaler  # persisted in-memory for this run.
    return clf, pd.DataFrame(history)


def mlp_probabilities(clf: MLPClassifier, scene: dict) -> np.ndarray:
    x, _ = mlp_edge_features(scene)
    return clf.predict_proba(clf._rockpile_scaler.transform(x))[:, 1]


def evaluate_mlp_thresholds(clf: MLPClassifier, rows: pd.DataFrame, split: str, thresholds: np.ndarray) -> pd.DataFrame:
    all_rows = []
    for row_idx, row in rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        prob = mlp_probabilities(clf, scene)
        for threshold in thresholds:
            labels = components_from_edge_probabilities(len(scene["points_xyz"]), scene["edges"], prob, threshold=float(threshold), min_cluster_points=10)
            all_rows.append(evaluate_labels(scene, labels, "MLP affinity", f"threshold={threshold:.3f}", split))
        print(f"mlp {split} scene {row_idx + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(all_rows)


def graph_score(scene: dict, dist_scale: float, z_scale: float, curv_scale: float) -> np.ndarray:
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


def evaluate_graph_threshold(rows: pd.DataFrame, split: str, settings: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    all_rows = []
    for row_idx, row in rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        for dist_scale, z_scale, curv_scale, threshold in settings:
            score = graph_score(scene, dist_scale, z_scale, curv_scale)
            labels = components_from_edge_probabilities(len(scene["points_xyz"]), scene["edges"], score, threshold=threshold, min_cluster_points=10)
            setting = f"d={dist_scale:.3f},z={z_scale:.3f},c={curv_scale:.3f},thr={threshold:.2f}"
            all_rows.append(evaluate_labels(scene, labels, "Graph threshold", setting, split))
        print(f"graph {split} scene {row_idx + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(all_rows)


def evaluate_dbscan(rows: pd.DataFrame, split: str, settings: list[tuple[float, int, float]]) -> pd.DataFrame:
    all_rows = []
    for row_idx, row in rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        points = np.asarray(scene["points_xyz"], dtype=float)
        for eps, min_samples, z_weight in settings:
            x = points.copy()
            x[:, 2] *= z_weight
            labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(x)
            setting = f"eps={eps:.3f},min={min_samples},zw={z_weight:.2f}"
            all_rows.append(evaluate_labels(scene, labels, "DBSCAN", setting, split))
        print(f"dbscan {split} scene {row_idx + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(all_rows)


def evaluate_region_growing(rows: pd.DataFrame, split: str, settings: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    all_rows = []
    for row_idx, row in rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        features = np.asarray(scene["edge_features"], dtype=float)
        dist = features[:, 6]
        z_jump = features[:, 8]
        normal_angle = features[:, 10]
        curv_delta = features[:, 13]
        for radius, angle, curv, zmax in settings:
            prob = (
                (dist <= radius)
                & (normal_angle <= angle)
                & (curv_delta <= curv)
                & (z_jump <= zmax)
            ).astype(float)
            labels = components_from_edge_probabilities(len(scene["points_xyz"]), scene["edges"], prob, threshold=0.5, min_cluster_points=10)
            setting = f"r={radius:.3f},a={angle:.3f},c={curv:.3f},z={zmax:.3f}"
            all_rows.append(evaluate_labels(scene, labels, "Region growing", setting, split))
        print(f"region {split} scene {row_idx + 1}/{len(rows)}", flush=True)
    return pd.DataFrame(all_rows)


def evaluate_selected(method_fn, rows: pd.DataFrame, split: str, selected_setting: str, settings: list) -> pd.DataFrame:
    selected = [item for item in settings if item[-1] == selected_setting]
    if not selected:
        raise ValueError(f"Could not find selected setting {selected_setting}")
    return method_fn(rows, split, [selected[0][:-1]])


def plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    ordered = summary.sort_values("mean_abs_P80_error_pct")
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    ax.bar(ordered["method"], ordered["mean_abs_P80_error_pct"], color="#3D6C7C")
    ax.set_ylabel("Mean absolute P80 error (%)")
    ax.set_xlabel("")
    ax.set_title("150-fragment held-out test comparison")
    ax.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(ordered["mean_abs_P80_error_pct"]):
        ax.text(idx, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global OUT_TABLES, OUT_FIGURES

    parser = argparse.ArgumentParser(description="Train/evaluate non-EdgeConv baselines on the latest 150-fragment dataset.")
    parser.add_argument("--scene-index", type=Path, default=OUT_TABLES / "dem_noboundary_relax150_100scene_index.csv")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-edges", type=int, default=22000)
    parser.add_argument("--max-val-edges", type=int, default=36000)
    parser.add_argument("--val-metric-scenes", type=int, default=12)
    parser.add_argument("--photogrammetry-realism", type=float, default=0.75)
    parser.add_argument("--noise-penalty", type=float, default=0.10)
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional output namespace under outputs/runs/<run-name>, matching the EdgeConv run layout.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.run_name:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.run_name).strip("_")
        if not safe_name:
            raise SystemExit("--run-name did not contain any usable characters")
        run_root = ROOT / "outputs" / "runs" / safe_name
        OUT_TABLES = run_root / "tables"
        OUT_FIGURES = run_root / "figures"
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    index = load_scene_index(args.scene_index)
    train_rows = index[index["split"] == "train"].reset_index(drop=True)
    val_rows = index[index["split"] == "val"].reset_index(drop=True)
    test_rows = index[index["split"] == "test"].reset_index(drop=True)
    print(f"split sizes: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}", flush=True)

    mlp, mlp_history = train_mlp_affinity(train_rows, val_rows, args)
    mlp_history.to_csv(OUT_TABLES / "baseline_150frag_mlp_training_history.csv", index=False)
    thresholds = np.unique(np.r_[np.linspace(0.50, 0.95, 16), np.linspace(0.96, 0.999, 8)])
    mlp_val = evaluate_mlp_thresholds(mlp, val_rows, "val", thresholds)
    mlp_best = choose_best(mlp_val, args.noise_penalty)
    mlp_test_all = evaluate_mlp_thresholds(mlp, test_rows, "test", np.array([float(str(mlp_best["setting"]).split("=")[1])]))
    mlp_test_all["setting"] = str(mlp_best["setting"])

    dbscan_settings = [(eps, min_samples, z_weight) for eps in [0.025, 0.030, 0.035, 0.040, 0.050, 0.060] for min_samples in [6, 10, 14] for z_weight in [0.75, 1.00, 1.35]]
    dbscan_val = evaluate_dbscan(val_rows, "val", dbscan_settings)
    dbscan_best = choose_best(dbscan_val, args.noise_penalty)
    dbscan_setting_map = {f"eps={eps:.3f},min={min_samples},zw={z_weight:.2f}": (eps, min_samples, z_weight) for eps, min_samples, z_weight in dbscan_settings}
    dbscan_test = evaluate_dbscan(test_rows, "test", [dbscan_setting_map[str(dbscan_best["setting"])]])

    graph_settings = [(d, z, c, thr) for d in [0.035, 0.050, 0.070] for z in [0.025, 0.040, 0.060] for c in [0.025, 0.050, 0.080] for thr in [0.55, 0.65, 0.75, 0.85]]
    graph_val = evaluate_graph_threshold(val_rows, "val", graph_settings)
    graph_best = choose_best(graph_val, args.noise_penalty)
    graph_setting_map = {f"d={d:.3f},z={z:.3f},c={c:.3f},thr={thr:.2f}": (d, z, c, thr) for d, z, c, thr in graph_settings}
    graph_test = evaluate_graph_threshold(test_rows, "test", [graph_setting_map[str(graph_best["setting"])]])

    region_settings = [(r, a, c, z) for r in [0.035, 0.045, 0.060, 0.080] for a in [0.10, 0.14, 0.18, 0.24] for c in [0.035, 0.060, 0.090] for z in [0.035, 0.055, 0.080]]
    region_val = evaluate_region_growing(val_rows, "val", region_settings)
    region_best = choose_best(region_val, args.noise_penalty)
    region_setting_map = {f"r={r:.3f},a={a:.3f},c={c:.3f},z={z:.3f}": (r, a, c, z) for r, a, c, z in region_settings}
    region_test = evaluate_region_growing(test_rows, "test", [region_setting_map[str(region_best["setting"])]])

    validation_rows = pd.concat([mlp_val, dbscan_val, graph_val, region_val], ignore_index=True)
    validation_summary = summarize(validation_rows)
    validation_summary.to_csv(OUT_TABLES / "baseline_150frag_validation_summary.csv", index=False)
    pd.DataFrame([mlp_best, dbscan_best, graph_best, region_best]).to_csv(OUT_TABLES / "baseline_150frag_selected_settings.csv", index=False)

    test_rows_all = pd.concat([mlp_test_all, dbscan_test, graph_test, region_test], ignore_index=True)
    test_rows_all.to_csv(OUT_TABLES / "baseline_150frag_test_scene_results.csv", index=False)
    test_summary = summarize(test_rows_all)
    test_summary.to_csv(OUT_TABLES / "baseline_150frag_test_summary.csv", index=False)

    edgeconv_path = OUT_TABLES / "edgeconv_test_summary.csv"
    if edgeconv_path.exists():
        edgeconv = pd.read_csv(edgeconv_path)
        selected = edgeconv[edgeconv["selected_for_transfer"].astype(bool)].copy()
        if selected.empty:
            selected = edgeconv.head(1).copy()
        edgeconv_summary = pd.DataFrame(
            [
                {
                    "method": f"EdgeConv {selected['variant'].iloc[0]}",
                    "setting": (
                        f"threshold={float(selected['threshold'].iloc[0]):.4f},"
                        f"bridge={float(selected['bridge_probability'].iloc[0]):.2f},"
                        f"graph={float(selected['graph_threshold'].iloc[0]):.2f}"
                    ),
                    "n_scenes": int(selected["n_scenes"].iloc[0]),
                    "bias_P80_error_pct": float("nan"),
                    "mean_abs_P80_error_pct": float(selected["mean_abs_P80_error_pct"].iloc[0]),
                    "std_abs_P80_error_pct": float("nan"),
                    "median_abs_P80_error_pct": float(selected["median_abs_P80_error_pct"].iloc[0]),
                    "max_abs_P80_error_pct": float("nan"),
                    "mean_abs_P80_error_mm": float(selected["mean_abs_P80_error_mm"].iloc[0]),
                    "mean_NMI": float(selected["mean_NMI"].iloc[0]),
                    "mean_ARI": float(selected["mean_ARI"].iloc[0]),
                    "mean_noise_fraction": float(selected["mean_noise_fraction"].iloc[0]),
                    "mean_predicted_clusters": float("nan"),
                }
            ]
        )
        comparison = pd.concat([edgeconv_summary, test_summary], ignore_index=True)
    else:
        comparison = test_summary
    comparison = comparison.sort_values("mean_abs_P80_error_pct").reset_index(drop=True)
    comparison.to_csv(OUT_TABLES / "model_comparison_150frag_test_summary.csv", index=False)
    plot_comparison(comparison, OUT_FIGURES / "model_comparison_150frag_p80.png")
    print(comparison, flush=True)


if __name__ == "__main__":
    main()
