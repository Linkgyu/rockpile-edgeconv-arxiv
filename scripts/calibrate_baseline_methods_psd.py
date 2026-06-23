from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_evaluate_baselines_150frag import (  # noqa: E402
    evaluate_dbscan,
    evaluate_graph_threshold,
    evaluate_mlp_thresholds,
    evaluate_region_growing,
    load_scene_index,
    set_seed,
    summarize,
    train_mlp_affinity,
)


TABLES = ROOT / "results" / "tables"


def parse_mlp(setting: str) -> float:
    return float(setting.split("=")[1])


def parse_dbscan(setting: str) -> tuple[float, int, float]:
    m = re.fullmatch(r"eps=([0-9.]+),min=([0-9]+),zw=([0-9.]+)", setting)
    if not m:
        raise ValueError(setting)
    return float(m.group(1)), int(m.group(2)), float(m.group(3))


def parse_graph(setting: str) -> tuple[float, float, float, float]:
    m = re.fullmatch(r"d=([0-9.]+),z=([0-9.]+),c=([0-9.]+),thr=([0-9.]+)", setting)
    if not m:
        raise ValueError(setting)
    return tuple(float(m.group(i)) for i in range(1, 5))


def parse_region(setting: str) -> tuple[float, float, float, float]:
    m = re.fullmatch(r"r=([0-9.]+),a=([0-9.]+),c=([0-9.]+),z=([0-9.]+)", setting)
    if not m:
        raise ValueError(setting)
    return tuple(float(m.group(i)) for i in range(1, 5))


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(x[ok], y[ok], deg=1)
    return float(a), float(b)


def error_vector(pred: pd.Series, true: pd.Series) -> np.ndarray:
    return ((pred.to_numpy(float) - true.to_numpy(float)) / true.to_numpy(float) * 100.0)


def bootstrap_mean_abs_ci(errors_pct: np.ndarray, seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    ok = np.asarray(errors_pct, dtype=float)
    ok = ok[np.isfinite(ok)]
    if ok.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(np.abs(ok), size=(n_boot, ok.size), replace=True).mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]).astype(float))


def summarize_calibrated(val: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    a, b = fit_linear(val["predicted_P80_mm"].to_numpy(float), val["ground_truth_P80_mm"].to_numpy(float))
    out = test.copy()
    out["calibrated_P80_mm"] = a * out["predicted_P80_mm"] + b
    raw_err = error_vector(out["predicted_P80_mm"], out["ground_truth_P80_mm"])
    cal_err = error_vector(out["calibrated_P80_mm"], out["ground_truth_P80_mm"])
    raw_lo, raw_hi = bootstrap_mean_abs_ci(raw_err, seed=seed)
    cal_lo, cal_hi = bootstrap_mean_abs_ci(cal_err, seed=seed + 1)
    first = out.iloc[0]
    return {
        "method": first["method"],
        "setting": first["setting"],
        "n_val_scenes": int(val["scene_id"].nunique()),
        "n_test_scenes": int(out["scene_id"].nunique()),
        "calibration_slope": a,
        "calibration_intercept_mm": b,
        "raw_full_mean_abs_P80_error_pct": float(np.nanmean(np.abs(raw_err))),
        "raw_full_mean_abs_P80_error_ci95_low": raw_lo,
        "raw_full_mean_abs_P80_error_ci95_high": raw_hi,
        "calibrated_full_mean_abs_P80_error_pct": float(np.nanmean(np.abs(cal_err))),
        "calibrated_full_mean_abs_P80_error_ci95_low": cal_lo,
        "calibrated_full_mean_abs_P80_error_ci95_high": cal_hi,
        "calibrated_full_mean_signed_error_pct": float(np.nanmean(cal_err)),
        "mean_NMI": float(out["normalized_mutual_info"].mean()),
        "mean_ARI": float(out["adjusted_rand_index"].mean()),
        "mean_noise_fraction": float(out["noise_fraction"].mean()),
        "mean_predicted_clusters": float(out["n_predicted_clusters"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected baseline settings and fit exterior-to-full P80 calibration.")
    parser.add_argument("--scene-index", type=Path, default=TABLES / "dem_noboundary_relax150_100scene_hpr_index.csv")
    parser.add_argument("--selected-settings", type=Path, default=TABLES / "baseline_150frag_selected_settings.csv")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-train-edges", type=int, default=22000)
    parser.add_argument("--max-val-edges", type=int, default=36000)
    parser.add_argument("--val-metric-scenes", type=int, default=12)
    parser.add_argument("--photogrammetry-realism", type=float, default=0.75)
    args = parser.parse_args()

    set_seed(args.seed)
    TABLES.mkdir(parents=True, exist_ok=True)
    index = load_scene_index(args.scene_index)
    train_rows = index[index["split"] == "train"].reset_index(drop=True)
    val_rows = index[index["split"] == "val"].reset_index(drop=True)
    test_rows = index[index["split"] == "test"].reset_index(drop=True)
    selected = pd.read_csv(args.selected_settings)

    val_parts = []
    test_parts = []

    mlp_setting = selected[selected["method"].eq("MLP affinity")]["setting"].iloc[0]
    mlp, _history = train_mlp_affinity(train_rows, val_rows, args)
    mlp_thr = parse_mlp(mlp_setting)
    mlp_val = evaluate_mlp_thresholds(mlp, val_rows, "val", np.array([mlp_thr]))
    mlp_test = evaluate_mlp_thresholds(mlp, test_rows, "test", np.array([mlp_thr]))
    mlp_val["setting"] = mlp_setting
    mlp_test["setting"] = mlp_setting
    val_parts.append(mlp_val)
    test_parts.append(mlp_test)

    dbscan_setting = selected[selected["method"].eq("DBSCAN")]["setting"].iloc[0]
    val_parts.append(evaluate_dbscan(val_rows, "val", [parse_dbscan(dbscan_setting)]))
    test_parts.append(evaluate_dbscan(test_rows, "test", [parse_dbscan(dbscan_setting)]))

    graph_setting = selected[selected["method"].eq("Graph threshold")]["setting"].iloc[0]
    val_parts.append(evaluate_graph_threshold(val_rows, "val", [parse_graph(graph_setting)]))
    test_parts.append(evaluate_graph_threshold(test_rows, "test", [parse_graph(graph_setting)]))

    region_setting = selected[selected["method"].eq("Region growing")]["setting"].iloc[0]
    val_parts.append(evaluate_region_growing(val_rows, "val", [parse_region(region_setting)]))
    test_parts.append(evaluate_region_growing(test_rows, "test", [parse_region(region_setting)]))

    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    val.to_csv(TABLES / "baseline_150frag_selected_validation_scene_results.csv", index=False)
    test.to_csv(TABLES / "baseline_150frag_selected_test_scene_results.csv", index=False)
    summarize(val).to_csv(TABLES / "baseline_150frag_selected_validation_summary.csv", index=False)
    summarize(test).to_csv(TABLES / "baseline_150frag_selected_test_summary.csv", index=False)

    rows = []
    for (method, setting), test_group in test.groupby(["method", "setting"], sort=False):
        val_group = val[val["method"].eq(method) & val["setting"].eq(setting)]
        rows.append(summarize_calibrated(val_group, test_group, seed=args.seed + len(rows) * 17))
    out = pd.DataFrame(rows).sort_values("calibrated_full_mean_abs_P80_error_pct").reset_index(drop=True)
    out.to_csv(TABLES / "baseline_150frag_calibrated_test_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
