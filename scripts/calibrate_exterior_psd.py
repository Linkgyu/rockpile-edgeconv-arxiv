from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fragmentation.psd import cumulative_psd, percentile_size  # noqa: E402
from src.fragmentation.surface_proxy import estimate_surface_proxy  # noqa: E402


TABLES = ROOT / "results" / "tables"
RUN_TABLES = ROOT / "outputs" / "runs" / "hpr_multiview_24ep" / "tables"
INDEX_PATH = TABLES / "dem_noboundary_relax150_100scene_hpr_index.csv"


def p80_from_labels(points_xyz: np.ndarray, labels: np.ndarray) -> float:
    sizes = estimate_surface_proxy(points_xyz, labels, min_points=10)
    if sizes.empty:
        return float("nan")
    psd = cumulative_psd(sizes["diameter_proxy_m"].to_numpy(), sizes["proxy_volume_m3"].to_numpy())
    return percentile_size(psd, 80.0)


def write_oracle_exterior_p80() -> pd.DataFrame:
    rows = []
    index = pd.read_csv(INDEX_PATH)
    for _, row in index.iterrows():
        scene = np.load(row["path"])
        oracle = p80_from_labels(scene["points_xyz"], scene["instance_labels"])
        full = float(scene["ground_truth_P80_mm"][0])
        rows.append(
            {
                "scene_id": int(row["scene_id"]),
                "split": row["split"],
                "oracle_exterior_P80_mm": oracle,
                "ground_truth_P80_mm": full,
                "oracle_signed_error_pct_vs_full": (oracle - full) / full * 100.0,
                "oracle_abs_error_pct_vs_full": abs(oracle - full) / full * 100.0,
                "n_visible_fragments": int(len(np.unique(scene["instance_labels"]))),
                "n_exterior_points": int(len(scene["points_xyz"])),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "hpr_oracle_exterior_label_p80.csv", index=False)
    return out


def setting_key(df: pd.DataFrame) -> pd.Series:
    parts = [
        df["variant"].astype(str),
        df["threshold"].map(lambda x: f"{float(x):.4f}"),
        df["bridge_probability"].fillna(-1).map(lambda x: f"{float(x):.4f}"),
        df["graph_threshold"].fillna(-1).map(lambda x: f"{float(x):.4f}"),
    ]
    return parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3]


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(x[ok], y[ok], deg=1)
    return float(a), float(b)


def summarize_errors(values: pd.DataFrame, pred_col: str, true_col: str) -> dict:
    signed = (values[pred_col] - values[true_col]) / values[true_col] * 100.0
    abs_err = signed.abs()
    return {
        "mean_signed_error_pct": float(signed.mean()),
        "mean_abs_error_pct": float(abs_err.mean()),
        "median_abs_error_pct": float(abs_err.median()),
        "mean_abs_error_mm": float((values[pred_col] - values[true_col]).abs().mean()),
    }


def calibrate_edgeconv(oracle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = pd.read_csv(RUN_TABLES / "edgeconv_validation_threshold_sweep.csv")
    test = pd.read_csv(TABLES / "edgeconv_test_results.csv")
    val["setting_key"] = setting_key(val)
    test["setting_key"] = setting_key(test)
    oracle_lookup = oracle[["scene_id", "oracle_exterior_P80_mm"]]
    test = test.merge(oracle_lookup, on="scene_id", how="left")

    coef_rows = []
    summary_rows = []
    for key, test_group in test.groupby("setting_key", sort=False):
        val_group = val[val["setting_key"].eq(key)].copy()
        if val_group.empty:
            continue
        a, b = fit_linear(val_group["predicted_P80_mm"].to_numpy(float), val_group["ground_truth_P80_mm"].to_numpy(float))
        val_group["calibrated_P80_mm"] = a * val_group["predicted_P80_mm"] + b
        test_group = test_group.copy()
        test_group["calibrated_P80_mm"] = a * test_group["predicted_P80_mm"] + b

        first = test_group.iloc[0]
        coef_rows.append(
            {
                "variant": first["variant"],
                "threshold": float(first["threshold"]),
                "bridge_probability": first["bridge_probability"],
                "graph_threshold": first["graph_threshold"],
                "calibration_slope": a,
                "calibration_intercept_mm": b,
                **{f"val_calibrated_{k}": v for k, v in summarize_errors(val_group, "calibrated_P80_mm", "ground_truth_P80_mm").items()},
            }
        )

        raw_full = summarize_errors(test_group, "predicted_P80_mm", "ground_truth_P80_mm")
        exterior = summarize_errors(test_group, "predicted_P80_mm", "oracle_exterior_P80_mm")
        calibrated_full = summarize_errors(test_group, "calibrated_P80_mm", "ground_truth_P80_mm")
        summary_rows.append(
            {
                "variant": first["variant"],
                "threshold": float(first["threshold"]),
                "bridge_probability": first["bridge_probability"],
                "graph_threshold": first["graph_threshold"],
                "n_scenes": int(test_group["scene_id"].nunique()),
                "mean_NMI": float(test_group["normalized_mutual_info"].mean()),
                "mean_ARI": float(test_group["adjusted_rand_index"].mean()),
                "mean_noise_fraction": float(test_group["noise_fraction"].mean()),
                "raw_full_mean_abs_P80_error_pct": raw_full["mean_abs_error_pct"],
                "raw_full_median_abs_P80_error_pct": raw_full["median_abs_error_pct"],
                "exterior_proxy_mean_abs_error_pct_vs_oracle": exterior["mean_abs_error_pct"],
                "exterior_proxy_median_abs_error_pct_vs_oracle": exterior["median_abs_error_pct"],
                "calibrated_full_mean_abs_P80_error_pct": calibrated_full["mean_abs_error_pct"],
                "calibrated_full_median_abs_P80_error_pct": calibrated_full["median_abs_error_pct"],
                "calibrated_full_mean_signed_error_pct": calibrated_full["mean_signed_error_pct"],
                "calibrated_full_mean_abs_error_mm": calibrated_full["mean_abs_error_mm"],
                "calibration_slope": a,
                "calibration_intercept_mm": b,
            }
        )

    coefs = pd.DataFrame(coef_rows).sort_values("val_calibrated_mean_abs_error_pct").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values("calibrated_full_mean_abs_P80_error_pct").reset_index(drop=True)
    coefs.to_csv(TABLES / "edgeconv_exterior_to_full_calibration_coefficients.csv", index=False)
    summary.to_csv(TABLES / "edgeconv_three_stage_calibrated_test_summary.csv", index=False)
    return coefs, summary


def write_oracle_summary(oracle: pd.DataFrame) -> pd.DataFrame:
    out = (
        oracle.groupby("split", as_index=False)
        .agg(
            n_scenes=("scene_id", "count"),
            mean_oracle_exterior_P80_mm=("oracle_exterior_P80_mm", "mean"),
            mean_full_ground_truth_P80_mm=("ground_truth_P80_mm", "mean"),
            mean_signed_bias_pct=("oracle_signed_error_pct_vs_full", "mean"),
            mean_abs_bias_pct=("oracle_abs_error_pct_vs_full", "mean"),
            mean_visible_fragments=("n_visible_fragments", "mean"),
            mean_exterior_points=("n_exterior_points", "mean"),
        )
        .sort_values("split")
    )
    out.to_csv(TABLES / "hpr_oracle_exterior_to_full_bias_summary.csv", index=False)
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    oracle = write_oracle_exterior_p80()
    oracle_summary = write_oracle_summary(oracle)
    coefs, summary = calibrate_edgeconv(oracle)
    print("Oracle exterior label bias:")
    print(oracle_summary.to_string(index=False))
    print("\nBest calibrated EdgeConv variants:")
    print(
        summary[
            [
                "variant",
                "threshold",
                "bridge_probability",
                "graph_threshold",
                "mean_NMI",
                "mean_ARI",
                "mean_noise_fraction",
                "exterior_proxy_mean_abs_error_pct_vs_oracle",
                "raw_full_mean_abs_P80_error_pct",
                "calibrated_full_mean_abs_P80_error_pct",
                "calibrated_full_mean_signed_error_pct",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print("\nCalibration coefficients:")
    print(coefs.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
