from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
RUN_TABLES = ROOT / "outputs" / "runs" / "hpr_multiview_24ep" / "tables"


SELECTED_EDGE_VARIANT = "edgeconv_hybrid_bridge_absorb_post_split"


def setting_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["variant"].astype(str)
        + "|"
        + df["threshold"].map(lambda x: f"{float(x):.4f}")
        + "|"
        + df["bridge_probability"].fillna(-1).map(lambda x: f"{float(x):.4f}")
        + "|"
        + df["graph_threshold"].fillna(-1).map(lambda x: f"{float(x):.4f}")
    )


def error_vector(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return (pred - true) / true * 100.0


def bootstrap_mean_abs_ci(errors_pct: np.ndarray, seed: int = 42, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(errors_pct, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(np.abs(values), size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]).astype(float))


def selected_edgeconv_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    val = pd.read_csv(RUN_TABLES / "edgeconv_validation_threshold_sweep.csv")
    test = pd.read_csv(TABLES / "edgeconv_test_results.csv")
    coefs = pd.read_csv(TABLES / "edgeconv_exterior_to_full_calibration_coefficients.csv")
    coef = coefs[coefs["variant"].eq(SELECTED_EDGE_VARIANT)].iloc[0]
    mask_val = (
        val["variant"].eq(SELECTED_EDGE_VARIANT)
        & np.isclose(val["threshold"], float(coef["threshold"]))
        & np.isclose(val["bridge_probability"].fillna(-1), float(coef["bridge_probability"]))
        & np.isclose(val["graph_threshold"].fillna(-1), float(coef["graph_threshold"]))
    )
    mask_test = (
        test["variant"].eq(SELECTED_EDGE_VARIANT)
        & np.isclose(test["threshold"], float(coef["threshold"]))
        & np.isclose(test["bridge_probability"].fillna(-1), float(coef["bridge_probability"]))
        & np.isclose(test["graph_threshold"].fillna(-1), float(coef["graph_threshold"]))
    )
    return val[mask_val].copy(), test[mask_test].copy(), coef


def write_combined_calibrated_summary() -> pd.DataFrame:
    baseline = pd.read_csv(TABLES / "baseline_150frag_calibrated_test_summary.csv")
    baseline = baseline.rename(columns={"method": "Method", "setting": "Setting"})
    baseline["Variant"] = baseline["Method"]

    val, test, coef = selected_edgeconv_rows()
    a = float(coef["calibration_slope"])
    b = float(coef["calibration_intercept_mm"])
    test["calibrated_P80_mm"] = a * test["predicted_P80_mm"] + b
    raw_err = error_vector(test["predicted_P80_mm"].to_numpy(float), test["ground_truth_P80_mm"].to_numpy(float))
    cal_err = error_vector(test["calibrated_P80_mm"].to_numpy(float), test["ground_truth_P80_mm"].to_numpy(float))
    raw_lo, raw_hi = bootstrap_mean_abs_ci(raw_err, seed=101)
    cal_lo, cal_hi = bootstrap_mean_abs_ci(cal_err, seed=102)
    edge = pd.DataFrame(
        [
            {
                "Method": "EdgeConv",
                "Setting": "threshold=0.9999,bridge=0.50,graph=0.85",
                "Variant": SELECTED_EDGE_VARIANT.replace("edgeconv_", "").replace("_", " "),
                "n_val_scenes": int(val["scene_id"].nunique()),
                "n_test_scenes": int(test["scene_id"].nunique()),
                "calibration_slope": a,
                "calibration_intercept_mm": b,
                "raw_full_mean_abs_P80_error_pct": float(np.nanmean(np.abs(raw_err))),
                "raw_full_mean_abs_P80_error_ci95_low": raw_lo,
                "raw_full_mean_abs_P80_error_ci95_high": raw_hi,
                "calibrated_full_mean_abs_P80_error_pct": float(np.nanmean(np.abs(cal_err))),
                "calibrated_full_mean_abs_P80_error_ci95_low": cal_lo,
                "calibrated_full_mean_abs_P80_error_ci95_high": cal_hi,
                "calibrated_full_mean_signed_error_pct": float(np.nanmean(cal_err)),
                "mean_NMI": float(test["normalized_mutual_info"].mean()),
                "mean_ARI": float(test["adjusted_rand_index"].mean()),
                "mean_noise_fraction": float(test["noise_fraction"].mean()),
                "mean_predicted_clusters": float("nan"),
            }
        ]
    )
    common = [
        "Method",
        "Setting",
        "Variant",
        "n_val_scenes",
        "n_test_scenes",
        "calibration_slope",
        "calibration_intercept_mm",
        "raw_full_mean_abs_P80_error_pct",
        "raw_full_mean_abs_P80_error_ci95_low",
        "raw_full_mean_abs_P80_error_ci95_high",
        "calibrated_full_mean_abs_P80_error_pct",
        "calibrated_full_mean_abs_P80_error_ci95_low",
        "calibrated_full_mean_abs_P80_error_ci95_high",
        "calibrated_full_mean_signed_error_pct",
        "mean_NMI",
        "mean_ARI",
        "mean_noise_fraction",
        "mean_predicted_clusters",
    ]
    combined = pd.concat([baseline[common], edge[common]], ignore_index=True)
    combined = combined.sort_values("calibrated_full_mean_abs_P80_error_pct").reset_index(drop=True)
    combined.to_csv(TABLES / "calibrated_method_comparison_test_summary.csv", index=False)
    return combined


def plot_edgeconv_calibration_scatter() -> None:
    val, test, coef = selected_edgeconv_rows()
    a = float(coef["calibration_slope"])
    b = float(coef["calibration_intercept_mm"])
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(val["predicted_P80_mm"], val["ground_truth_P80_mm"], s=42, marker="o", label="Validation", alpha=0.85)
    ax.scatter(test["predicted_P80_mm"], test["ground_truth_P80_mm"], s=42, marker="s", label="Test", alpha=0.85)
    xs = np.linspace(
        min(val["predicted_P80_mm"].min(), test["predicted_P80_mm"].min()) * 0.97,
        max(val["predicted_P80_mm"].max(), test["predicted_P80_mm"].max()) * 1.03,
        100,
    )
    ax.plot(xs, xs, color="0.55", lw=1.2, ls="--", label="1:1")
    ax.plot(xs, a * xs + b, color="#b3432d", lw=2.0, label=f"Val fit: y={a:.3f}x+{b:.1f}")
    ax.set_xlabel("Predicted exterior P80 (mm)")
    ax.set_ylabel("Full ground-truth P80 (mm)")
    ax.set_title("Exterior-to-full P80 calibration for selected EdgeConv")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "edgeconv_p80_calibration_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity() -> None:
    val = pd.read_csv(RUN_TABLES / "edgeconv_validation_threshold_sweep.csv")
    focus = val[val["variant"].isin(["edgeconv_hybrid_bridge_absorb_post_split", "edgeconv_post_split"])].copy()
    summary = (
        focus.groupby(["variant", "threshold"], as_index=False)
        .agg(
            mean_abs_P80_error_pct=("abs_P80_error_pct", "mean"),
            mean_NMI=("normalized_mutual_info", "mean"),
            mean_noise=("noise_fraction", "mean"),
        )
        .sort_values("threshold")
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for variant, group in summary.groupby("variant"):
        label = variant.replace("edgeconv_", "").replace("_", " ")
        ax.plot(group["threshold"], group["mean_abs_P80_error_pct"], marker="o", ms=3, lw=1.5, label=label)
    ax.set_xlabel("Affinity threshold")
    ax.set_ylabel("Validation mean absolute P80 error (%)")
    ax.set_title("EdgeConv threshold sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "edgeconv_threshold_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_calibrated_method_comparison(summary: pd.DataFrame) -> None:
    labels = summary["Method"].replace({"Graph threshold": "Graph", "MLP affinity": "MLP"}).tolist()
    x = np.arange(len(summary))
    raw = summary["raw_full_mean_abs_P80_error_pct"].to_numpy(float)
    cal = summary["calibrated_full_mean_abs_P80_error_pct"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x - 0.18, raw, width=0.36, label="Raw exterior P80")
    ax.bar(x + 0.18, cal, width=0.36, label="Validation-calibrated full P80")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Mean absolute P80 error (%)")
    ax.set_title("Raw versus calibrated full-P80 error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibrated_method_comparison_p80.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = write_combined_calibrated_summary()
    plot_edgeconv_calibration_scatter()
    plot_threshold_sensitivity()
    plot_calibrated_method_comparison(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
