from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "edgeconv_hybrid_bridge_calib_12ep"
REEVAL = RUN / "reevaluations" / "fine_threshold_p80_selected"
RESULT_TABLES = ROOT / "results" / "tables"
RESULT_FIGURES = ROOT / "results" / "figures"
MANUSCRIPT_FIGURES = ROOT / "manuscript" / "figures"
OVERLEAF_FIGURES = ROOT / "manuscript" / "overleaf_project" / "figures"
ARXIV_FIGURES = ROOT / "manuscript" / "arxiv_source_upload" / "figures"


def ensure_inputs() -> None:
    required = [
        RUN / "tables" / "edgeconv_training_history.csv",
        RUN / "figures" / "02_edgeconv_training_curve.png",
        REEVAL / "tables" / "edgeconv_test_results.csv",
        REEVAL / "tables" / "edgeconv_test_summary.csv",
        REEVAL / "tables" / "edgeconv_validation_threshold_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required improved EdgeConv outputs:\n" + "\n".join(missing))


def copy_edgeconv_outputs() -> None:
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN / "tables" / "edgeconv_training_history.csv", RESULT_TABLES / "edgeconv_training_history.csv")
    shutil.copy2(REEVAL / "tables" / "edgeconv_test_results.csv", RESULT_TABLES / "edgeconv_test_results.csv")
    shutil.copy2(REEVAL / "tables" / "edgeconv_test_summary.csv", RESULT_TABLES / "edgeconv_test_summary.csv")
    shutil.copy2(REEVAL / "tables" / "edgeconv_validation_threshold_summary.csv", RESULT_TABLES / "edgeconv_validation_threshold_summary.csv")
    shutil.copy2(RUN / "figures" / "02_edgeconv_training_curve.png", RESULT_FIGURES / "02_edgeconv_training_curve.png")


def selected_edgeconv_row() -> dict:
    test_results = pd.read_csv(REEVAL / "tables" / "edgeconv_test_results.csv")
    selected = test_results[test_results["variant"] == "edgeconv_post_split"].copy()
    if selected.empty:
        raise SystemExit("No edgeconv_post_split rows in improved test results")
    signed_pct = (selected["predicted_P80_mm"] - selected["ground_truth_P80_mm"]) / selected["ground_truth_P80_mm"] * 100.0
    threshold = float(selected["threshold"].iloc[0])
    return {
        "method": "EdgeConv post split",
        "setting": f"threshold={threshold:.4f}",
        "n_scenes": int(selected["scene_id"].nunique()),
        "bias_P80_error_pct": float(signed_pct.mean()),
        "mean_abs_P80_error_pct": float(selected["abs_P80_error_pct"].mean()),
        "std_abs_P80_error_pct": float(selected["abs_P80_error_pct"].std(ddof=1)),
        "median_abs_P80_error_pct": float(selected["abs_P80_error_pct"].median()),
        "max_abs_P80_error_pct": float(selected["abs_P80_error_pct"].max()),
        "mean_abs_P80_error_mm": float(selected["abs_P80_error_mm"].mean()),
        "mean_NMI": float(selected["normalized_mutual_info"].mean()),
        "mean_ARI": float(selected["adjusted_rand_index"].mean()),
        "mean_noise_fraction": float(selected["noise_fraction"].mean()),
        "mean_predicted_clusters": float(selected["n_predicted_clusters"].mean()),
    }


def update_model_comparison() -> pd.DataFrame:
    baseline = pd.read_csv(RESULT_TABLES / "baseline_150frag_test_summary.csv")
    edge_row = selected_edgeconv_row()
    comparison = pd.concat([baseline, pd.DataFrame([edge_row])], ignore_index=True)
    comparison = comparison.sort_values("mean_abs_P80_error_pct").reset_index(drop=True)
    comparison.to_csv(RESULT_TABLES / "model_comparison_150frag_test_summary.csv", index=False)
    return comparison


def write_comparison_figure(comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    colors = ["#2F6B7A" if method.startswith("EdgeConv") else "#6B7C8F" for method in comparison["method"]]
    bars = ax.bar(comparison["method"], comparison["mean_abs_P80_error_pct"], color=colors)
    ax.set_ylabel("Mean absolute P80 error [%]")
    ax.set_title("150-fragment held-out test comparison")
    ax.grid(True, axis="y", color="#D8DEE8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=28)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for bar, value in zip(bars, comparison["mean_abs_P80_error_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ymax = max(25.0, float(comparison["mean_abs_P80_error_pct"].max()) * 1.18)
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(RESULT_FIGURES / "model_comparison_150frag_p80.png", bbox_inches="tight")
    plt.close(fig)


def write_edgeconv_histogram() -> None:
    test_results = pd.read_csv(REEVAL / "tables" / "edgeconv_test_results.csv")
    selected = test_results[test_results["variant"] == "edgeconv_post_split"].copy()
    values = selected["abs_P80_error_pct"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    ax.hist(values, bins=np.linspace(0, max(5.0, values.max() + 2.0), 9), color="#2F6B7A", edgecolor="white")
    ax.set_xlabel("Absolute P80 error [%]")
    ax.set_ylabel("Number of test scenes")
    ax.grid(True, axis="y", color="#D8DEE8", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(RESULT_FIGURES / "03_edgeconv_test_p80_error_histogram.png", bbox_inches="tight")
    plt.close(fig)


def sync_manuscript_figures() -> None:
    for target_dir in [MANUSCRIPT_FIGURES, OVERLEAF_FIGURES, ARXIV_FIGURES]:
        if not target_dir.exists():
            continue
        for name in ["02_edgeconv_training_curve.png", "03_edgeconv_test_p80_error_histogram.png", "model_comparison_150frag_p80.png"]:
            shutil.copy2(RESULT_FIGURES / name, target_dir / name)


def main() -> None:
    ensure_inputs()
    copy_edgeconv_outputs()
    comparison = update_model_comparison()
    write_comparison_figure(comparison)
    write_edgeconv_histogram()
    sync_manuscript_figures()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
