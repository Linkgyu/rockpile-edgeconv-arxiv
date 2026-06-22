from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "hpr_multiview_24ep"
RESULT_TABLES = ROOT / "results" / "tables"
RESULT_FIGURES = ROOT / "results" / "figures"
MANUSCRIPT_FIGURES = ROOT / "manuscript" / "figures"


def copy_outputs() -> None:
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    for name in [
        "edgeconv_training_history.csv",
        "edgeconv_validation_threshold_summary.csv",
        "edgeconv_test_results.csv",
        "edgeconv_test_summary.csv",
        "edgeconv_retraining_metadata.json",
        "baseline_150frag_mlp_training_history.csv",
        "baseline_150frag_selected_settings.csv",
        "baseline_150frag_test_scene_results.csv",
        "baseline_150frag_test_summary.csv",
        "baseline_150frag_validation_summary.csv",
        "model_comparison_150frag_test_summary.csv",
    ]:
        shutil.copy2(RUN / "tables" / name, RESULT_TABLES / name)
    shutil.copy2(ROOT / "outputs" / "tables" / "dem_noboundary_relax150_100scene_hpr_index.csv", RESULT_TABLES / "dem_noboundary_relax150_100scene_hpr_index.csv")

    for name in ["02_edgeconv_training_curve.png", "03_edgeconv_test_p80_error_histogram.png", "model_comparison_150frag_p80.png"]:
        shutil.copy2(RUN / "figures" / name, RESULT_FIGURES / name)


def write_comparison_figure() -> None:
    comparison = pd.read_csv(RESULT_TABLES / "model_comparison_150frag_test_summary.csv").copy()
    labels = []
    for method in comparison["method"]:
        if str(method).startswith("EdgeConv"):
            labels.append("EdgeConv hybrid")
        else:
            labels.append(str(method))
    comparison["plot_label"] = labels
    ordered = comparison.sort_values("mean_abs_P80_error_pct").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    colors = ["#2F6B7A" if label.startswith("EdgeConv") else "#6B7C8F" for label in ordered["plot_label"]]
    bars = ax.bar(ordered["plot_label"], ordered["mean_abs_P80_error_pct"], color=colors)
    ax.set_ylabel("Mean absolute P80 error [%]")
    ax.set_title("HPR exterior scan: 150-fragment held-out test comparison")
    ax.grid(True, axis="y", color="#D8DEE8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for bar, value in zip(bars, ordered["mean_abs_P80_error_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(40.0, float(ordered["mean_abs_P80_error_pct"].max()) * 1.18))
    fig.tight_layout()
    fig.savefig(RESULT_FIGURES / "model_comparison_150frag_p80.png", bbox_inches="tight")
    plt.close(fig)


def sync_manuscript_figures() -> None:
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)
    for name in [
        "02_edgeconv_training_curve.png",
        "03_edgeconv_test_p80_error_histogram.png",
        "model_comparison_150frag_p80.png",
        "exterior_filter_section_scan_scene000.png",
    ]:
        shutil.copy2(RESULT_FIGURES / name, MANUSCRIPT_FIGURES / name)


def main() -> None:
    copy_outputs()
    write_comparison_figure()
    sync_manuscript_figures()
    print(pd.read_csv(RESULT_TABLES / "model_comparison_150frag_test_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
