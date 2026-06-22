from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.retrain_and_evaluate_edgeconv as runner  # noqa: E402
from src.data.scene_dataset import load_scene_index  # noqa: E402
from src.models.edgeconv import EdgeAffinityDGCNN  # noqa: E402


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.lower() in {"", "none", "nan"}:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one fixed EdgeConv post-processing setting on the test split.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--bridge-probability", default=None)
    parser.add_argument("--graph-threshold", default=None)
    parser.add_argument("--scene-index", type=Path, default=None)
    parser.add_argument("--out-name", default="")
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / "runs" / args.run_name
    ckpt_path = run_dir / "models" / "edgeconv_affinity.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    index = load_scene_index(args.scene_index)
    test_rows = index[index["split"] == "test"].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EdgeAffinityDGCNN(point_channels=7, edge_attr_channels=15, hidden=48, emb=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bridge_probability = parse_optional_float(args.bridge_probability)
    graph_threshold = parse_optional_float(args.graph_threshold)
    out_name = args.out_name or args.variant
    out_dir = run_dir / "fixed_evaluations" / out_name
    old_tables, old_models, old_figures = runner.OUT_TABLES, runner.OUT_MODELS, runner.OUT_FIGURES
    runner.OUT_TABLES = out_dir / "tables"
    runner.OUT_MODELS = out_dir / "models"
    runner.OUT_FIGURES = out_dir / "figures"
    runner.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    runner.OUT_MODELS.mkdir(parents=True, exist_ok=True)
    runner.OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        test_results = runner.evaluate_test_split(
            model,
            test_rows,
            device,
            args.variant,
            args.threshold,
            selected_bridge_probability=bridge_probability,
            selected_graph_threshold=graph_threshold,
        )
        test_results.to_csv(runner.OUT_TABLES / "edgeconv_test_results.csv", index=False)
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
        test_summary["selected_for_transfer"] = test_summary["variant"].eq(args.variant)
        test_summary.to_csv(runner.OUT_TABLES / "edgeconv_test_summary.csv", index=False)
        print(test_summary.to_string(index=False))
        print(f"saved: {out_dir}")
    finally:
        runner.OUT_TABLES, runner.OUT_MODELS, runner.OUT_FIGURES = old_tables, old_models, old_figures


if __name__ == "__main__":
    main()
