from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.retrain_and_evaluate_edgeconv as runner  # noqa: E402
from src.data.scene_dataset import load_scene_index  # noqa: E402
from src.models.edgeconv import EdgeAffinityDGCNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate an existing EdgeConv run with new threshold-selection rules.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scene-index", type=Path, default=None)
    parser.add_argument("--noise-penalty", type=float, default=0.0)
    parser.add_argument("--max-noise-fraction", type=float, default=1.0)
    parser.add_argument("--out-name", default="")
    args = parser.parse_args()

    run_dir = ROOT / "outputs" / "runs" / args.run_name
    ckpt_path = run_dir / "models" / "edgeconv_affinity.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    metadata_path = run_dir / "tables" / "edgeconv_retraining_metadata.json"
    scene_index = args.scene_index
    if scene_index is None and metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = meta.get("scene_index")
        if raw and raw != "default":
            scene_index = Path(raw)

    index = load_scene_index(scene_index)
    val_rows = index[index["split"] == "val"].reset_index(drop=True)
    test_rows = index[index["split"] == "test"].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeAffinityDGCNN(point_channels=7, edge_attr_channels=15, hidden=48, emb=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_suffix = args.out_name or f"noise_penalty_{args.noise_penalty:g}_max_noise_{args.max_noise_fraction:g}"
    out_dir = run_dir / "reevaluations" / out_suffix
    old_tables = runner.OUT_TABLES
    old_models = runner.OUT_MODELS
    old_figures = runner.OUT_FIGURES
    runner.OUT_TABLES = out_dir / "tables"
    runner.OUT_MODELS = out_dir / "models"
    runner.OUT_FIGURES = out_dir / "figures"
    runner.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    runner.OUT_MODELS.mkdir(parents=True, exist_ok=True)
    runner.OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    try:
        sweep = runner.validation_threshold_sweep(model, val_rows, device)
        sweep.to_csv(runner.OUT_TABLES / "edgeconv_validation_threshold_sweep.csv", index=False)
        selected_variant, selected_threshold, validation_summary = runner.choose_validation_setting(
            sweep,
            noise_penalty=args.noise_penalty,
            max_noise_fraction=args.max_noise_fraction,
        )
        validation_summary.to_csv(runner.OUT_TABLES / "edgeconv_validation_threshold_summary.csv", index=False)
        test_results = runner.evaluate_test_split(model, test_rows, device, selected_variant, selected_threshold)
        test_results.to_csv(runner.OUT_TABLES / "edgeconv_test_results.csv", index=False)
        test_summary = (
            test_results.groupby("variant", as_index=False)
            .agg(
                n_scenes=("scene_id", "count"),
                threshold=("threshold", "first"),
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
        test_summary.to_csv(runner.OUT_TABLES / "edgeconv_test_summary.csv", index=False)
        (runner.OUT_TABLES / "reevaluation_metadata.json").write_text(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "scene_index": str(scene_index) if scene_index else "default",
                    "noise_penalty": args.noise_penalty,
                    "max_noise_fraction": args.max_noise_fraction,
                    "selected_variant": selected_variant,
                    "selected_threshold": selected_threshold,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(test_summary.to_string(index=False))
        print(f"saved: {out_dir}")
    finally:
        runner.OUT_TABLES = old_tables
        runner.OUT_MODELS = old_models
        runner.OUT_FIGURES = old_figures


if __name__ == "__main__":
    main()
