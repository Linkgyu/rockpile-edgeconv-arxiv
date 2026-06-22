# Data Policy

The full generated 100-scene `.npz` dataset and model checkpoints are not
committed to this repository because they are generated artifacts and can be
large. The committed files include:

- the exact scene index used in the reported run:
  `results/tables/dem_noboundary_relax150_100scene_index.csv`
- training history, validation threshold summary, and held-out test summary
- scripts needed to regenerate scenes and retrain the EdgeConv model
- an exterior-filter diagnostic:
  `results/figures/exterior_filter_diagnostic_scene000.png` and
  `results/tables/exterior_filter_diagnostic_scene000.csv`

The exterior filter now preserves points seen from side viewpoints before
applying the plan-view height-envelope cleanup. The previous top-only envelope
could remove physically visible slope/side points and even drop visible
fragments. Newly generated scenes are marked with
`angular_nearest_plus_xy_height_envelope_preserve_side_visible`.

To regenerate the dataset, place or generate the Synthetic_Rockpile fragment
catalog and set the input paths if they differ from the author's workstation:

```powershell
$env:SYNTHETIC_ROCKPILE_ROOT = "C:\path\to\Synthetic_Rockpile"
$env:REALISTIC_ROCKPILE_GENERATOR_ROOT = "C:\path\to\physics-informed-realistic-rockpile-generator"
```

Then run:

```powershell
python scripts/regenerate_multi_scene_dataset.py `
  --placement-backend realistic-dem `
  --dem-preset noboundary_axis_clump_150_fast `
  --dem-steps 800 `
  --dem-dt 0.0005 `
  --n-scenes 100 `
  --start-scene 0 `
  --train-scenes 60 `
  --val-scenes 20 `
  --dataset-tag dem_noboundary_relax150_100scene `
  --n-fragments 150 `
  --total-surface-points 30000 `
  --workers 4 `
  --index-name dem_noboundary_relax150_100scene_index.csv
```

Then train/evaluate:

```powershell
python scripts/retrain_and_evaluate_edgeconv.py `
  --scene-index outputs/tables/dem_noboundary_relax150_100scene_index.csv `
  --run-name dem_noboundary_relax150_100scene_e12_fine_threshold `
  --max-epochs 12 `
  --patience 6 `
  --max-train-edges 18000 `
  --max-val-edges 28000 `
  --val-metric-scenes 8 `
  --photogrammetry-realism 0.75 `
  --noise-penalty 0.0 `
  --max-noise-fraction 1.0
```

To inspect the exterior filter on one regenerated scene:

```powershell
python scripts/diagnose_exterior_filter.py `
  --scene-id 0 `
  --n-fragments 150 `
  --dem-preset noboundary_axis_clump_150_fast
```
