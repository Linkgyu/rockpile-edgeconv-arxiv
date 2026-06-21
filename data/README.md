# Data Policy

The full generated 100-scene `.npz` dataset and model checkpoints are not
committed to this repository because they are generated artifacts and can be
large. The committed files include:

- the exact scene index used in the reported run:
  `results/tables/dem_noboundary_relax150_100scene_index.csv`
- training history, validation threshold summary, and held-out test summary
- scripts needed to regenerate scenes and retrain the EdgeConv model

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
  --run-name dem_noboundary_relax150_100scene_e24 `
  --max-epochs 24 `
  --patience 6 `
  --max-train-edges 22000 `
  --max-val-edges 36000 `
  --val-metric-scenes 12 `
  --photogrammetry-realism 0.75 `
  --noise-penalty 0.10 `
  --max-noise-fraction 0.60
```
