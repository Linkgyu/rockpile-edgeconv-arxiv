# Synthetic Exterior Rockpile EdgeConv Benchmark

This repository contains the arXiv-ready manuscript source, code, figures, and
summary results for a synthetic exterior point-cloud benchmark for rockpile
fragmentation and P80 estimation using a DGCNN/EdgeConv edge-affinity model.
The manuscript narrative follows the full benchmark chain: synthetic fragments
are generated first with known identity and volume, those fragments are arranged
into labelled rockpile scenes, only exterior scan-like points are retained for
learning, and EdgeConv edge affinity is evaluated as a transferable alternative
to scene-specific fragment-ID classification or purely geometric clustering.

## What Is Included

- `manuscript/arxiv_main.tex`: arXiv-oriented LaTeX source.
- `manuscript/arxiv_manuscript.docx`: Word/PDF submission source generated from
  the same manuscript content.
- `manuscript/figures/`: compact figures used in the manuscript.
- `scripts/`: scene generation, preview rendering, training, and evaluation.
- `src/`: EdgeConv model, point-cloud loading, PSD proxy, segmentation, and
  post-processing utilities.
- `results/tables/`: scene index and summary CSVs from the reported run.

The full generated `.npz` scenes and `.pt` model checkpoints are intentionally
not committed. They can be regenerated from the included scripts.

## Reported Run

Dataset:

- 100 synthetic no-boundary DEM-relaxed exterior rockpile scenes
- 150 requested fragments per scene
- Scene-level split: 60 train, 20 validation, 20 held-out test
- Mean visible fragments: 146.68
- Mean exterior points: 5735.63
- Mean pile base radius: 0.898 m
- Mean pile height: 1.084 m

EdgeConv training/evaluation:

- 12-epoch CPU-compatible PyTorch calibration run
- Best validation AP: 0.9208 at epoch 12
- Validation-selected variant: `edgeconv_post_split`
- Validation-selected threshold: 0.9995 after a fine high-threshold sweep
- Held-out test mean absolute P80 error: 12.40%
- Held-out test median absolute P80 error: 13.32%
- Mean test noise fraction: 0.831

## Reproduce

Install dependencies:

```bash
conda env create -f environment.yml
conda activate rockpile-edgeconv-arxiv
```

Regenerate data and train/evaluate using the commands in `data/README.md`.

## Important Limitation

The model learns edge affinity clearly, but PSD estimation remains sensitive to
threshold and connected-component post-processing. In the reported DEM-relaxed
dataset, validation selects a high threshold of 0.9995. This lowers held-out P80
error below the hand-crafted graph baseline, but it also produces a high noise
fraction on test scenes. This repository should therefore be treated as a
controlled synthetic benchmark and a reproducible baseline, not as a field-ready
fragmentation monitoring system.

## Exterior Filter Update

The exterior scan filter was updated after diagnosing that the previous global
top-envelope cleanup removed visible side/slope points. The new
`preserve_side_visible` mode keeps points seen from side viewpoints and applies
the height-envelope cleanup only as an overhead/interior safeguard. See
`results/figures/exterior_filter_diagnostic_scene000.png` for the comparison.
