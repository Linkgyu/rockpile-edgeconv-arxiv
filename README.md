# Synthetic Rockpile EdgeConv Benchmark

This repository contains the arXiv-ready manuscript source, code, figures, and
summary results for a synthetic exterior point-cloud benchmark for rockpile
fragmentation and P80 estimation using a DGCNN/EdgeConv edge-affinity model.

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

- 24 epochs, CPU-compatible PyTorch implementation
- Best validation AP: 0.9313 at epoch 24
- Validation-selected variant: `edgeconv_post_split`
- Validation-selected threshold: 0.997
- Held-out test mean absolute P80 error: 19.04%
- Held-out test median absolute P80 error: 18.36%
- Mean test noise fraction: 0.603

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
dataset, validation selects a high threshold of 0.997, producing a high noise
fraction on test scenes. This repository should therefore be treated as a
controlled synthetic benchmark and a reproducible baseline, not as a field-ready
fragmentation monitoring system.

