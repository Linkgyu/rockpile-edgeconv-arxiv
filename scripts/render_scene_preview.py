from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _subsample(points: np.ndarray, labels: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], labels[idx]


def _colors(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    lut = {label: i for i, label in enumerate(unique)}
    color_idx = np.array([lut[label] for label in labels], dtype=int)
    return plt.cm.tab20((color_idx % 20) / 19.0)


def _set_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = float((maxs - mins).max()) * 1.08
    span = max(span, 0.8)
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(max(-0.05, mins[2] - 0.04), maxs[2] + 0.08)
    ax.set_axis_off()


def render(scene_path: Path, output_png: Path, output_gif: Path | None, max_points: int, seed: int) -> None:
    data = np.load(scene_path)
    points = np.asarray(data["points_xyz"], dtype=float)
    labels = np.asarray(data["instance_labels"], dtype=int)
    points, labels = _subsample(points, labels, max_points=max_points, seed=seed)
    colors = _colors(labels)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.4, alpha=0.88, linewidths=0)
    _set_axes(ax, points)
    ax.view_init(elev=27, azim=-42)
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(output_png)

    if output_gif is None:
        return

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 5.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame_idx: int):
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.2, alpha=0.88, linewidths=0)
        _set_axes(ax, points)
        ax.view_init(elev=27, azim=-50 + frame_idx * 4)
        return []

    anim = FuncAnimation(fig, draw, frames=72, interval=80, blit=False)
    anim.save(output_gif, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(output_gif)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a generated rockpile scene preview.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-gif", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    render(args.scene, args.output_png, args.output_gif, args.max_points, args.seed)


if __name__ == "__main__":
    main()
