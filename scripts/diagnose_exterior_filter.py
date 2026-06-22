from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.regenerate_multi_scene_dataset import BASE_SEED, DEM_PRESETS  # noqa: E402
from src.data.exterior_filter import default_viewpoints, exterior_points_from_viewpoints  # noqa: E402
from src.data.synthetic_piles import generate_physics_informed_dem_scene, load_fragment_catalog  # noqa: E402


SOURCE_ROOT = Path(os.environ.get("SYNTHETIC_ROCKPILE_ROOT", r"C:/Users/creep/code/python/Synthetic_Rockpile"))
OUT_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"


def label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(123)
    unique = np.unique(labels)
    colors = {int(label): rng.uniform(0.10, 0.95, size=3) for label in unique}
    return np.array([colors[int(label)] for label in labels])


def downsample(points: np.ndarray, labels: np.ndarray, max_points: int, seed: int):
    if len(points) <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], labels[idx]


def set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-58)


def half_cut(points: np.ndarray, labels: np.ndarray, cut_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one half of the pile so the interior section is visible."""

    keep = points[:, 0] <= cut_x
    return points[keep], labels[keep]


def summarize(name: str, points: np.ndarray, labels: np.ndarray, full_labels: np.ndarray) -> dict:
    counts = pd.Series(labels).value_counts()
    full_counts = pd.Series(full_labels).value_counts()
    joined = pd.DataFrame({"kept": counts, "full": full_counts}).fillna(0)
    retained_ratio = (joined["kept"] / joined["full"].clip(lower=1)).to_numpy()
    return {
        "stage": name,
        "n_points": int(len(points)),
        "n_fragments": int(len(np.unique(labels))),
        "z_min": float(points[:, 2].min()),
        "z_q05": float(np.quantile(points[:, 2], 0.05)),
        "z_median": float(np.median(points[:, 2])),
        "z_q95": float(np.quantile(points[:, 2], 0.95)),
        "z_max": float(points[:, 2].max()),
        "median_fragment_retained_ratio": float(np.median(retained_ratio)),
        "q10_fragment_retained_ratio": float(np.quantile(retained_ratio, 0.10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and side-preserving exterior filters on one regenerated scene.")
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument("--n-fragments", type=int, default=150)
    parser.add_argument("--total-surface-points", type=int, default=42000)
    parser.add_argument("--dem-preset", choices=sorted(DEM_PRESETS), default="noboundary_axis_clump_150_fast")
    parser.add_argument("--max-plot-points", type=int, default=9000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    catalog = load_fragment_catalog(SOURCE_ROOT)
    rng = np.random.default_rng(BASE_SEED + args.scene_id)
    dem_params = dict(DEM_PRESETS[args.dem_preset])
    dem_params.update({"n_steps": 4200, "dt_s": 2.5e-4, "point_noise_std_m": 0.0})
    full_points, full_labels, _, _ = generate_physics_informed_dem_scene(
        catalog,
        scene_id=args.scene_id,
        rng=rng,
        n_fragments=args.n_fragments,
        total_surface_points=args.total_surface_points,
        params=dem_params,
    )
    viewpoints = default_viewpoints(full_points, margin=0.8)
    view_points, view_labels, _ = exterior_points_from_viewpoints(
        full_points,
        full_labels,
        viewpoints,
        angular_resolution_deg=0.24,
        height_envelope_grid_m=None,
        height_envelope_mode="none",
    )
    old_points, old_labels, _ = exterior_points_from_viewpoints(
        full_points,
        full_labels,
        viewpoints,
        angular_resolution_deg=0.24,
        height_envelope_grid_m=0.035,
        height_envelope_tolerance_m=0.030,
        height_envelope_mode="top_only",
    )
    new_points, new_labels, _ = exterior_points_from_viewpoints(
        full_points,
        full_labels,
        viewpoints,
        angular_resolution_deg=0.24,
        height_envelope_grid_m=0.035,
        height_envelope_tolerance_m=0.030,
        height_envelope_mode="preserve_side_visible",
    )

    summary = pd.DataFrame(
        [
            summarize("full sampled surface", full_points, full_labels, full_labels),
            summarize("viewpoint visible only", view_points, view_labels, full_labels),
            summarize("old top-only envelope", old_points, old_labels, full_labels),
            summarize("new preserve-side envelope", new_points, new_labels, full_labels),
        ]
    )
    summary_path = TABLE_DIR / f"exterior_filter_diagnostic_scene{args.scene_id:03d}.csv"
    summary.to_csv(summary_path, index=False)

    panels = [
        ("A. Full sampled surface", full_points, full_labels),
        ("B. Viewpoint visible", view_points, view_labels),
        ("C. Old top-only envelope", old_points, old_labels),
        ("D. New preserve-side envelope", new_points, new_labels),
    ]
    fig = plt.figure(figsize=(14, 10), dpi=170)
    for i, (title, points, labels) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        pts, labs = downsample(points, labels, args.max_plot_points, seed=10 + i)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=label_colors(labs), s=1.4, alpha=0.78, linewidths=0)
        ax.set_title(f"{title}\n{len(points):,} points, {len(np.unique(labels))} fragments")
        set_equal_3d(ax, full_points)
    fig.suptitle("Exterior filter diagnostic: old top-only envelope vs side-preserving envelope", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = OUT_DIR / f"exterior_filter_diagnostic_scene{args.scene_id:03d}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    cut_x = float(np.median(full_points[:, 0]))
    section_panels = [
        ("A. Full sampled surface, half cut", full_points, full_labels),
        ("B. Viewpoint visible, half cut", view_points, view_labels),
        ("C. Old top-only envelope, half cut", old_points, old_labels),
        ("D. New preserve-side envelope, half cut", new_points, new_labels),
    ]
    fig = plt.figure(figsize=(14, 10), dpi=170)
    for i, (title, points, labels) in enumerate(section_panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        section_points, section_labels = half_cut(points, labels, cut_x)
        pts, labs = downsample(section_points, section_labels, args.max_plot_points, seed=100 + i)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=label_colors(labs), s=1.8, alpha=0.82, linewidths=0)
        ax.set_title(f"{title}\n{len(section_points):,} section points")
        set_equal_3d(ax, full_points)
        ax.view_init(elev=18, azim=-82)
    fig.suptitle(f"Exterior filter half-cut diagnostic: x <= {cut_x:.3f} m", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    section_fig_path = OUT_DIR / f"exterior_filter_halfcut_scene{args.scene_id:03d}.png"
    fig.savefig(section_fig_path, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(fig_path)
    print(section_fig_path)
    print(summary_path)


if __name__ == "__main__":
    main()
