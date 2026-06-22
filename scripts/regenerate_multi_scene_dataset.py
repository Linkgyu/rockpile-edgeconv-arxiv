from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.exterior_filter import default_viewpoints, exterior_points_from_viewpoints
from src.data.synthetic_piles import (
    generate_chrono_dem_scene,
    generate_synthetic_rockpile_v1_scene,
    generate_physics_informed_dem_scene,
    ground_truth_psd_from_metadata,
    load_fragment_catalog,
    percentile_from_psd,
)
from src.features.surface_features import edge_features, estimate_normals_curvature, knn_edges


SOURCE_ROOT = Path(os.environ.get("SYNTHETIC_ROCKPILE_ROOT", r"C:/Users/creep/code/python/Synthetic_Rockpile"))
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

N_FRAGMENTS_PER_SCENE = 220
TOTAL_SURFACE_POINTS = 42_000
BASE_SEED = 20260525
FILTER_VERSION = "angular_depthbuffer_plus_xy_height_envelope_preserve_side_visible"


DEM_PRESETS = {
    "sequential_sphere": {
        "placement_mode": "sequential_drop",
        "contact_model": "sphere",
        "collision_radius_scale": 0.55,
        "drop_source_radius_m": 0.10,
        "drop_height_m": 0.08,
        "drop_initial_speed_m_s": 1.10,
        "drop_lateral_speed_m_s": 0.45,
        "drop_angular_speed_rad_s": 4.0,
        "settle_steps_per_fragment": 80,
        "angle_of_repose_deg": 38.0,
        "rollout_trials_per_fragment": 96,
        "rollout_radial_penalty": 0.045,
        "final_settle_steps": None,
        "wall_release_fraction": 0.35,
        "linear_damping": 0.018,
    },
    "sequential_axis_clump": {
        "placement_mode": "sequential_drop",
        "contact_model": "axis_clump",
        "collision_radius_scale": 0.55,
        "clump_radius_scale": 0.31,
        "clump_spread_scale": 0.44,
        "drop_source_radius_m": 0.10,
        "drop_height_m": 0.08,
        "drop_initial_speed_m_s": 1.10,
        "drop_lateral_speed_m_s": 0.45,
        "drop_angular_speed_rad_s": 4.0,
        "settle_steps_per_fragment": 60,
        "angle_of_repose_deg": 38.0,
        "rollout_trials_per_fragment": 96,
        "rollout_radial_penalty": 0.045,
        "final_settle_steps": None,
        "wall_release_fraction": 0.35,
        "linear_damping": 0.020,
    },
    "loose_sphere": {
        "placement_mode": "envelope_relax",
    },
    "compact_sphere": {
        "placement_mode": "envelope_relax",
        "contact_model": "sphere",
        "collision_radius_scale": 0.55,
        "initial_column_radius_m": 0.62,
        "pile_envelope_height_m": 0.82,
        "temporary_wall_radius_m": 0.82,
        "wall_release_fraction": 0.82,
        "linear_damping": 0.018,
        "initial_radial_power": 1.25,
        "radial_penalty_weight": 0.055,
        "below_envelope_weight": 0.45,
    },
    "wide_stable_sphere": {
        "placement_mode": "envelope_relax",
        "contact_model": "sphere",
        "collision_radius_scale": 0.54,
        "initial_column_radius_m": 0.78,
        "pile_envelope_height_m": 0.88,
        "temporary_wall_radius_m": 0.98,
        "wall_release_fraction": 0.84,
        "linear_damping": 0.020,
        "initial_radial_power": 1.18,
        "radial_penalty_weight": 0.045,
        "below_envelope_weight": 0.42,
    },
    "compact_axis_clump": {
        "placement_mode": "envelope_relax",
        "contact_model": "axis_clump",
        "collision_radius_scale": 0.55,
        "clump_radius_scale": 0.31,
        "clump_spread_scale": 0.44,
        "initial_column_radius_m": 0.66,
        "pile_envelope_height_m": 0.84,
        "temporary_wall_radius_m": 0.86,
        "wall_release_fraction": 0.86,
        "linear_damping": 0.022,
        "initial_radial_power": 1.25,
        "radial_penalty_weight": 0.060,
        "below_envelope_weight": 0.48,
    },
    "wide_axis_clump": {
        "placement_mode": "envelope_relax",
        "contact_model": "axis_clump",
        "collision_radius_scale": 0.55,
        "clump_radius_scale": 0.30,
        "clump_spread_scale": 0.48,
        "initial_column_radius_m": 0.78,
        "pile_envelope_height_m": 0.88,
        "temporary_wall_radius_m": 1.00,
        "wall_release_fraction": 0.88,
        "linear_damping": 0.022,
        "initial_radial_power": 1.20,
        "radial_penalty_weight": 0.050,
        "below_envelope_weight": 0.46,
    },
    "noboundary_axis_clump_150_fast": {
        "placement_mode": "envelope_relax",
        "contact_model": "axis_clump",
        "collision_radius_scale": 0.55,
        "clump_radius_scale": 0.30,
        "clump_spread_scale": 0.48,
        "initial_column_radius_m": 0.78,
        "pile_envelope_height_m": 0.88,
        "temporary_wall_radius_m": None,
        "wall_release_fraction": 0.0,
        "linear_damping": 0.028,
        "initial_radial_power": 1.20,
        "radial_penalty_weight": 0.060,
        "below_envelope_weight": 0.50,
    },
}


CHRONO_PRESETS = {
    "bench_convex": {
        "contact_representation": "convex_hull",
        "friction": 0.82,
        "restitution": 0.08,
        "damping": 0.18,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.18,
        "release_height_m": 1.15,
        "initial_down_speed_m_s": 0.35,
        "initial_lateral_speed_m_s": 0.42,
    },
    "bench_clump": {
        "contact_representation": "clump",
        "friction": 0.82,
        "restitution": 0.08,
        "damping": 0.18,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.18,
        "release_height_m": 1.15,
        "initial_down_speed_m_s": 0.35,
        "initial_lateral_speed_m_s": 0.42,
        "clump_radius_scale": 0.34,
        "clump_spread_scale": 0.42,
    },
    "contained_convex": {
        "contact_representation": "convex_hull",
        "friction": 0.95,
        "restitution": 0.08,
        "damping": 0.18,
        "floor_size_m": 14.0,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.04,
        "release_height_m": 0.40,
        "initial_down_speed_m_s": 0.12,
        "initial_lateral_speed_m_s": 0.02,
    },
    "contained_convex_220": {
        "contact_representation": "convex_hull",
        "friction": 0.95,
        "restitution": 0.04,
        "damping": 0.22,
        "floor_size_m": 18.0,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.14,
        "release_height_m": 0.35,
        "release_spacing_m": 0.018,
        "initial_placement_mode": "cone_pack",
        "initial_pile_radius_m": 2.1,
        "initial_collision_radius_scale": 0.78,
        "placement_trials_per_fragment": 160,
        "placement_radial_penalty": 0.035,
        "sequential_release": False,
        "release_batch_size": 8,
        "settle_steps_per_release": 140,
        "final_settle_steps": 7000,
        "initial_down_speed_m_s": 0.12,
        "initial_lateral_speed_m_s": 0.025,
    },
    "sequential_drop_220": {
        "contact_representation": "convex_hull",
        "friction": 1.05,
        "static_friction": 1.15,
        "sliding_friction": 1.05,
        "rolling_friction": 0.32,
        "spinning_friction": 0.20,
        "restitution": 0.02,
        "damping": 0.24,
        "floor_size_m": 18.0,
        "containment_radius_m": 1.35,
        "containment_wall_height_m": 1.55,
        "containment_wall_thickness_m": 0.12,
        "containment_segments": 36,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.055,
        "release_height_m": 0.30,
        "release_spacing_m": 0.018,
        "initial_placement_mode": "release",
        "sequential_release": True,
        "release_batch_size": 4,
        "settle_steps_per_release": 450,
        "final_settle_steps": 7000,
        "settle_speed_m_s": 0.08,
        "initial_down_speed_m_s": 0.02,
        "initial_lateral_speed_m_s": 0.0,
    },
    "granite_sequential_220": {
        "contact_representation": "convex_hull",
        "density_kg_m3": 2670.0,
        "young_modulus_pa": 5.0e10,
        "poisson_ratio": 0.25,
        "friction": 1.05,
        "static_friction": 1.15,
        "sliding_friction": 0.90,
        "rolling_friction": 0.28,
        "spinning_friction": 0.18,
        "restitution": 0.03,
        "damping": 0.24,
        "floor_size_m": 18.0,
        "containment_radius_m": 1.35,
        "containment_shape": "box",
        "containment_wall_height_m": 1.55,
        "containment_wall_thickness_m": 0.12,
        "containment_segments": 36,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.055,
        "release_height_m": 0.30,
        "release_spacing_m": 0.018,
        "initial_placement_mode": "release",
        "sequential_release": True,
        "release_batch_size": 4,
        "settle_steps_per_release": 450,
        "final_settle_steps": 7000,
        "settle_speed_m_s": 0.08,
        "initial_down_speed_m_s": 0.02,
        "initial_lateral_speed_m_s": 0.0,
    },
    "granite_noboundary_seq200_fast": {
        "contact_representation": "convex_hull",
        "density_kg_m3": 2670.0,
        "young_modulus_pa": 5.0e10,
        "poisson_ratio": 0.25,
        "friction": 0.85,
        "static_friction": 0.85,
        "sliding_friction": 0.65,
        "rolling_friction": 0.15,
        "spinning_friction": 0.03,
        "restitution": 0.20,
        "damping": 0.24,
        "floor_size_m": 18.0,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.055,
        "release_height_m": 0.30,
        "release_spacing_m": 0.018,
        "initial_placement_mode": "release",
        "sequential_release": True,
        "release_batch_size": 2,
        "release_top_quantile": 0.85,
        "settle_steps_per_release": 420,
        "final_settle_steps": 5000,
        "settle_speed_m_s": 0.06,
        "initial_down_speed_m_s": 0.02,
        "initial_lateral_speed_m_s": 0.0,
    },
    "granite_noboundary_clump_seq150_fast": {
        "contact_representation": "clump",
        "density_kg_m3": 2670.0,
        "young_modulus_pa": 5.0e10,
        "poisson_ratio": 0.25,
        "friction": 0.85,
        "static_friction": 0.85,
        "sliding_friction": 0.65,
        "rolling_friction": 0.15,
        "spinning_friction": 0.03,
        "restitution": 0.20,
        "damping": 0.24,
        "floor_size_m": 18.0,
        "bench_height_m": 0.42,
        "bench_width_m": 0.75,
        "release_radius_m": 0.055,
        "release_height_m": 0.30,
        "release_spacing_m": 0.018,
        "initial_placement_mode": "release",
        "sequential_release": True,
        "release_batch_size": 8,
        "release_top_quantile": 0.85,
        "settle_steps_per_release": 120,
        "final_settle_steps": 2500,
        "settle_speed_m_s": 0.08,
        "settle_check_interval": 120,
        "initial_down_speed_m_s": 0.02,
        "initial_lateral_speed_m_s": 0.0,
        "clump_radius_scale": 0.36,
        "clump_spread_scale": 0.46,
    },
}


def split_for_scene(scene_id: int, train_scenes: int, val_scenes: int) -> str:
    if scene_id < train_scenes:
        return "train"
    if scene_id < train_scenes + val_scenes:
        return "val"
    return "test"


def generate_scene(
    catalog: pd.DataFrame,
    scene_id: int,
    force: bool,
    n_fragments: int,
    total_surface_points: int,
    train_scenes: int,
    val_scenes: int,
    dataset_tag: str = "",
    split_override: str = "",
    placement_backend: str = "realistic-dem",
    dem_params: dict | None = None,
    chrono_params: dict | None = None,
) -> dict:
    split = split_override or split_for_scene(scene_id, train_scenes=train_scenes, val_scenes=val_scenes)
    prefix = f"scene_{dataset_tag}_{scene_id:03d}_{split}" if dataset_tag else f"scene_{scene_id:03d}_{split}"
    scene_path = DATA_DIR / f"{prefix}.npz"
    metadata_path = DATA_DIR / f"{prefix}_fragments.csv"
    psd_path = DATA_DIR / f"{prefix}_ground_truth_psd.csv"

    if scene_path.exists() and metadata_path.exists() and psd_path.exists() and not force:
        data_existing = np.load(scene_path)
        return {
            "scene_id": scene_id,
            "split": split,
            "path": str(scene_path),
            "dataset_tag": dataset_tag,
            "n_fragments": int(len(np.unique(data_existing["instance_labels"]))),
            "n_exterior_points": int(len(data_existing["points_xyz"])),
            "n_edges": int(len(data_existing["edges"])),
            "positive_edge_fraction": float(data_existing["edge_same_fragment"].mean()),
            "ground_truth_P10_mm": float(data_existing["ground_truth_P10_mm"][0]),
            "ground_truth_P50_mm": float(data_existing["ground_truth_P50_mm"][0]),
            "ground_truth_P80_mm": float(data_existing["ground_truth_P80_mm"][0]),
            "cone_base_radius_m": float(data_existing["cone_base_radius_m"][0]),
            "cone_height_m": float(data_existing["cone_height_m"][0]),
            "scan_filter_version": str(data_existing["scan_filter_version"][0])
            if "scan_filter_version" in data_existing.files
            else "unknown",
            "status": "reused_existing",
        }

    rng = np.random.default_rng(BASE_SEED + scene_id)
    if placement_backend == "synthetic-v1":
        full_points, full_labels, metadata, cone_geometry = generate_synthetic_rockpile_v1_scene(
            catalog,
            scene_id=scene_id,
            rng=rng,
            n_fragments=n_fragments,
            total_surface_points=total_surface_points,
        )
    elif placement_backend == "realistic-dem":
        full_points, full_labels, metadata, cone_geometry = generate_physics_informed_dem_scene(
            catalog,
            scene_id=scene_id,
            rng=rng,
            n_fragments=n_fragments,
            total_surface_points=total_surface_points,
            params=dem_params,
        )
    elif placement_backend == "chrono-dem":
        full_points, full_labels, metadata, cone_geometry = generate_chrono_dem_scene(
            catalog,
            scene_id=scene_id,
            rng=rng,
            n_fragments=n_fragments,
            total_surface_points=total_surface_points,
            params=chrono_params,
        )
    else:
        raise ValueError(f"Unknown placement backend: {placement_backend}")
    viewpoints = default_viewpoints(full_points, margin=0.8)
    exterior_points, exterior_labels, visible_idx = exterior_points_from_viewpoints(
        full_points,
        full_labels,
        viewpoints,
        angular_resolution_deg=0.24,
        range_tolerance_m=0.012,
        occlusion_neighbor_bins=2,
        height_envelope_grid_m=0.035,
        height_envelope_tolerance_m=0.030,
        height_envelope_mode="preserve_side_visible",
    )

    exterior_points = exterior_points.astype(np.float32)
    exterior_labels = exterior_labels.astype(np.int32)
    exterior_points = exterior_points + rng.normal(0, 0.0015, size=exterior_points.shape).astype(np.float32)
    keep_prob = float(rng.uniform(0.78, 0.92))
    keep = rng.random(len(exterior_points)) < keep_prob
    exterior_points = exterior_points[keep]
    exterior_labels = exterior_labels[keep]
    visible_idx = visible_idx[keep]

    normals, curvature = estimate_normals_curvature(exterior_points, k_neighbors=30)
    edges = knn_edges(exterior_points, k=12)
    x_edges = edge_features(exterior_points, normals, curvature, edges)
    y_edges = (exterior_labels[edges[:, 0]] == exterior_labels[edges[:, 1]]).astype(np.int8)

    psd = ground_truth_psd_from_metadata(metadata)
    gt_p10 = percentile_from_psd(psd, 10)
    gt_p50 = percentile_from_psd(psd, 50)
    gt_p80 = percentile_from_psd(psd, 80)

    np.savez_compressed(
        scene_path,
        scene_id=np.array([scene_id]),
        split=np.array([split]),
        points_xyz=exterior_points,
        instance_labels=exterior_labels,
        exterior_source_indices=visible_idx.astype(np.int64),
        normals=normals,
        curvature=curvature,
        edges=edges,
        edge_features=x_edges,
        edge_same_fragment=y_edges,
        ground_truth_P10_mm=np.array([gt_p10]),
        ground_truth_P50_mm=np.array([gt_p50]),
        ground_truth_P80_mm=np.array([gt_p80]),
        cone_base_radius_m=np.array([cone_geometry["base_radius_m"]]),
        cone_height_m=np.array([cone_geometry["pile_height_m"]]),
        angular_resolution_deg=np.array([0.24]),
        range_tolerance_m=np.array([0.012]),
        occlusion_neighbor_bins=np.array([2]),
        height_envelope_grid_m=np.array([0.035]),
        height_envelope_tolerance_m=np.array([0.030]),
        height_envelope_mode=np.array(["preserve_side_visible"]),
        density_keep_fraction=np.array([keep_prob]),
        scan_filter_version=np.array([FILTER_VERSION]),
        dataset_tag=np.array([dataset_tag]),
        placement_backend=np.array([placement_backend]),
        requested_n_fragments=np.array([n_fragments], dtype=np.int32),
        requested_total_surface_points=np.array([total_surface_points], dtype=np.int32),
    )
    metadata.to_csv(metadata_path, index=False)
    psd.to_csv(psd_path, index=False)

    return {
        "scene_id": scene_id,
        "split": split,
        "path": str(scene_path),
        "dataset_tag": dataset_tag,
        "n_fragments": int(len(np.unique(exterior_labels))),
        "n_exterior_points": int(len(exterior_points)),
        "n_edges": int(len(edges)),
        "positive_edge_fraction": float(y_edges.mean()),
        "ground_truth_P10_mm": float(gt_p10),
        "ground_truth_P50_mm": float(gt_p50),
        "ground_truth_P80_mm": float(gt_p80),
        "cone_base_radius_m": float(cone_geometry["base_radius_m"]),
        "cone_height_m": float(cone_geometry["pile_height_m"]),
        "scan_filter_version": FILTER_VERSION,
        "placement_backend": placement_backend,
        "status": "generated",
    }


def generate_scene_worker(payload: dict) -> dict:
    return generate_scene(**payload)


def build_scene_payloads(
    catalog: pd.DataFrame,
    args: argparse.Namespace,
    train_scenes: int,
    val_scenes: int,
    dataset_tag: str,
    dem_params: dict | None,
    chrono_params: dict | None = None,
) -> list[dict]:
    return [
        {
            "catalog": catalog,
            "scene_id": scene_id,
            "force": args.force,
            "n_fragments": args.n_fragments,
            "total_surface_points": args.total_surface_points,
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "dataset_tag": dataset_tag,
            "split_override": args.split_override,
            "placement_backend": args.placement_backend,
            "dem_params": dem_params,
            "chrono_params": chrono_params,
        }
        for scene_id in range(args.start_scene, args.n_scenes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--start-scene", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-fragments", type=int, default=N_FRAGMENTS_PER_SCENE)
    parser.add_argument("--total-surface-points", type=int, default=TOTAL_SURFACE_POINTS)
    parser.add_argument("--train-scenes", type=int, default=None)
    parser.add_argument("--val-scenes", type=int, default=None)
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default="",
        help="Optional tag added to scene filenames so larger datasets do not overwrite the baseline scenes.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="",
        help="Optional CSV name under outputs/tables. Defaults to the historical name for the 100-scene baseline.",
    )
    parser.add_argument(
        "--split-override",
        choices=["train", "val", "test"],
        default="",
        help="Force all generated scenes into one split; useful for appending extra training scenes.",
    )
    parser.add_argument(
        "--placement-backend",
        choices=["synthetic-v1", "realistic-dem", "chrono-dem"],
        default="realistic-dem",
        help="Pile placement backend. realistic-dem and chrono-dem use physics-informed-realistic-rockpile-generator.",
    )
    parser.add_argument("--dem-steps", type=int, default=4200, help="Soft-sphere DEM steps for --placement-backend realistic-dem.")
    parser.add_argument("--dem-dt", type=float, default=2.5e-4, help="Soft-sphere DEM timestep for --placement-backend realistic-dem.")
    parser.add_argument("--dem-point-noise-std", type=float, default=0.0)
    parser.add_argument("--dem-preset", choices=sorted(DEM_PRESETS), default="sequential_axis_clump")
    parser.add_argument("--dem-contact-model", choices=["sphere", "axis_clump"], default=None)
    parser.add_argument("--dem-collision-radius-scale", type=float, default=None)
    parser.add_argument("--dem-wall-release-fraction", type=float, default=None)
    parser.add_argument("--dem-temporary-wall-radius", type=float, default=None)
    parser.add_argument("--dem-linear-damping", type=float, default=None)
    parser.add_argument("--dem-placement-mode", choices=["sequential_drop", "envelope_relax"], default=None)
    parser.add_argument("--dem-drop-source-radius", type=float, default=None)
    parser.add_argument("--dem-drop-height", type=float, default=None)
    parser.add_argument("--dem-settle-steps-per-fragment", type=int, default=None)
    parser.add_argument("--chrono-preset", choices=sorted(CHRONO_PRESETS), default="bench_convex")
    parser.add_argument("--chrono-steps", type=int, default=8000)
    parser.add_argument("--chrono-dt", type=float, default=1.0e-3)
    parser.add_argument("--chrono-contact-representation", choices=["convex_hull", "clump"], default=None)
    parser.add_argument("--chrono-friction", type=float, default=None)
    parser.add_argument("--chrono-density", type=float, default=None)
    parser.add_argument("--chrono-static-friction", type=float, default=None)
    parser.add_argument("--chrono-sliding-friction", type=float, default=None)
    parser.add_argument("--chrono-rolling-friction", type=float, default=None)
    parser.add_argument("--chrono-spinning-friction", type=float, default=None)
    parser.add_argument("--chrono-restitution", type=float, default=None)
    parser.add_argument("--chrono-containment-radius", type=float, default=None)
    parser.add_argument("--chrono-release-batch-size", type=int, default=None)
    parser.add_argument("--chrono-release-top-quantile", type=float, default=None)
    parser.add_argument("--chrono-settle-steps-per-release", type=int, default=None)
    parser.add_argument("--chrono-final-settle-steps", type=int, default=None)
    parser.add_argument("--chrono-settle-speed", type=float, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of CPU worker processes for scene generation. Use 1 for sequential deterministic logging.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_fragment_catalog(SOURCE_ROOT)
    if args.n_fragments > len(catalog):
        raise SystemExit(f"--n-fragments={args.n_fragments} exceeds catalog size {len(catalog)}")
    train_scenes = args.train_scenes if args.train_scenes is not None else int(round(args.n_scenes * 0.60))
    val_scenes = args.val_scenes if args.val_scenes is not None else int(round(args.n_scenes * 0.20))
    if not args.split_override and train_scenes + val_scenes >= args.n_scenes:
        raise SystemExit("--train-scenes + --val-scenes must leave at least one test scene")
    dataset_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.dataset_tag).strip("_")
    rows = []
    dem_params = None
    chrono_params = None
    if args.placement_backend == "realistic-dem":
        dem_params = dict(DEM_PRESETS[args.dem_preset])
        dem_params.update({
            "n_steps": args.dem_steps,
            "dt_s": args.dem_dt,
            "point_noise_std_m": args.dem_point_noise_std,
        })
        if args.dem_contact_model is not None:
            dem_params["contact_model"] = args.dem_contact_model
        if args.dem_collision_radius_scale is not None:
            dem_params["collision_radius_scale"] = args.dem_collision_radius_scale
        if args.dem_wall_release_fraction is not None:
            dem_params["wall_release_fraction"] = args.dem_wall_release_fraction
        if args.dem_temporary_wall_radius is not None:
            dem_params["temporary_wall_radius_m"] = None if args.dem_temporary_wall_radius <= 0 else args.dem_temporary_wall_radius
        if args.dem_linear_damping is not None:
            dem_params["linear_damping"] = args.dem_linear_damping
        if args.dem_placement_mode is not None:
            dem_params["placement_mode"] = args.dem_placement_mode
        if args.dem_drop_source_radius is not None:
            dem_params["drop_source_radius_m"] = args.dem_drop_source_radius
        if args.dem_drop_height is not None:
            dem_params["drop_height_m"] = args.dem_drop_height
        if args.dem_settle_steps_per_fragment is not None:
            dem_params["settle_steps_per_fragment"] = args.dem_settle_steps_per_fragment
    if args.placement_backend == "chrono-dem":
        chrono_params = dict(CHRONO_PRESETS[args.chrono_preset])
        chrono_params.update({"n_steps": args.chrono_steps, "dt_s": args.chrono_dt})
        if args.chrono_contact_representation is not None:
            chrono_params["contact_representation"] = args.chrono_contact_representation
        if args.chrono_friction is not None:
            chrono_params["friction"] = args.chrono_friction
        if args.chrono_density is not None:
            chrono_params["density_kg_m3"] = args.chrono_density
        if args.chrono_static_friction is not None:
            chrono_params["static_friction"] = args.chrono_static_friction
        if args.chrono_sliding_friction is not None:
            chrono_params["sliding_friction"] = args.chrono_sliding_friction
        if args.chrono_rolling_friction is not None:
            chrono_params["rolling_friction"] = args.chrono_rolling_friction
        if args.chrono_spinning_friction is not None:
            chrono_params["spinning_friction"] = args.chrono_spinning_friction
        if args.chrono_restitution is not None:
            chrono_params["restitution"] = args.chrono_restitution
        if args.chrono_containment_radius is not None:
            chrono_params["containment_radius_m"] = args.chrono_containment_radius
        if args.chrono_release_batch_size is not None:
            chrono_params["release_batch_size"] = args.chrono_release_batch_size
        if args.chrono_release_top_quantile is not None:
            chrono_params["release_top_quantile"] = args.chrono_release_top_quantile
        if args.chrono_settle_steps_per_release is not None:
            chrono_params["settle_steps_per_release"] = args.chrono_settle_steps_per_release
        if args.chrono_final_settle_steps is not None:
            chrono_params["final_settle_steps"] = args.chrono_final_settle_steps
        if args.chrono_settle_speed is not None:
            chrono_params["settle_speed_m_s"] = args.chrono_settle_speed
    payloads = build_scene_payloads(catalog, args, train_scenes, val_scenes, dataset_tag, dem_params, chrono_params)
    if args.workers <= 1 or len(payloads) <= 1:
        for payload in payloads:
            row = generate_scene_worker(payload)
            rows.append(row)
            print(row, flush=True)
    else:
        print(f"generating {len(payloads)} scenes with {args.workers} CPU workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_scene_worker, payload): payload["scene_id"] for payload in payloads}
            for future in as_completed(futures):
                scene_id = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Scene {scene_id} failed") from exc
                rows.append(row)
                print(row, flush=True)
        rows.sort(key=lambda item: int(item["scene_id"]))

    scene_index = pd.DataFrame(rows)
    if args.index_name:
        out = TABLE_DIR / args.index_name
    elif args.start_scene == 0 and args.n_scenes == 100 and not dataset_tag:
        out = TABLE_DIR / "multi_scene_index.csv"
    else:
        tag_part = f"_{dataset_tag}" if dataset_tag else ""
        out = TABLE_DIR / f"multi_scene_index{tag_part}_{args.start_scene:03d}_{args.n_scenes:03d}.csv"
    scene_index.to_csv(out, index=False)
    print(out)
    print(
        scene_index.groupby("split").agg(
            n_scenes=("scene_id", "count"),
            mean_points=("n_exterior_points", "mean"),
            mean_edges=("n_edges", "mean"),
        )
    )


if __name__ == "__main__":
    main()
