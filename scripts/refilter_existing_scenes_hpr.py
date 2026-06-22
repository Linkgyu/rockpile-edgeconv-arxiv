from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.regenerate_multi_scene_dataset import BASE_SEED  # noqa: E402
from src.data.exterior_filter import default_viewpoints, exterior_points_from_hpr_viewpoints  # noqa: E402
from src.features.surface_features import edge_features, estimate_normals_curvature, knn_edges  # noqa: E402


DATA_DIR = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
FILTER_VERSION = "hpr_multiview_exterior"


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def load_centered_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Could not load a valid mesh from {mesh_path}")
    mesh.apply_translation(-mesh.centroid)
    return mesh


def placed_mesh_from_row(row) -> trimesh.Trimesh:
    mesh = load_centered_mesh(Path(row.source_mesh_path))
    rotation = np.eye(4)
    rotation[:3, :3] = quaternion_to_matrix(
        np.array([row.orientation_qw, row.orientation_qx, row.orientation_qy, row.orientation_qz], dtype=float)
    )
    mesh.apply_transform(rotation)
    mesh.apply_translation([row.x_center_m, row.y_center_m, row.z_center_m])
    return mesh


def sample_full_surface(metadata: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    point_parts = []
    label_parts = []
    for row in metadata.sort_values("local_fragment_id").itertuples(index=False):
        mesh = placed_mesh_from_row(row)
        n_points = int(row.n_surface_points)
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        pts = np.asarray(pts, dtype=np.float32)
        point_parts.append(pts)
        label_parts.append(np.full(len(pts), int(row.local_fragment_id), dtype=np.int32))
    return np.vstack(point_parts).astype(np.float32), np.concatenate(label_parts).astype(np.int32)


def derive_companion_paths(scene_path: Path) -> tuple[Path, Path]:
    stem = scene_path.stem
    return scene_path.with_name(f"{stem}_fragments.csv"), scene_path.with_name(f"{stem}_ground_truth_psd.csv")


def refilter_scene(payload: dict) -> dict:
    row = payload["row"]
    output_tag = payload["output_tag"]
    force = payload["force"]
    radius_scale = payload["radius_scale"]

    scene_id = int(row["scene_id"])
    split = str(row["split"])
    old_scene_path = Path(row["path"])
    old_meta_path, old_psd_path = derive_companion_paths(old_scene_path)
    if not old_meta_path.exists():
        raise FileNotFoundError(f"Missing fragment metadata: {old_meta_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"scene_{output_tag}_{scene_id:03d}_{split}"
    scene_path = DATA_DIR / f"{prefix}.npz"
    metadata_path = DATA_DIR / f"{prefix}_fragments.csv"
    psd_path = DATA_DIR / f"{prefix}_ground_truth_psd.csv"

    if scene_path.exists() and metadata_path.exists() and psd_path.exists() and not force:
        existing = np.load(scene_path)
        return {
            **row,
            "path": str(scene_path),
            "dataset_tag": output_tag,
            "n_fragments": int(len(np.unique(existing["instance_labels"]))),
            "n_exterior_points": int(len(existing["points_xyz"])),
            "n_edges": int(len(existing["edges"])),
            "positive_edge_fraction": float(existing["edge_same_fragment"].mean()),
            "scan_filter_version": str(existing["scan_filter_version"][0]),
            "status": "reused_existing_hpr_refiltered",
        }

    old_scene = np.load(old_scene_path)
    metadata = pd.read_csv(old_meta_path)
    full_points, full_labels = sample_full_surface(metadata, seed=BASE_SEED + scene_id)
    viewpoints = default_viewpoints(full_points, margin=0.8)
    exterior_points, exterior_labels, visible_idx = exterior_points_from_hpr_viewpoints(
        full_points,
        full_labels,
        viewpoints,
        radius_scale=radius_scale,
        height_envelope_grid_m=None,
    )

    rng = np.random.default_rng(BASE_SEED + scene_id + 100_000)
    keep_prob = float(old_scene["density_keep_fraction"][0]) if "density_keep_fraction" in old_scene.files else 1.0
    exterior_points = exterior_points.astype(np.float32)
    exterior_labels = exterior_labels.astype(np.int32)
    exterior_points = exterior_points + rng.normal(0, 0.0015, size=exterior_points.shape).astype(np.float32)
    keep = rng.random(len(exterior_points)) < keep_prob
    exterior_points = exterior_points[keep]
    exterior_labels = exterior_labels[keep]
    visible_idx = visible_idx[keep]

    normals, curvature = estimate_normals_curvature(exterior_points, k_neighbors=30)
    edges = knn_edges(exterior_points, k=12)
    x_edges = edge_features(exterior_points, normals, curvature, edges)
    y_edges = (exterior_labels[edges[:, 0]] == exterior_labels[edges[:, 1]]).astype(np.int8)

    np.savez_compressed(
        scene_path,
        scene_id=np.array([scene_id]),
        split=np.array([split]),
        points_xyz=exterior_points.astype(np.float32),
        instance_labels=exterior_labels.astype(np.int32),
        exterior_source_indices=visible_idx.astype(np.int64),
        normals=normals,
        curvature=curvature,
        edges=edges,
        edge_features=x_edges,
        edge_same_fragment=y_edges,
        ground_truth_P10_mm=old_scene["ground_truth_P10_mm"],
        ground_truth_P50_mm=old_scene["ground_truth_P50_mm"],
        ground_truth_P80_mm=old_scene["ground_truth_P80_mm"],
        cone_base_radius_m=old_scene["cone_base_radius_m"],
        cone_height_m=old_scene["cone_height_m"],
        hpr_radius_scale=np.array([radius_scale]),
        height_envelope_grid_m=np.array([np.nan]),
        height_envelope_tolerance_m=np.array([np.nan]),
        height_envelope_mode=np.array(["none"]),
        density_keep_fraction=np.array([keep_prob]),
        scan_filter_version=np.array([FILTER_VERSION]),
        dataset_tag=np.array([output_tag]),
        placement_backend=old_scene["placement_backend"] if "placement_backend" in old_scene.files else np.array(["realistic-dem"]),
        requested_n_fragments=old_scene["requested_n_fragments"] if "requested_n_fragments" in old_scene.files else np.array([len(metadata)]),
        requested_total_surface_points=old_scene["requested_total_surface_points"]
        if "requested_total_surface_points" in old_scene.files
        else np.array([len(full_points)]),
        source_scene_path=np.array([str(old_scene_path)]),
        source_scan_filter_version=old_scene["scan_filter_version"] if "scan_filter_version" in old_scene.files else np.array(["unknown"]),
    )
    metadata.to_csv(metadata_path, index=False)
    if old_psd_path.exists():
        shutil.copyfile(old_psd_path, psd_path)

    return {
        "scene_id": scene_id,
        "split": split,
        "path": str(scene_path),
        "dataset_tag": output_tag,
        "n_fragments": int(len(np.unique(exterior_labels))),
        "n_exterior_points": int(len(exterior_points)),
        "n_edges": int(len(edges)),
        "positive_edge_fraction": float(y_edges.mean()),
        "ground_truth_P10_mm": float(old_scene["ground_truth_P10_mm"][0]),
        "ground_truth_P50_mm": float(old_scene["ground_truth_P50_mm"][0]),
        "ground_truth_P80_mm": float(old_scene["ground_truth_P80_mm"][0]),
        "cone_base_radius_m": float(old_scene["cone_base_radius_m"][0]),
        "cone_height_m": float(old_scene["cone_height_m"][0]),
        "scan_filter_version": FILTER_VERSION,
        "placement_backend": str(old_scene["placement_backend"][0]) if "placement_backend" in old_scene.files else "realistic-dem",
        "status": "hpr_refiltered_existing_pose",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply HPR exterior filtering to existing DEM scene poses.")
    parser.add_argument("--source-index", type=Path, default=ROOT / "results" / "tables" / "dem_noboundary_relax150_100scene_index.csv")
    parser.add_argument("--output-tag", type=str, default="dem_noboundary_relax150_100scene_hpr")
    parser.add_argument("--index-name", type=str, default="dem_noboundary_relax150_100scene_hpr_index.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--radius-scale", type=float, default=100.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.source_index)
    if args.limit is not None:
        source = source.head(args.limit).copy()

    payloads = [
        {
            "row": row.to_dict(),
            "output_tag": args.output_tag,
            "force": args.force,
            "radius_scale": args.radius_scale,
        }
        for _, row in source.iterrows()
    ]
    rows = []
    if args.workers <= 1 or len(payloads) <= 1:
        for idx, payload in enumerate(payloads, start=1):
            result = refilter_scene(payload)
            rows.append(result)
            print(f"[{idx}/{len(payloads)}] scene {result['scene_id']:03d} {result['status']}", flush=True)
    else:
        print(f"refiltering {len(payloads)} existing scenes with {args.workers} CPU workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_scene = {executor.submit(refilter_scene, payload): payload["row"]["scene_id"] for payload in payloads}
            for idx, future in enumerate(as_completed(future_to_scene), start=1):
                result = future.result()
                rows.append(result)
                print(f"[{idx}/{len(payloads)}] scene {result['scene_id']:03d} {result['status']}", flush=True)

    out = pd.DataFrame(rows).sort_values("scene_id").reset_index(drop=True)
    out_path = OUT_TABLES / args.index_name
    out.to_csv(out_path, index=False)
    print(out.groupby("split").size().to_string(), flush=True)
    print(out[["n_fragments", "n_exterior_points", "n_edges", "positive_edge_fraction"]].describe().to_string(), flush=True)
    print(out_path, flush=True)


if __name__ == "__main__":
    main()
