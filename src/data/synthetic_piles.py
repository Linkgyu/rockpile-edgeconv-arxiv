"""Generate multiple labelled synthetic muckpile point-cloud scenes.

The generator uses fragment meshes from Synthetic_Rockpile but creates new
random pile arrangements inside this repository. It is a benchmark generator,
not a DEM simulator.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import trimesh


REALISTIC_GENERATOR_ROOT = Path(
    os.environ.get(
        "REALISTIC_ROCKPILE_GENERATOR_ROOT",
        r"C:/Users/creep/code/python/physics-informed-realistic-rockpile-generator",
    )
)


def load_fragment_catalog(source_root: Path) -> pd.DataFrame:
    """Load fragment metadata and resolve mesh paths."""

    source_root = Path(source_root)
    metadata_path = source_root / "outputs" / "metadata" / "fragment_generation_metadata.csv"
    catalog = pd.read_csv(metadata_path)
    catalog["mesh_abs_path"] = catalog["mesh_path"].apply(lambda p: str(source_root / p))
    return catalog


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Draw a uniformly distributed 3D rotation matrix."""

    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _sample_mesh_points(mesh_path: Path, n_points: int, rng: np.random.Generator) -> np.ndarray:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        vertices = np.asarray(mesh.vertices, dtype=float)
        idx = rng.choice(len(vertices), size=n_points, replace=True)
        return vertices[idx]
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=float)


def generate_random_pile_scene(
    catalog: pd.DataFrame,
    scene_id: int,
    rng: np.random.Generator,
    n_fragments: int = 220,
    total_surface_points: int = 45_000,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate a labelled synthetic pile by randomly arranging fragment meshes."""

    scene_catalog = catalog.sample(n=n_fragments, replace=False, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    scene_catalog = scene_catalog.reset_index(drop=True)
    volumes = scene_catalog["volume_m3"].to_numpy(float)
    diameters = scene_catalog["equivalent_diameter_m"].to_numpy(float)
    weights = volumes / volumes.sum()
    counts = np.maximum(24, np.round(total_surface_points * weights).astype(int))
    # Keep the exact point count close to target without starving small fragments.
    scale = total_surface_points / counts.sum()
    counts = np.maximum(24, np.round(counts * scale).astype(int))

    median_d = float(np.median(diameters))
    pile_radius = max(1.0, 0.31 * np.sqrt(n_fragments) * median_d)
    pile_height = 1.15 * pile_radius

    all_points = []
    all_labels = []
    rows = []
    for local_id, row in scene_catalog.iterrows():
        d = float(row["equivalent_diameter_m"])
        r = pile_radius * np.sqrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(theta) + rng.normal(0, 0.035)
        y = r * np.sin(theta) + rng.normal(0, 0.035)
        mound_z = pile_height * max(0.0, 1.0 - r / pile_radius)
        z = 0.5 * d + mound_z + rng.normal(0, 0.07)

        pts = _sample_mesh_points(Path(row["mesh_abs_path"]), int(counts[local_id]), rng)
        pts = pts - pts.mean(axis=0, keepdims=True)
        rot = random_rotation_matrix(rng)
        pts = pts @ rot.T + np.array([x, y, z])
        all_points.append(pts.astype(np.float32))
        all_labels.append(np.full(len(pts), local_id, dtype=np.int32))
        rows.append(
            {
                "scene_id": int(scene_id),
                "local_fragment_id": int(local_id),
                "source_fragment_id": int(row["fragment_id"]),
                "volume_m3": float(row["volume_m3"]),
                "equivalent_diameter_m": d,
                "equivalent_diameter_mm": d * 1000,
                "n_surface_points": int(len(pts)),
                "x_center_m": float(x),
                "y_center_m": float(y),
                "z_center_m": float(z),
            }
        )

    points = np.vstack(all_points).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    metadata = pd.DataFrame(rows)
    return points, labels, metadata


def ground_truth_psd_from_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-weighted ground-truth PSD for a generated scene."""

    df = metadata.sort_values("equivalent_diameter_m").copy()
    volume = df["volume_m3"].to_numpy(float)
    df["volume_fraction"] = volume / np.clip(volume.sum(), 1e-12, None)
    df["cumulative_passing_pct"] = np.cumsum(df["volume_fraction"]) * 100
    return df


def percentile_from_psd(psd: pd.DataFrame, pct: float) -> float:
    return float(np.interp(pct, psd["cumulative_passing_pct"], psd["equivalent_diameter_mm"]))



DEFAULT_SYNTHETIC_ROCKPILE_PARAMS = {
    "angle_of_repose_deg": 38.0,
    "target_packing_fraction": 0.62,
    "collision_radius_scale": 0.60,
    "separation_factor": 0.92,
    "trials_per_fragment": 520,
    "radial_penalty_weight": 0.035,
    "max_expansion_steps": 6,
    "radius_expansion_factor_per_step": 1.06,
}


def _recenter_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    centered = mesh.copy()
    centered.apply_translation(-centered.centroid)
    return centered


def _estimate_bounding_radius(mesh: trimesh.Trimesh) -> float:
    distances = np.linalg.norm(mesh.vertices - mesh.centroid, axis=1)
    return float(distances.max())


def _compute_ground_clearance(mesh: trimesh.Trimesh) -> float:
    return float(-mesh.bounds[0, 2])


def _apply_random_rotation(mesh: trimesh.Trimesh, rng: np.random.Generator) -> tuple[trimesh.Trimesh, tuple[float, float, float]]:
    rotated = mesh.copy()
    rx, ry, rz = rng.uniform(0, 2 * np.pi, size=3)
    rotation_matrix = trimesh.transformations.euler_matrix(rx, ry, rz, axes="sxyz")
    rotated.apply_transform(rotation_matrix)
    return rotated, (float(rx), float(ry), float(rz))


def _sample_xy_inside_disk(max_radius_m: float, rng: np.random.Generator) -> tuple[float, float]:
    theta = rng.uniform(0, 2 * np.pi)
    radius = max_radius_m * np.sqrt(rng.random())
    return float(radius * np.cos(theta)), float(radius * np.sin(theta))


def _estimate_cone_geometry(prepared_fragments: list[dict], params: dict, radius_expansion_factor: float = 1.0) -> dict:
    theta_rad = np.radians(params["angle_of_repose_deg"])
    tan_theta = np.tan(theta_rad)
    effective_collision_volume = 0.0
    for frag in prepared_fragments:
        r = frag["collision_radius_m"]
        effective_collision_volume += (4.0 / 3.0) * np.pi * (r**3)
    target_bulk_volume = effective_collision_volume / params["target_packing_fraction"]
    base_radius_m = ((3.0 * target_bulk_volume) / (np.pi * tan_theta)) ** (1.0 / 3.0)
    base_radius_m *= radius_expansion_factor
    pile_height_m = base_radius_m * tan_theta
    return {
        "theta_rad": float(theta_rad),
        "tan_theta": float(tan_theta),
        "base_radius_m": float(base_radius_m),
        "pile_height_m": float(pile_height_m),
        "effective_collision_volume_m3": float(effective_collision_volume),
        "target_bulk_volume_m3": float(target_bulk_volume),
    }


def _allowed_radial_distance_at_height(z_center_m: float, collision_radius_m: float, cone_geometry: dict) -> float:
    raw_radius = (cone_geometry["pile_height_m"] - z_center_m) / cone_geometry["tan_theta"]
    return float(max(raw_radius - 0.35 * collision_radius_m, 0.0))


def _compute_settled_z_at_xy(
    x_new: float,
    y_new: float,
    ground_clearance_m: float,
    collision_radius_new: float,
    placed_records: list[dict],
    separation_factor: float,
) -> tuple[float, int]:
    z_settled = float(ground_clearance_m)
    support_candidates = []
    for placed in placed_records:
        dx = x_new - placed["x_center_m"]
        dy = y_new - placed["y_center_m"]
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        required_distance = separation_factor * (collision_radius_new + placed["collision_radius_m"])
        if horizontal_distance >= required_distance:
            continue
        vertical_offset = np.sqrt(max(required_distance**2 - horizontal_distance**2, 0.0))
        support_candidates.append(placed["z_center_m"] + vertical_offset)
    if support_candidates:
        z_contact = max(support_candidates)
        if z_contact > z_settled:
            z_settled = float(z_contact)
    support_count = 0
    if support_candidates and z_settled > ground_clearance_m:
        support_count = sum(np.isclose(candidate_z, z_settled, atol=1e-5) for candidate_z in support_candidates)
    return float(z_settled), int(support_count)


def _inside_cone(x_center_m: float, y_center_m: float, z_center_m: float, collision_radius_m: float, cone_geometry: dict) -> bool:
    radial_distance_m = np.sqrt(x_center_m**2 + y_center_m**2)
    allowed_radius_m = _allowed_radial_distance_at_height(z_center_m, collision_radius_m, cone_geometry)
    return radial_distance_m <= allowed_radius_m


def _attempt_drop_and_settle(prepared_fragments: list[dict], cone_geometry: dict, params: dict, rng: np.random.Generator) -> tuple[bool, list[dict]]:
    placed_records = []
    for frag_index, frag in enumerate(prepared_fragments):
        if frag_index == 0:
            placed_records.append({
                **frag,
                "x_center_m": 0.0,
                "y_center_m": 0.0,
                "z_center_m": float(frag["ground_clearance_m"]),
                "radial_distance_m": 0.0,
                "support_count": 0,
                "placement_score": 0.0,
            })
            continue
        candidate_records = []
        for _ in range(params["trials_per_fragment"]):
            x_candidate, y_candidate = _sample_xy_inside_disk(cone_geometry["base_radius_m"], rng)
            z_candidate, support_count = _compute_settled_z_at_xy(
                x_candidate,
                y_candidate,
                frag["ground_clearance_m"],
                frag["collision_radius_m"],
                placed_records,
                params["separation_factor"],
            )
            if not _inside_cone(x_candidate, y_candidate, z_candidate, frag["collision_radius_m"], cone_geometry):
                continue
            radial_distance_m = np.sqrt(x_candidate**2 + y_candidate**2)
            placement_score = z_candidate + params["radial_penalty_weight"] * radial_distance_m / cone_geometry["base_radius_m"]
            candidate_records.append({
                "x_center_m": float(x_candidate),
                "y_center_m": float(y_candidate),
                "z_center_m": float(z_candidate),
                "radial_distance_m": float(radial_distance_m),
                "support_count": int(support_count),
                "placement_score": float(placement_score),
            })
        if not candidate_records:
            return False, placed_records
        best_candidate = min(candidate_records, key=lambda c: c["placement_score"])
        placed_records.append({**frag, **best_candidate})
    return True, placed_records


def _prepare_fragments_from_catalog(catalog: pd.DataFrame, rng: np.random.Generator, n_fragments: int) -> list[dict]:
    selected = catalog.sample(n=n_fragments, replace=False, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    selected = selected.sort_values("equivalent_diameter_m", ascending=False).reset_index(drop=True)
    prepared = []
    for local_id, row in selected.iterrows():
        mesh = trimesh.load_mesh(row["mesh_abs_path"], process=True)
        mesh_rotated, rotation_angles = _apply_random_rotation(mesh, rng)
        mesh_centered = _recenter_mesh(mesh_rotated)
        bounding_radius_m = _estimate_bounding_radius(mesh_centered)
        collision_radius_m = bounding_radius_m * DEFAULT_SYNTHETIC_ROCKPILE_PARAMS["collision_radius_scale"]
        ground_clearance_m = _compute_ground_clearance(mesh_centered)
        prepared.append({
            "local_fragment_id": int(local_id),
            "source_fragment_id": int(row["fragment_id"]),
            "mesh": mesh_centered,
            "source_mesh_path": str(row["mesh_abs_path"]),
            "bounding_radius_m": bounding_radius_m,
            "collision_radius_m": collision_radius_m,
            "ground_clearance_m": ground_clearance_m,
            "rotation_x_rad": rotation_angles[0],
            "rotation_y_rad": rotation_angles[1],
            "rotation_z_rad": rotation_angles[2],
            "equivalent_diameter_m": float(row["equivalent_diameter_m"]),
            "volume_m3": float(row["volume_m3"]),
        })
    return prepared


def generate_synthetic_rockpile_v1_scene(
    catalog: pd.DataFrame,
    scene_id: int,
    rng: np.random.Generator,
    n_fragments: int = 220,
    total_surface_points: int = 45_000,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Generate a scene using the cone/drop-and-settle logic from Synthetic_Rockpile notebook 02.

    This mirrors the v1 Synthetic_Rockpile pile-generation heuristic while
    keeping all generated outputs inside the caller repository.
    """

    params_merged = dict(DEFAULT_SYNTHETIC_ROCKPILE_PARAMS)
    if params:
        params_merged.update(params)
    prepared = _prepare_fragments_from_catalog(catalog, rng, n_fragments)
    placement_success = False
    final_placed_records = None
    final_cone_geometry = None
    for expansion_step in range(params_merged["max_expansion_steps"]):
        radius_expansion_factor = params_merged["radius_expansion_factor_per_step"] ** expansion_step
        cone_geometry = _estimate_cone_geometry(prepared, params_merged, radius_expansion_factor)
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + expansion_step)
        success, placed_records = _attempt_drop_and_settle(prepared, cone_geometry, params_merged, local_rng)
        if success:
            placement_success = True
            final_placed_records = placed_records
            final_cone_geometry = cone_geometry
            break
    if not placement_success:
        raise RuntimeError("Could not place all fragments with Synthetic_Rockpile v1 cone/drop heuristic.")

    volumes = np.array([frag["volume_m3"] for frag in final_placed_records], dtype=float)
    weights = volumes / volumes.sum()
    counts = np.maximum(24, np.round(total_surface_points * weights).astype(int))
    counts = np.maximum(24, np.round(counts * (total_surface_points / counts.sum())).astype(int))

    point_parts = []
    label_parts = []
    rows = []
    for new_label, (placed, n_points) in enumerate(zip(final_placed_records, counts)):
        mesh_placed = placed["mesh"].copy()
        mesh_placed.apply_translation([placed["x_center_m"], placed["y_center_m"], placed["z_center_m"]])
        pts, _ = trimesh.sample.sample_surface(mesh_placed, int(n_points))
        pts = np.asarray(pts, dtype=np.float32)
        point_parts.append(pts)
        label_parts.append(np.full(len(pts), new_label, dtype=np.int32))
        rows.append({
            "scene_id": int(scene_id),
            "local_fragment_id": int(new_label),
            "source_fragment_id": int(placed["source_fragment_id"]),
            "source_mesh_path": placed["source_mesh_path"],
            "x_center_m": float(placed["x_center_m"]),
            "y_center_m": float(placed["y_center_m"]),
            "z_center_m": float(placed["z_center_m"]),
            "radial_distance_m": float(placed["radial_distance_m"]),
            "support_count": int(placed["support_count"]),
            "placement_score": float(placed["placement_score"]),
            "bounding_radius_m": float(placed["bounding_radius_m"]),
            "collision_radius_m": float(placed["collision_radius_m"]),
            "ground_clearance_m": float(placed["ground_clearance_m"]),
            "rotation_x_rad": float(placed["rotation_x_rad"]),
            "rotation_y_rad": float(placed["rotation_y_rad"]),
            "rotation_z_rad": float(placed["rotation_z_rad"]),
            "volume_m3": float(placed["volume_m3"]),
            "equivalent_diameter_m": float(placed["equivalent_diameter_m"]),
            "equivalent_diameter_mm": float(placed["equivalent_diameter_m"] * 1000),
            "n_surface_points": int(len(pts)),
        })

    points = np.vstack(point_parts).astype(np.float32)
    labels = np.concatenate(label_parts).astype(np.int32)
    metadata = pd.DataFrame(rows)
    return points, labels, metadata, final_cone_geometry


def _load_realistic_dem_api(generator_root: Path = REALISTIC_GENERATOR_ROOT):
    src_path = Path(generator_root) / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Realistic generator src directory not found: {src_path}")
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    from realistic_rockpile.dem import DEMParams, generate_realistic_dem_scene

    return DEMParams, generate_realistic_dem_scene


def _load_chrono_muckpile_api(generator_root: Path = REALISTIC_GENERATOR_ROOT):
    src_path = Path(generator_root) / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Realistic generator src directory not found: {src_path}")
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    from realistic_rockpile.chrono_pile import ChronoMuckpileParams, generate_chrono_muckpile_scene

    return ChronoMuckpileParams, generate_chrono_muckpile_scene


def generate_physics_informed_dem_scene(
    catalog: pd.DataFrame,
    scene_id: int,
    rng: np.random.Generator,
    n_fragments: int = 220,
    total_surface_points: int = 45_000,
    params: dict | None = None,
    generator_root: Path = REALISTIC_GENERATOR_ROOT,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Generate a scene using the physics-informed realistic rockpile DEM backend."""

    DEMParams, generate_realistic_dem_scene = _load_realistic_dem_api(generator_root)
    dem_params = DEMParams(**(params or {}))
    return generate_realistic_dem_scene(
        catalog,
        scene_id=scene_id,
        rng=rng,
        n_fragments=n_fragments,
        total_surface_points=total_surface_points,
        params=dem_params,
    )


def generate_chrono_dem_scene(
    catalog: pd.DataFrame,
    scene_id: int,
    rng: np.random.Generator,
    n_fragments: int = 220,
    total_surface_points: int = 45_000,
    params: dict | None = None,
    generator_root: Path = REALISTIC_GENERATOR_ROOT,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Generate a scene using Project Chrono as the muckpile dynamics backend."""

    ChronoMuckpileParams, generate_chrono_muckpile_scene = _load_chrono_muckpile_api(generator_root)
    chrono_params = ChronoMuckpileParams(total_surface_points=total_surface_points, **(params or {}))
    return generate_chrono_muckpile_scene(
        catalog,
        scene_id=scene_id,
        rng=rng,
        n_fragments=n_fragments,
        total_surface_points=total_surface_points,
        params=chrono_params,
    )
