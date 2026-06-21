from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_METADATA = Path(
    r"C:/Users/creep/code/python/dnn-rockpile-affinity-psd/data/processed/"
    r"scene_dem_noboundary_relax150_100scene_000_train_fragments.csv"
)
DEFAULT_SCENE = Path(
    r"C:/Users/creep/code/python/dnn-rockpile-affinity-psd/data/processed/"
    r"scene_dem_noboundary_relax150_100scene_000_train.npz"
)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.clip(np.linalg.norm(q), 1e-12, None)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def load_meshes(metadata_path: Path, max_faces_per_fragment: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    metadata = pd.read_csv(metadata_path)
    polygons: list[np.ndarray] = []
    centers = metadata[["x_center_m", "y_center_m", "z_center_m"]].to_numpy(float)
    for _, row in metadata.iterrows():
        mesh = trimesh.load_mesh(row["source_mesh_path"], process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        vertices = np.asarray(mesh.vertices, dtype=float)
        vertices = vertices - vertices.mean(axis=0, keepdims=True)
        rot = quaternion_to_matrix(
            np.array(
                [row["orientation_qw"], row["orientation_qx"], row["orientation_qy"], row["orientation_qz"]],
                dtype=float,
            )
        )
        vertices = vertices @ rot.T + np.array([row["x_center_m"], row["y_center_m"], row["z_center_m"]], dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        if len(faces) > max_faces_per_fragment:
            faces = faces[rng.choice(len(faces), size=max_faces_per_fragment, replace=False)]
        polygons.extend(vertices[faces])
    return polygons, centers


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max()) * 1.10
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(max(-0.03, mins[2] - 0.04), maxs[2] + 0.08)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.grid(False)


def label_colors(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    lut = {label: i for i, label in enumerate(unique)}
    return plt.cm.tab20(np.array([lut[label] % 20 for label in labels]) / 19.0)


def render(metadata_path: Path, scene_path: Path, output_path: Path) -> None:
    polygons, centers = load_meshes(metadata_path, max_faces_per_fragment=40, seed=7)
    data = np.load(scene_path)
    exterior_points = np.asarray(data["points_xyz"], dtype=float)
    exterior_labels = np.asarray(data["instance_labels"], dtype=int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.0, 5.3), dpi=180)

    ax_mesh = fig.add_subplot(1, 2, 1, projection="3d")
    collection = Poly3DCollection(
        polygons,
        facecolors=(0.53, 0.57, 0.58, 0.42),
        edgecolors=(0.15, 0.17, 0.18, 0.11),
        linewidths=0.08,
    )
    ax_mesh.add_collection3d(collection)
    set_equal_axes(ax_mesh, centers)
    ax_mesh.view_init(elev=26, azim=-43)
    ax_mesh.set_title("A. Fragment mesh placement")

    ax_ext = fig.add_subplot(1, 2, 2, projection="3d")
    if len(exterior_points) > 9000:
        rng = np.random.default_rng(11)
        keep = rng.choice(len(exterior_points), size=9000, replace=False)
        exterior_points_plot = exterior_points[keep]
        exterior_labels_plot = exterior_labels[keep]
    else:
        exterior_points_plot = exterior_points
        exterior_labels_plot = exterior_labels
    ax_ext.scatter(
        exterior_points_plot[:, 0],
        exterior_points_plot[:, 1],
        exterior_points_plot[:, 2],
        c=label_colors(exterior_labels_plot),
        s=1.4,
        alpha=0.90,
        linewidths=0,
    )
    set_equal_axes(ax_ext, exterior_points)
    ax_ext.view_init(elev=26, azim=-43)
    ax_ext.set_title("B. Exterior-only labelled points")

    fig.suptitle("Scene 000: mesh-level pile visualisation and exterior scan target", y=0.98, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.metadata, args.scene, args.output)


if __name__ == "__main__":
    main()
