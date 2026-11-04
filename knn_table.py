"""Results table for the report (Tech tips Step 5): one row per query shape, the query in
the first column and its K nearest neighbors next to it, sorted by distance, each with
its class and distance. Saves the table as an image.

The location of the feature database is set by FEATURES_PATH in mesh_browser/knn.py.
Set the settings below and run:
    python knn_table.py
"""
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser.config import NORMALIZED_DATABASE_DIR
from mesh_browser.knn import ShapeIndex

# --- Settings: edit these ---------------------------------------------------------

QUERY_SHAPES = ["m1337.obj", "D00138.obj", "D00066.obj", "m1593.obj", "m357.obj"]
K = 6
WEIGHTED = True                 # True: every feature counts equally; False: plain L2
SAVE_PATH = "knn_results_table.png"
THUMBNAIL_TRIANGLES = 3000      # meshes are simplified to this many triangles for drawing

# ------------------------------------------------------------------------------------

LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])

QUERY_COLOR = np.array([0.16, 0.47, 0.84])     # blue: the query shape
MATCH_COLOR = np.array([0.11, 0.69, 0.48])     # green: same class as the query
OTHER_COLOR = np.array([0.55, 0.55, 0.52])     # gray: different class


def load_thumbnail_mesh(name, cls):
    """Load a normalized mesh and simplify it, so drawing it with matplotlib stays fast."""
    mesh = o3d.io.read_triangle_mesh(os.path.join(NORMALIZED_DATABASE_DIR, cls, name))
    if len(mesh.triangles) > THUMBNAIL_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(THUMBNAIL_TRIANGLES)
    mesh.compute_triangle_normals()
    return mesh


def draw_mesh(ax, mesh, color):
    """Draw a mesh in a 3D axis with simple shading, same viewpoint for every shape."""
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    shade = 0.35 + 0.65 * np.abs(np.asarray(mesh.triangle_normals) @ LIGHT)
    faces = Poly3DCollection(v[t], facecolors=np.outer(shade, color), edgecolor="none")
    ax.add_collection3d(faces)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()


def make_table(index, queries, k):
    """One row per query: the query shape, then its k nearest neighbors (nearest first)."""
    fig = plt.figure(figsize=(2.1 * (k + 1), 2.3 * len(queries)))
    for row, name in enumerate(queries):
        cls = index.classes[index.names.index(name)]
        results = index.knn_by_name(name, k)
        cells = [(name, cls, None)] + results
        for col, (cell_name, cell_class, dist) in enumerate(cells):
            ax = fig.add_subplot(len(queries), k + 1, row * (k + 1) + col + 1, projection="3d")
            if col == 0:
                color = QUERY_COLOR
            else:
                color = MATCH_COLOR if cell_class == cls else OTHER_COLOR
            draw_mesh(ax, load_thumbnail_mesh(cell_name, cell_class), color)
            if dist is None:
                label = f"QUERY\n{cell_class}\n{cell_name}"
            else:
                label = f"{cell_class}\n{cell_name}\nd = {dist:.2f}"
            ax.set_title(label, fontsize=7, pad=0,
                         fontweight="bold" if dist is None else "normal")
        print(f"Row {row + 1}/{len(queries)} done: {name}")
    fig.suptitle(f"{k} nearest neighbors per query (k-d tree, "
                 f"{'weighted' if index.weighted else 'plain'} L2); "
                 "green = same class as the query", fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    index = ShapeIndex.from_file(weighted=WEIGHTED)
    print(f"Loaded {len(index.names)} shapes ({'weighted' if WEIGHTED else 'plain L2'})")

    missing = [q for q in QUERY_SHAPES if q not in index.names]
    if missing:
        raise SystemExit(f"Not in the feature database: {', '.join(missing)}. "
                         f"Check QUERY_SHAPES.")

    fig = make_table(index, QUERY_SHAPES, K)
    fig.savefig(SAVE_PATH, dpi=150)
    print(f"Saved to {SAVE_PATH}")
    plt.show()