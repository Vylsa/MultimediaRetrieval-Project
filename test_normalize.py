'''Run this file to test the mesh normalizing functionality. Set the MESH_PATH to the specified mesh and the OUTPUT_DIR to the folder where you want the resampled mesh to be saved.'''
import os

import open3d as o3d

from mesh_browser.mesh_utils import load_mesh, clean_mesh, normalize_mesh, resample_mesh, mesh_stats

from mesh_browser.config import DATABASE_DIR, NORMALIZED_DATABASE_DIR




MESH_PATH = os.path.join(DATABASE_DIR, "0_resampling_tests", "sphere.obj")

OUTPUT_DIR = os.path.join(NORMALIZED_DATABASE_DIR, "0_normalizing_tests")

mesh = load_mesh(MESH_PATH)
if mesh is None:
    raise SystemExit(f"Could not load {MESH_PATH}")

mesh = clean_mesh(mesh)
v_before, t_before = mesh_stats(mesh)
mesh = normalize_mesh(mesh)
v_after, t_after = mesh_stats(mesh)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, os.path.basename(MESH_PATH))
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(out_path, mesh)


print(f"Saved to {out_path}")