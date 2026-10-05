'''Run this file to test the mesh resampling functionality. Set the MESH_PATH to the specified mesh and the OUTPUT_DIR to the folder where you want the resampled mesh to be saved.'''
import os

import open3d as o3d

from mesh_browser.mesh_utils import load_mesh, clean_mesh, resample_mesh, mesh_stats

from mesh_browser.config import DATABASE_DIR




MESH_PATH = os.path.join(DATABASE_DIR, "Tree", "D00200.obj")

OUTPUT_DIR = os.path.join(DATABASE_DIR, "0_resampling_tests")

mesh = load_mesh(MESH_PATH)
if mesh is None:
    raise SystemExit(f"Could not load {MESH_PATH}")

mesh = clean_mesh(mesh)
v_before, t_before = mesh_stats(mesh)
mesh, actions = resample_mesh(mesh)
v_after, t_after = mesh_stats(mesh)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, os.path.basename(MESH_PATH))
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(out_path, mesh)

print(f"{v_before} -> {v_after} vertices, {t_before} -> {t_after} triangles "
      f"({', '.join(actions) if actions else 'no changes'})")
print(f"Saved to {out_path}")