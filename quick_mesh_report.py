import os
import random
import open3d as o3d
import numpy as np

DATABASE_DIR = "normalized_database"   # <- point this at your database
LIMIT = 1
EXTENSIONS = (".obj", ".off", ".ply", ".stl")

# Collect all meshes, then take a random sample so the 200 aren't all from one class
items = []
for folder in sorted(os.listdir(DATABASE_DIR)):
    folder_path = os.path.join(DATABASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(EXTENSIONS):
            items.append((folder, name))

# random.seed(0)
if len(items) > LIMIT:
    # items = random.sample(items, LIMIT)
    items = items[:LIMIT]   # take first N instead of random sample, for reproducibility

def merge_with_tolerance(mesh, decimals=6):
    """Merge vertices whose positions agree after rounding to `decimals` decimals."""
    v = np.round(np.asarray(mesh.vertices), decimals)
    uniq, inverse = np.unique(v, axis=0, return_inverse=True)
    t = inverse.reshape(-1)[np.asarray(mesh.triangles)]
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(uniq)
    out.triangles = o3d.utility.Vector3iVector(t.astype(np.int32))
    out.remove_duplicated_triangles()
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    return out

problems = []   # meshes with problems that hole filling will NOT fix
with_holes = 0
load_failed = 0
issue_self_intersecting = []
issue_non_manifold_edges = []
issue_non_manifold_vertices = []
issue_non_orientable = []

for i, (folder, name) in enumerate(items, start=1):
    mesh = o3d.io.read_triangle_mesh(os.path.join(DATABASE_DIR, folder, name))
    if not mesh.has_triangles():
        load_failed += 1
        print(f"[{i}/{len(items)}] load failed: {folder}/{name}")
        continue

    # Merge duplicated vertices first, otherwise every UV/normal seam looks like a
    # crack and nearly every mesh fails the manifold checks
    # mesh.remove_duplicated_vertices()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_degenerate_triangles()
    # mesh.remove_unreferenced_vertices()
    mesh = merge_with_tolerance(mesh, decimals=6)


    # allow_boundary_edges=True: open boundaries (holes) are fine here, since we want
    # to find what pymeshlab can't fill
    issues = []
    if not mesh.is_edge_manifold(allow_boundary_edges=True):
        issues.append("not edge manifold")
        issue_non_manifold_edges.append("not edge manifold")
    if not mesh.is_vertex_manifold():
        issues.append("not vertex manifold")
        issue_non_manifold_vertices.append("not vertex manifold")
    if not mesh.is_orientable():
        issues.append("not orientable")
        issue_non_orientable.append("not orientable")
    if mesh.is_self_intersecting():     # slow on big meshes
        issues.append("self-intersecting")
        issue_self_intersecting.append("self-intersecting")

    if not mesh.is_edge_manifold(allow_boundary_edges=False):
        with_holes += 1                 # has boundary edges (and/or non-manifold edges)

    if issues:
        problems.append((folder, name, issues))
        print(f"[{i}/{len(items)}] {folder}/{name}: {', '.join(issues)}")

print(f"\nChecked {len(items)} meshes, {load_failed} failed to load")
print(f"{len(problems)} have problems that hole filling won't fix")
print(f"{with_holes} have boundary edges (holes) or non-manifold edges")
print(f"{len(issue_self_intersecting)} are self-intersecting")
print(f"{len(issue_non_manifold_edges)} have non-manifold edges")
print(f"{len(issue_non_manifold_vertices)} have non-manifold vertices")
print(f"{len(issue_non_orientable)} are non-orientable")