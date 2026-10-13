import numpy as np
from collections import defaultdict
import open3d as o3d
import pymeshlab
import mesh_browser.mesh_utils as utils
import os
from mesh_browser.config import DATABASE_DIR
import random
import time
import csv

def is_closed_and_consistent(mesh):
    t = np.asarray(mesh.triangles)
    count = defaultdict(int)
    for a, b, c in t:
        for u, w in ((a, b), (b, c), (c, a)):
            count[(u, w)] += 1
    return all(n == 1 and count.get((w, u), 0) == 1 for (u, w), n in count.items())

def repair_mesh(mesh):
    mesh = utils.clean_mesh(mesh)                 # merge seams, remove junk
    mesh = utils.fix_non_manifold(mesh)           # MeshLab needs manifold input to close holes
    mesh = utils.orient_consistently(mesh)
    mesh = utils.meshlab_close_holes(mesh)        # pass 1: MeshLab closes the holes it can
    mesh = utils.orient_consistently(mesh)
    mesh, n_holes = utils.fill_holes(mesh)        # pass 2: fan for whatever is still open
    mesh = utils.orient_consistently(mesh)        # pass 3: final orientation
    return mesh, n_holes                          # n_holes = holes the fan still had to fill

def surface_area(mesh):
    """Total surface area: the sum of all triangle areas."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())


def mesh_volume(mesh):
    """Enclosed volume via signed tetrahedra. Only meaningful for closed,
    consistently oriented meshes."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    signed = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    return float(abs(signed.sum()))   # abs: the sign depends on the winding direction


def compactness(mesh):
    """36*pi*V^2 / A^3. A sphere gives 1, less compact shapes give less."""
    area = surface_area(mesh)
    if area == 0:
        return 0.0
    volume = mesh_volume(mesh)
    return float(36.0 * np.pi * volume ** 2 / area ** 3)

def check(name, mesh, expected_volume=None):
    mesh, n_holes = repair_mesh(mesh)
    ok = is_closed_and_consistent(mesh)
    V = mesh_volume(mesh)
    A = surface_area(mesh)
    C = 36.0 * np.pi * V ** 2 / A ** 3 if A > 0 else 0.0
    print(f"{name}: holes filled={n_holes}, closed+consistent={ok}, "
          f"V={V:.4f} (expected {expected_volume}), A={A:.4f}, compactness={C:.4f}")

# 1. Intact shapes. Sphere: V≈4.19 (slightly less from discretisation), compactness ≈ 1.
#    Unit cube: V=1, compactness = 36π/216 ≈ 0.5236.
# check("sphere", o3d.geometry.TriangleMesh.create_sphere(1.0, 40), 4.18879)
# check("cube", o3d.geometry.TriangleMesh.create_box(1, 1, 1), 1.0)

# # 2. Flip random triangles: orientation should fix it, result should match test 1
# m = o3d.geometry.TriangleMesh.create_sphere(1.0, 40)
# t = np.asarray(m.triangles).copy()
# flip = np.random.default_rng(0).random(len(t)) < 0.3
# t[flip] = t[flip, ::-1]
# m.triangles = o3d.utility.Vector3iVector(t)
# check("sphere, 30% flipped", m, 4.18879)

# # 3. Punch a hole: remove a few triangles, the fan should close it
# m = o3d.geometry.TriangleMesh.create_sphere(1.0, 40)
# m.remove_triangles_by_index(list(range(0, 8)))   # near the pole, so likely adjacent
# check("sphere with hole", m, 4.18879)
# m = o3d.geometry.TriangleMesh.create_sphere(1.0, 40)
# v = np.asarray(m.vertices)
# t = np.asarray(m.triangles)
# cap = v[t][:, :, 2].mean(axis=1) > 0.7          # triangles whose centre is above z = 0.7
# m.remove_triangles_by_mask(cap)
# check("sphere, cap removed", m, "about 3.92")

# LIMIT = 200
# EXTENSIONS = (".obj", ".off", ".ply", ".stl")

# # Collect all meshes, then take a random sample so the 200 aren't all from one class
# items = []
# for folder in sorted(os.listdir(DATABASE_DIR)):
#     folder_path = os.path.join(DATABASE_DIR, folder)
#     if not os.path.isdir(folder_path):
#         continue
#     for name in sorted(os.listdir(folder_path)):
#         if name.lower().endswith(EXTENSIONS):
#             items.append((folder, name))

# random.seed(0)
# if len(items) > LIMIT:
#     items = random.sample(items, LIMIT)
#     # items = items[:LIMIT]

# reliable = unreliable = 0
# gt1_reliable, gt1_unreliable, holes_filled = [], [], 0

# # for folder, name in items:                      # same sample as your check script
# #     mesh = o3d.io.read_triangle_mesh(os.path.join(DATABASE_DIR, folder, name))
# #     if not mesh.has_triangles():
# #         continue
# #     mesh, n_holes = repair_mesh(mesh)
# #     holes_filled += n_holes
# #     c = compactness(mesh)

# #     if is_closed_and_consistent(mesh):
# #         reliable += 1
# #         if c > 1.0:
# #             gt1_reliable.append((folder, name, c))
# #     else:
# #         unreliable += 1
# #         if c > 1.0:
# #             gt1_unreliable.append((folder, name, c))

# # print(f"reliable: {reliable}, unreliable: {unreliable}, holes filled: {holes_filled}")
# # print(f"compactness > 1 among reliable: {len(gt1_reliable)}  (should be ~0)")
# # print(f"compactness > 1 among unreliable: {len(gt1_unreliable)}")
def failure_reasons(mesh):
    t = np.asarray(mesh.triangles)
    count = defaultdict(int)
    for a, b, c in t:
        for u, w in ((a, b), (b, c), (c, a)):
            count[(u, w)] += 1
    open_edges = sum(1 for (u, w) in count if (w, u) not in count)   # still holes
    dup_dir = sum(1 for n in count.values() if n > 1)                # same direction twice
    return open_edges, dup_dir

def compactness_final(mesh):
    """Returns (compactness or None, status)."""
    fixed, n_holes = repair_mesh(o3d.geometry.TriangleMesh(mesh))
    if not is_closed_and_consistent(fixed):
        return None, "not closed"
    a = surface_area(fixed)
    if a <= 0:
        return None, "zero area"
    c = 36 * np.pi * mesh_volume(fixed) ** 2 / a ** 3
    if not (0.0 <= c <= 1.0):
        return None, "implausible"
    return float(c), "ok"


LIMIT = 200
EXTENSIONS = (".obj", ".off", ".ply", ".stl")
MAX_HOLES = 50          # more fans than this = mostly junk geometry, don't trust the volume
AGREE_TOL = 0.05        # exact vs occupancy volume must agree within 5%

items = []
for folder in sorted(os.listdir(DATABASE_DIR)):
    folder_path = os.path.join(DATABASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    for name in sorted(os.listdir(folder_path)):
        if name.lower().endswith(EXTENSIONS):
            items.append((folder, name))

def edge_report(mesh):
    """Per undirected edge: how many triangles use it in each direction."""
    t = np.asarray(mesh.triangles)
    fwd = defaultdict(int)
    for a, b, c in t:
        for u, w in ((a, b), (b, c), (c, a)):
            fwd[(u, w)] += 1
    seen = set()
    boundary = nonmanifold = samedir = unbalanced = 0
    for (u, w) in list(fwd):
        key = (min(u, w), max(u, w))
        if key in seen:
            continue
        seen.add(key)
        f, r = fwd.get(key, 0), fwd.get((key[1], key[0]), 0)
        if f != r:
            unbalanced += 1
        if f + r == 1:
            boundary += 1            # still a hole edge
        elif f + r >= 3:
            nonmanifold += 1         # 3+ triangles on one edge
        elif f != 1:
            samedir += 1             # exactly 2 triangles but both go the same way
    return dict(boundary=boundary, nonmanifold=nonmanifold,
                samedir=samedir, unbalanced=unbalanced)


def volume_centered(mesh):
    """Signed-tetrahedra volume with the origin moved to the vertex mean.
    For a balanced mesh the origin doesn't matter; for an open one it reduces the error."""
    m = o3d.geometry.TriangleMesh(mesh)
    m.translate(-np.asarray(m.vertices).mean(axis=0))
    return mesh_volume(m)


os.makedirs("debug_failed", exist_ok=True)
rows, saved = [], 0
for folder, name in items:
    base = o3d.io.read_triangle_mesh(os.path.join(DATABASE_DIR, folder, name))
    if not base.has_triangles():
        continue
    fixed, n_holes = repair_mesh(o3d.geometry.TriangleMesh(base))
    closed = is_closed_and_consistent(fixed)
    rep = edge_report(fixed)
    a = surface_area(fixed)
    v = volume_centered(fixed)
    c = 36 * np.pi * v**2 / a**3 if a > 0 else float("nan")
    rows.append(dict(folder=folder, name=name, closed=closed,
                     balanced=(rep["unbalanced"] == 0), c=c,
                     faces=len(fixed.triangles), **rep))
    if not closed and saved < 8:
        out_dir = os.path.join("debug_failed", folder)
        os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(out_dir, name), fixed)
        saved += 1

failed = [r for r in rows if not r["closed"]]
print(f"closed+consistent: {len(rows) - len(failed)}/{len(rows)}")
print(f"balanced (looser test): {sum(r['balanced'] for r in rows)}/{len(rows)}")
print(f"\nOf the {len(failed)} that fail the strict test:")
print("  balanced anyway:", sum(r["balanced"] for r in failed))
print("  have boundary edges (holes):", sum(r["boundary"] > 0 for r in failed))
print("  have non-manifold edges:", sum(r["nonmanifold"] > 0 for r in failed))
print("  have same-direction pairs:", sum(r["samedir"] > 0 for r in failed))
print("  compactness in [0,1] when computed anyway:", sum(0 <= r["c"] <= 1 for r in failed))
print("  compactness > 1:", sum(r["c"] > 1 for r in failed))

print("\nSamples of failed meshes (also saved to debug_failed/):")
for r in failed[:8]:
    print(f"  {r['folder']}/{r['name']}: faces {r['faces']}, boundary {r['boundary']}, "
          f"nonmanifold {r['nonmanifold']}, samedir {r['samedir']}, "
          f"balanced {r['balanced']}, c={r['c']:.3f}")