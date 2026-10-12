import numpy as np
from collections import defaultdict
import open3d as o3d
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
    mesh = utils.clean_mesh(mesh)               # merge seams, remove junk
    mesh = utils.fix_non_manifold(mesh)         # remove non-manifold edges/vertices
    # Do NOT call clean_mesh again here: it would re-merge the vertices that
    # the non-manifold fix just split apart, and bring the problem back.
    mesh = utils.orient_consistently(mesh)      # make all windings agree
    mesh, n_holes = utils.fill_holes(mesh)      # close holes, including the ones the fix created
    mesh = utils.orient_consistently(mesh)      # re-check outward direction once closed
    return mesh, n_holes

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

# # stats = []
# # for folder, name in items:
# #     mesh = o3d.io.read_triangle_mesh(os.path.join(DATABASE_DIR, folder, name))
# #     if not mesh.has_triangles():
# #         continue
# #     mesh, n_holes = repair_mesh(mesh)
# #     if not is_closed_and_consistent(mesh):
# #         o, d = failure_reasons(mesh)
# #         stats.append((folder, name, n_holes, o, d))

# # print("still open only:", sum(1 for s in stats if s[3] > 0 and s[4] == 0))
# # print("duplicate direction only:", sum(1 for s in stats if s[3] == 0 and s[4] > 0))
# # print("both:", sum(1 for s in stats if s[3] > 0 and s[4] > 0))
# # holes = sorted(s[2] for s in stats)
# # print("holes filled in unreliable meshes: median", holes[len(holes)//2], "max", holes[-1])

# import random

# random.seed(0)                                   # same seed as the check script
# sample = random.sample(items, 200)               # `items` = list of (folder, name) as before

# def old_repair(mesh):
#     mesh = utils.clean_mesh(mesh)
#     mesh = utils.orient_consistently(mesh)
#     mesh, n = utils.fill_holes(mesh)
#     return utils.orient_consistently(mesh), n

# old_ok = new_ok = 0
# changes, holes_new = [], []
# for folder, name in sample:
#     path = os.path.join(DATABASE_DIR, folder, name)
#     base = o3d.io.read_triangle_mesh(path)
#     if not base.has_triangles():
#         continue
#     m_old, _ = old_repair(o3d.geometry.TriangleMesh(base))
#     m_new, n_holes = repair_mesh(o3d.geometry.TriangleMesh(base))
#     ok_old, ok_new = is_closed_and_consistent(m_old), is_closed_and_consistent(m_new)
#     old_ok += ok_old
#     new_ok += ok_new
#     holes_new.append(n_holes)
#     if ok_old and ok_new:                        # volume change on meshes reliable in both
#         v0, v1 = mesh_volume(m_old), mesh_volume(m_new)
#         if v0 > 0:
#             changes.append(abs(v1 - v0) / v0)

# print(f"reliable before: {old_ok}, after: {new_ok}")
# print(f"holes filled after fix: median {sorted(holes_new)[len(holes_new)//2]}, max {max(holes_new)}")
# if changes:
#     print(f"volume change on meshes reliable in both: median {np.median(changes):.2%}, max {max(changes):.2%}")

def volume_by_occupancy(mesh, n=64, nsamples=3):
    """Approximate volume by counting grid points that lie inside the shape."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    bbox = mesh.get_axis_aligned_bounding_box()
    centre = bbox.get_center()
    side = float(np.max(bbox.get_extent()))
    if side == 0:
        return 0.0
    step = side / n
    axis = (np.arange(n) + 0.5) * step - side / 2
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = (np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) + centre).astype(np.float32)

    try:
        inside = scene.compute_occupancy(o3d.core.Tensor(grid), nsamples=nsamples)
    except TypeError:                       # older Open3D without nsamples
        inside = scene.compute_occupancy(o3d.core.Tensor(grid))
    return float(inside.numpy().sum() * step ** 3)


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

random.seed(0)
if len(items) > LIMIT:
    items = random.sample(items, LIMIT)

rows = []
t0 = time.time()
for i, (folder, name) in enumerate(items, start=1):
    base = o3d.io.read_triangle_mesh(os.path.join(DATABASE_DIR, folder, name))
    if not base.has_triangles():
        rows.append(dict(folder=folder, name=name, status="load failed"))
        continue

    # Occupancy volume on the merely cleaned mesh (no repair, no fans)
    cleaned = utils.clean_mesh(o3d.geometry.TriangleMesh(base))
    v_occ = volume_by_occupancy(cleaned)
    a_clean = surface_area(cleaned)

    # Exact volume on the repaired mesh
    fixed, n_holes = repair_mesh(o3d.geometry.TriangleMesh(base))
    closed = is_closed_and_consistent(fixed)
    v_exact = mesh_volume(fixed)
    a_fixed = surface_area(fixed)
    o, d = (0, 0) if closed else failure_reasons(fixed)

    c_exact = 36 * np.pi * v_exact**2 / a_fixed**3 if a_fixed > 0 else 0.0
    c_occ = 36 * np.pi * v_occ**2 / a_clean**3 if a_clean > 0 else 0.0
    rel_diff = abs(v_occ - v_exact) / v_exact if v_exact > 0 else float("nan")

    rows.append(dict(folder=folder, name=name, status="ok", closed=closed,
                     n_holes=n_holes, open_edges=o, dup_dir=d,
                     v_exact=v_exact, v_occ=v_occ, rel_diff=rel_diff,
                     c_exact=c_exact, c_occ=c_occ))
    if i % 20 == 0:
        print(f"{i}/{len(items)} done ({time.time() - t0:.0f}s)")

# Save everything so you can inspect single meshes in the viewer
ok_rows = [r for r in rows if r["status"] == "ok"]
with open("final_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ok_rows[0].keys()))
    w.writeheader()
    w.writerows(ok_rows)

# ---- Summary ----
closed_rows = [r for r in ok_rows if r["closed"]]
trusted = [r for r in closed_rows if r["n_holes"] <= MAX_HOLES]
agree = [r for r in trusted if r["rel_diff"] <= AGREE_TOL]
failed = [r for r in ok_rows if not r["closed"]]

print(f"\nMeshes tested: {len(ok_rows)} ({len(rows) - len(ok_rows)} failed to load)")
print(f"Closed + consistent: {len(closed_rows)}")
print(f"  ...and <= {MAX_HOLES} holes filled (trusted): {len(trusted)}")
print(f"  ...and exact vs occupancy within {AGREE_TOL:.0%}: {len(agree)}")

if trusted:
    diffs = np.array([r["rel_diff"] for r in trusted])
    print(f"Volume difference (trusted meshes): median {np.median(diffs):.2%}, "
          f"90th percentile {np.percentile(diffs, 90):.2%}")
    print(f"Compactness > 1 among trusted: {sum(r['c_exact'] > 1 for r in trusted)} (should be ~0)")

if failed:
    print(f"\nStill failing: {len(failed)}")
    print("  open edges only:", sum(r['open_edges'] > 0 and r['dup_dir'] == 0 for r in failed))
    print("  duplicate direction only:", sum(r['open_edges'] == 0 and r['dup_dir'] > 0 for r in failed))
    print("  both:", sum(r['open_edges'] > 0 and r['dup_dir'] > 0 for r in failed))

print("\nWorst exact-vs-occupancy disagreements (open these in the viewer):")
for r in sorted(trusted, key=lambda r: -r["rel_diff"])[:5]:
    print(f"  {r['folder']}/{r['name']}: diff {r['rel_diff']:.1%}, holes {r['n_holes']}")