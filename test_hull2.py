"""Check MeshDescriptors on shapes whose descriptor values are known exactly:
a unit cube, a 1 x 0.5 x 0.25 box and a sphere (the sphere only approximately, because
the mesh is a polygon approximation).
Then test the convex hull retry on a cylinder and on the real mesh m465 (a cylinder-shaped
apartment building whose convexity came out as None).

Run from the project root:  python test_unit_cube.py
"""
import os

import numpy as np
import open3d as o3d

from mesh_browser import config
import mesh_browser.mesh_utils as utils
from mesh_browser.descriptors1 import MeshDescriptors, mesh_volume


def box(w, h, d):
    mesh = o3d.geometry.TriangleMesh.create_box(w, h, d)
    mesh.translate(-np.array([w, h, d]) / 2.0)             # centre it on the origin
    return mesh


def expected_box(w, h, d):
    area = 2 * (w * h + w * d + h * d)
    volume = w * h * d
    return {
        "area": area,
        "volume": volume,
        "compactness": 36 * np.pi * volume ** 2 / area ** 3,
        "rectangularity": 1.0,
        "convexity": 1.0,
        "diameter": np.sqrt(w ** 2 + h ** 2 + d ** 2),
        "eccentricity": (max(w, h, d) / min(w, h, d)) ** 2,   # for the 8 corner vertices
    }


def expected_sphere(r):
    return {
        "area": 4 * np.pi * r ** 2,
        "volume": 4 / 3 * np.pi * r ** 3,
        "compactness": 1.0,
        "rectangularity": np.pi / 6,
        "convexity": 1.0,
        "diameter": 2 * r,
        "eccentricity": 1.0,
    }


# --- Part 1: known shapes ---------------------------------------------------

# (name, mesh, expected values, tolerance, per-descriptor tolerance overrides)
cases = [
    ("unit cube", box(1, 1, 1), expected_box(1, 1, 1), 1e-6, {}),
    ("box 1 x 0.5 x 0.25", box(1, 0.5, 0.25), expected_box(1, 0.5, 0.25), 1e-6, {}),
    ("sphere r = 0.5", o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=50),
     expected_sphere(0.5), 0.02, {"eccentricity": 0.2}),
]

passed = total = 0
for name, mesh, expected, tol, overrides in cases:
    d = MeshDescriptors(mesh)
    print(f"\n{name}   (quality: {d.quality})")
    print(f"  {'descriptor':16}{'got':>12}{'expected':>12}   result")
    for key, want in expected.items():
        got = getattr(d, key)
        limit = overrides.get(key, tol)
        ok = got is not None and abs(got - want) <= limit
        passed += int(ok)
        total += 1
        got_text = "None" if got is None else f"{got:.6f}"
        print(f"  {key:16}{got_text:>12}{want:>12.6f}   {'PASS' if ok else 'FAIL'}")

# Our volume function against Open3D's on the cube's hull
hull, _ = box(1, 1, 1).compute_convex_hull()
print(f"\nHull volume of the unit cube: ours = {mesh_volume(hull):.6f}, "
      f"Open3D = {hull.get_volume():.6f}  (both should be 1.0)")

print(f"\n{passed} of {total} checks passed")


# --- Part 2: convex hull retry ----------------------------------------------

def hull_with_retry(mesh):
    """compute_convex_hull, and if Qhull fails, retry with slightly perturbed points.
    Returns (hull or None, how) where how says which attempt worked."""
    try:
        return mesh.compute_convex_hull()[0], "plain"
    except RuntimeError:
        pass
    try:
        return mesh.compute_convex_hull(joggle_inputs=True)[0], "joggled (retry needed)"
    except (RuntimeError, TypeError) as e:     # TypeError: Open3D too old for joggle_inputs
        return None, f"FAILED ({type(e).__name__}: {e})"


def report_hull(name, mesh):
    hull, how = hull_with_retry(mesh)
    print(f"\n{name}: hull attempt = {how}")
    if hull is not None:
        print(f"  hull volume (ours):           {mesh_volume(hull):.6f}")
    d = MeshDescriptors(mesh)
    print(f"  MeshDescriptors.hull is None: {d.hull is None}")
    print(f"  MeshDescriptors.convexity:    {d.convexity}")
    if hull is not None and d.hull is None:
        print("  -> the retry builds a hull, but MeshDescriptors.hull does not use it yet")
    return d



cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.4, height=1.0, resolution=60)
d = report_hull("cylinder r = 0.4, h = 1 (convexity should be 1.0)", cylinder)

m465_path = os.path.join(config.NORMALIZED_DATABASE_DIR, "Apartment", "m465.obj")
mesh = utils.load_mesh(m465_path) if os.path.exists(m465_path) else None
if os.path.exists(m465_path):
    # report_hull("m465.obj (convexity was None)", utils.load_mesh(m465_path))
    # hull2, how = hull_with_retry(utils.load_mesh(m465_path))
    r = MeshDescriptors(mesh).repaired
    labels, counts, _ = r.cluster_connected_triangles()
    labels = np.asarray(labels)
    for i, n in enumerate(counts):
        part = o3d.geometry.TriangleMesh(r)
        part.remove_triangles_by_mask(labels != i)
        part.remove_unreferenced_vertices()
        print(i, n, "triangles, volume", mesh_volume(part))
    print("hull volume", mesh_volume(mesh.compute_convex_hull()[0]))

else:
    print(f"\nm465.obj not found at {m465_path}, skipped")