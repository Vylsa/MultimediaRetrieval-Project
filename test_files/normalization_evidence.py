"""Statistical evidence that alignment and flipping work.

For every mesh, before (refined database) and after (normalized database) normalization:
  * alignment: the angle between the major eigenvector e1 and the x-axis, and between the
    medium eigenvector e2 and the y-axis. After alignment both should be ~0 degrees.
  * flipping: the three moments f_x, f_y, f_z (area-weighted, Equation 1 in the report),
    divided by sum(A_t * C_t,i^2) so they lie in [-1, 1] and can be compared between
    meshes. After flipping all three should be >= 0. Values within MOMENT_TOLERANCE of 0
    (symmetric shapes, where f is ~1e-12 rounding noise) count as 0.
  * the eigenvalue ratios lambda1/lambda2 and lambda2/lambda3: when a ratio is close to 1,
    the corresponding axes are not well defined (e.g. a sphere or a disk), which explains
    the few alignment "failures".
Writes normalization_evidence.xlsx (one row per mesh) and normalization_evidence.png.

Optionally (RUN_ROTATION_TEST) it also rotates a sample of meshes randomly, normalizes them
again, and measures how far the result is from the normalized original: if the
normalization works, the result should not depend on the starting orientation."""
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from openpyxl import Workbook

from pathlib import Path
import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser import config
from mesh_browser.mesh_utils import (align_mesh, center_mesh, list_meshes, list_subfolders,
                                     load_mesh, normalize_mesh, principal_axes)

AXIS_TOLERANCE_DEG = 1.0    # alignment counts as correct when the axis is within this angle
MOMENT_TOLERANCE = 1e-6     # relative moments within this distance of 0 count as 0
RUN_ROTATION_TEST = True
ROTATION_TEST_MESHES = 50   # how many random meshes for the rotation test
ROTATIONS_PER_MESH = 5
SEED = 0


def moments(mesh):
    """Area-weighted moments f_x, f_y, f_z (the moment test), divided by the total
    sum(A_t * C_t,i^2) per axis: -1 = all surface on the negative side, +1 = all positive."""
    v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    c = (p0 + p1 + p2) / 3.0
    f = np.sum(areas[:, None] * np.sign(c) * c ** 2, axis=0)
    total = np.sum(areas[:, None] * c ** 2, axis=0)
    return f / np.maximum(total, 1e-12)


def measure(mesh):
    """Alignment angles (degrees), moments and eigenvalue ratios of one mesh."""
    eigenvalues, eigenvectors = principal_axes(np.asarray(mesh.vertices))
    e1, e2 = eigenvectors[:, 0], eigenvectors[:, 1]
    angle_e1_x = np.degrees(np.arccos(np.clip(abs(e1[0]), 0, 1)))  # sign-free: e1 is a line
    angle_e2_y = np.degrees(np.arccos(np.clip(abs(e2[1]), 0, 1)))
    l1, l2, l3 = eigenvalues
    return {"angle_e1_x": angle_e1_x, "angle_e2_y": angle_e2_y,
            "f": moments(mesh),
            "ratio_12": l1 / max(l2, 1e-12), "ratio_23": l2 / max(l3, 1e-12)}


def moments_before_flip(mesh):
    """Moments of a copy of the mesh that is centered and aligned, but not yet flipped:
    the fair 'before' for the flipping step."""
    m = o3d.geometry.TriangleMesh(mesh)
    center_mesh(m)
    align_mesh(m)
    return moments(m)


def measure_database(database_dir, before_flip=False):
    rows = []
    items = [(f, m) for f in list_subfolders(database_dir)
             for m in list_meshes(os.path.join(database_dir, f))]
    for i, (folder, name) in enumerate(items, start=1):
        mesh = load_mesh(os.path.join(database_dir, folder, name))
        if mesh is not None:
            row = {"name": name, "class": folder, **measure(mesh)}
            if before_flip:
                row["f"] = moments_before_flip(mesh)
            rows.append(row)
        print(f"\r{database_dir}: {i}/{len(items)}", end="", flush=True)
    print()
    return rows


def summarize(label, rows):
    a1 = np.array([r["angle_e1_x"] for r in rows])
    a2 = np.array([r["angle_e2_y"] for r in rows])
    f = np.array([r["f"] for r in rows])
    n = len(rows)
    print(f"\n{label} ({n} meshes)")
    print(f"  angle e1-x: median {np.median(a1):.3f} deg, 95th pct {np.percentile(a1, 95):.3f}, "
          f"within {AXIS_TOLERANCE_DEG} deg: {100 * np.mean(a1 <= AXIS_TOLERANCE_DEG):.1f}%")
    print(f"  angle e2-y: median {np.median(a2):.3f} deg, 95th pct {np.percentile(a2, 95):.3f}, "
          f"within {AXIS_TOLERANCE_DEG} deg: {100 * np.mean(a2 <= AXIS_TOLERANCE_DEG):.1f}%")
    neg = (f < -MOMENT_TOLERANCE).sum(axis=0)
    print(f"  negative moments: f_x {neg[0]} ({100 * neg[0] / n:.1f}%), "
          f"f_y {neg[1]} ({100 * neg[1] / n:.1f}%), f_z {neg[2]} ({100 * neg[2] / n:.1f}%)")
    print(f"  all three moments >= 0: {100 * np.mean(np.all(f >= -MOMENT_TOLERANCE, axis=1)):.1f}%")
    return a1, a2, f


def random_rotation(rng):
    """A uniformly random 3D rotation matrix, using NumPy only: the QR decomposition of
    a random Gaussian matrix gives a random orthonormal matrix Q. The sign fixes make
    the distribution uniform, and flipping one column if needed makes det(Q) = +1,
    so Q is a rotation and not a mirroring."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotation_test(database_dir, n_meshes, n_rotations, rng):
    """Max vertex distance between the normalized mesh and the same mesh after a random
    rotation + translation and normalizing again. ~0 = independent of start orientation."""
    items = [(f, m) for f in list_subfolders(database_dir)
             for m in list_meshes(os.path.join(database_dir, f))]
    chosen = rng.choice(len(items), size=min(n_meshes, len(items)), replace=False)
    results = []
    for k in chosen:
        folder, name = items[k]
        mesh = load_mesh(os.path.join(database_dir, folder, name))
        if mesh is None:
            continue
        reference = np.asarray(normalize_mesh(o3d.geometry.TriangleMesh(mesh)).vertices)
        ratio_12 = measure(mesh)["ratio_12"]
        worst = 0.0
        for _ in range(n_rotations):
            m = o3d.geometry.TriangleMesh(mesh)
            R = random_rotation(rng)
            m.rotate(R, center=(0, 0, 0))
            m.translate(rng.uniform(-1, 1, size=3))
            result = np.asarray(normalize_mesh(m).vertices)
            worst = max(worst, float(np.abs(result - reference).max()))
        results.append((f"{folder}/{name}", worst, ratio_12))
    return results


def write_excel(before, after, path="normalization_evidence.xlsx"):
    wb = Workbook()
    for title, rows in (("after", after), ("before", before)):
        ws = wb.active if title == "after" else wb.create_sheet(title)
        ws.title = title
        ws.append(["name", "class", "angle e1-x (deg)", "angle e2-y (deg)",
                   "f_x (relative)", "f_y (relative)", "f_z (relative)", "lambda1/lambda2", "lambda2/lambda3"])
        for r in rows:
            ws.append([r["name"], r["class"], float(r["angle_e1_x"]), float(r["angle_e2_y"]),
                       *map(float, r["f"]), float(r["ratio_12"]), float(r["ratio_23"])])
    wb.save(path)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _moment_histogram(ax, before, after, axis_index, axis_name):
    """Distribution of the relative moment f_i over all meshes, just before flipping
    (centered + aligned) and after. Before: both signs occur. After: every value >= 0."""
    bins = np.linspace(-1, 1, 41)
    f_before = [r["f"][axis_index] for r in before]
    f_after = [r["f"][axis_index] for r in after]
    ax.axvspan(-1, 0, color="#f3e4df", zorder=0)            # the side flipping should empty
    ax.hist(f_after, bins=bins, color="#2a78d6", alpha=0.85, label="after flipping")
    ax.hist(f_before, bins=bins, histtype="step", color="#55554f", linewidth=1.8,
            label="before flipping")
    ax.axvline(0, color="#22221f", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_title(f"Flipping: moment {axis_name} (relative)", loc="left", fontsize=10)
    ax.set_xlabel(f"{axis_name}  (negative = heavier side on the negative {axis_name[-1]}-side)")
    ax.set_ylabel("number of meshes")
    ax.text(-0.97, 0.97, "negative side", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color="#9a5a45")
    _style(ax)


def plot(before, after, path="normalization_evidence.png"):
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)

    # Row 1: alignment
    bins = np.linspace(0, 90, 46)
    for col, key, title in ((0, "angle_e1_x", "Alignment: angle between e1 and x-axis"),
                            (3, "angle_e2_y", "Alignment: angle between e2 and y-axis")):
        ax = fig.add_subplot(grid[0, col:col + 3])
        ax.hist([r[key] for r in before], bins=bins, color="#8a8a85", label="before normalization")
        ax.hist([r[key] for r in after], bins=bins, color="#2a78d6", label="after normalization")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("number of meshes")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)

    # Row 2: flipping, one panel per axis
    for i, name in enumerate(["f_x", "f_y", "f_z"]):
        ax = fig.add_subplot(grid[1, 2 * i:2 * i + 2])
        _moment_histogram(ax, before, after, i, name)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    before = measure_database(config.REFINED_DATABASE_DIR, before_flip=True)
    after = measure_database(config.NORMALIZED_DATABASE_DIR)
    summarize("BEFORE (alignment: original meshes; moments: centered + aligned, not flipped)", before)
    a1, _, _ = summarize("AFTER normalization (normalized database)", after)

    # Which meshes are not aligned within the tolerance, and is it because of equal eigenvalues?
    off = [r for r in after if r["angle_e1_x"] > AXIS_TOLERANCE_DEG or r["angle_e2_y"] > AXIS_TOLERANCE_DEG]
    if off:
        print(f"\n  {len(off)} meshes outside {AXIS_TOLERANCE_DEG} deg; their eigenvalue ratios:")
        for r in sorted(off, key=lambda r: -max(r["angle_e1_x"], r["angle_e2_y"]))[:15]:
            print(f"    {r['class']}/{r['name']}: e1-x {r['angle_e1_x']:.1f}, e2-y {r['angle_e2_y']:.1f} deg, "
                  f"l1/l2 = {r['ratio_12']:.3f}, l2/l3 = {r['ratio_23']:.3f}")

    write_excel(before, after)
    plot(before, after)
    print("\nWritten normalization_evidence.xlsx and normalization_evidence.png")

    if RUN_ROTATION_TEST:
        rng = np.random.default_rng(SEED)
        results = rotation_test(config.REFINED_DATABASE_DIR, ROTATION_TEST_MESHES,
                                ROTATIONS_PER_MESH, rng)
        errors = np.array([r[1] for r in results])
        print(f"\nRotation test ({len(results)} meshes x {ROTATIONS_PER_MESH} random rotations):")
        print(f"  max vertex distance: median {np.median(errors):.2e}, "
              f"below 1e-6 for {100 * np.mean(errors < 1e-6):.1f}% of meshes")
        for name, err, ratio in sorted(results, key=lambda r: -r[1])[:5]:
            print(f"    {name}: {err:.2e}  (l1/l2 = {ratio:.3f})")