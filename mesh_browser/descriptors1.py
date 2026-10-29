"""Shape descriptors for the mesh browser.

Pipeline: raw -> clean + resample -> refined -> normalize -> normalized -> descriptors.
Run this on the NORMALIZED database.
"""
import os
from collections import defaultdict
from functools import cached_property

import numpy as np
import open3d as o3d
import pymeshlab
from openpyxl import Workbook

import mesh_utils as utils

from descriptors import histogram_feature_names, histogram_feature_vector

# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def make_mesh(vertices, triangles):
    """Build an Open3D mesh from NumPy arrays."""
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    out.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return out


def triangle_corners(mesh):
    """The three corner arrays (p0, p1, p2), each of shape (n_triangles, 3)."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    return v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]


def directed_edge_counts(mesh):
    """{(u, w): how many triangles use the directed edge u -> w}."""
    counts = defaultdict(int)
    for a, b, c in np.asarray(mesh.triangles).tolist():
        for u, w in ((a, b), (b, c), (c, a)):
            counts[(u, w)] += 1
    return counts


def to_meshlab(mesh):
    """Open3D mesh -> a PyMeshLab MeshSet holding that mesh."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.triangles, dtype=np.int32)))
    return ms


def from_meshlab(ms):
    """Current mesh of a PyMeshLab MeshSet -> Open3D mesh."""
    m = ms.current_mesh()
    return make_mesh(m.vertex_matrix(), m.face_matrix())


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def surface_area(mesh):
    """Total surface area: the sum of all triangle areas."""
    return float(mesh.get_surface_area())


def mesh_volume(mesh, origin=None):
    """Enclosed volume via signed tetrahedra (needs consistent winding).

    For a balanced mesh the origin does not matter. For an open mesh it does, and
    passing the vertex mean as origin reduces the error."""
    p0, p1, p2 = triangle_corners(mesh)
    if origin is not None:
        p0, p1, p2 = p0 - origin, p1 - origin, p2 - origin
    signed = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    return float(abs(signed.sum()))   # abs: the sign depends on the winding direction


def is_closed_and_consistent(mesh):
    """Strict test: every directed edge used exactly once, and its reverse exactly once."""
    count = directed_edge_counts(mesh)
    return all(n == 1 and count.get((w, u), 0) == 1 for (u, w), n in count.items())


def edge_report(mesh):
    """Per undirected edge: boundary (1 triangle), nonmanifold (3+), samedir (2 triangles,
    same direction), unbalanced (forward count != backward count)."""
    fwd = directed_edge_counts(mesh)
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
            boundary += 1
        elif f + r >= 3:
            nonmanifold += 1
        elif f != 1:
            samedir += 1
    return dict(boundary=boundary, nonmanifold=nonmanifold,
                samedir=samedir, unbalanced=unbalanced)


def max_pairwise_distance(points, chunk=128):
    """Largest distance between any two of the given points (an n x 3 array)."""
    points = np.asarray(points)
    best = 0.0
    for start in range(0, len(points), chunk):  # chunked, so no huge n x n matrix
        diff = points[start:start + chunk, None, :] - points[None, :, :]
        best = max(best, float((diff ** 2).sum(axis=2).max()))
    return float(np.sqrt(best))


def covariance_eccentricity(mesh):
    """Largest / smallest eigenvalue of the covariance matrix of the vertex positions.
    1 for a ball-like spread, large for elongated or flat shapes. Returns None when the
    smallest eigenvalue is (almost) zero, i.e. the vertices lie in a plane or on a line."""
    points = np.asarray(mesh.vertices)
    if len(points) < 4:
        return None
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))     # symmetric 3x3, ascending order
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if largest <= 0 or smallest <= 1e-12 * largest:
        return None
    return largest / smallest


# ---------------------------------------------------------------------------
# Repair pipeline
# ---------------------------------------------------------------------------

def fix_non_manifold(mesh):
    """Remove non-manifold edges (by removing faces) and split non-manifold vertices.
    Do NOT call clean_mesh afterwards: it would merge the split vertices again."""
    ms = to_meshlab(mesh)
    ms.meshing_repair_non_manifold_edges(method="Remove Faces")
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    ms.meshing_remove_unreferenced_vertices()
    return from_meshlab(ms)


def meshlab_close_holes(mesh, max_hole_size=200):
    """MeshLab's hole closing. max_hole_size = max edges on a hole that still gets closed."""
    ms = to_meshlab(mesh)
    ms.meshing_close_holes(maxholesize=max_hole_size)
    return from_meshlab(ms)


def orient_consistently(mesh):
    """Make winding agree across manifold edges (exactly 2 triangles), then flip each
    connected piece so its signed volume is positive (outward). Returns a new mesh."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles).copy()
    m = len(t)

    # Which triangles use each undirected edge
    edge_tris = defaultdict(list)
    for i, (a, b, c) in enumerate(t.tolist()):
        for u, w in ((a, b), (b, c), (c, a)):
            edge_tris[(min(u, w), max(u, w))].append(i)

    # Connect triangles only across manifold edges (exactly 2 triangles)
    neighbours = [[] for _ in range(m)]
    for (u, w), tris in edge_tris.items():
        if len(tris) == 2:
            i, j = tris
            neighbours[i].append((j, u, w))
            neighbours[j].append((i, u, w))

    def has_directed(tri, u, w):
        a, b, c = tri
        return (a, b) == (u, w) or (b, c) == (u, w) or (c, a) == (u, w)

    visited = np.zeros(m, dtype=bool)
    for s in range(m):
        if visited[s]:
            continue
        visited[s] = True
        component, stack = [s], [s]
        while stack:
            i = stack.pop()
            for j, u, w in neighbours[i]:
                if visited[j]:
                    continue
                visited[j] = True
                # Neighbours must traverse the shared edge in opposite directions
                if has_directed(t[i], u, w) == has_directed(t[j], u, w):
                    t[j] = t[j, ::-1].copy()
                component.append(j)
                stack.append(j)

        # Outward-facing: the signed volume of this piece should be positive
        idx = np.array(component)
        p0, p1, p2 = v[t[idx, 0]], v[t[idx, 1]], v[t[idx, 2]]
        if np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() < 0:
            t[idx] = t[idx, ::-1].copy()

    return make_mesh(v, t)


def boundary_loops(mesh):
    """Closed vertex loops along boundary edges (an edge whose reverse is unused)."""
    directed = set(directed_edge_counts(mesh))
    nxt = defaultdict(list)
    for a, b in directed:
        if (b, a) not in directed:
            nxt[a].append(b)

    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, cur, closed = [start], start, True
        while True:
            if cur not in nxt:                 # dead end: boundary does not close up
                closed = False
                break
            n = nxt[cur].pop()
            if not nxt[cur]:
                del nxt[cur]
            if n == start:
                break
            loop.append(n)
            cur = n
        if closed and len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_holes(mesh):
    """Close each hole with a triangle fan around the barycentre of its boundary loop.
    Returns (new mesh, number of holes filled)."""
    loops = boundary_loops(mesh)
    if not loops:
        return mesh, 0

    old_vertices = np.asarray(mesh.vertices)
    vertices = list(old_vertices)
    triangles = np.asarray(mesh.triangles).tolist()

    for loop in loops:
        c = len(vertices)
        vertices.append(old_vertices[loop].mean(axis=0))
        for i in range(len(loop)):
            u, w = loop[i], loop[(i + 1) % len(loop)]   # boundary edge u -> w
            triangles.append([w, u, c])                 # reversed, so winding matches
    return make_mesh(vertices, triangles), len(loops)


def repair_mesh(mesh, max_hole_size=200):
    """Repair a mesh so a volume can be computed. Returns (new mesh, holes the fan filled).
    Works in place on the input for the clean step, so pass a copy."""
    mesh = utils.clean_mesh(mesh)                  # merge seams, remove junk
    mesh = fix_non_manifold(mesh)                  # MeshLab needs manifold input
    mesh = orient_consistently(mesh)
    mesh = meshlab_close_holes(mesh, max_hole_size)  # pass 1: MeshLab closes what it can
    mesh = orient_consistently(mesh)
    mesh, n_holes = fill_holes(mesh)               # pass 2: fan for whatever is still open
    mesh = orient_consistently(mesh)               # pass 3: final orientation
    return mesh, n_holes


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

def unit_or_none(x, tol=1e-6):
    """x if it lies in [0, 1], else None. Values within tol of the range are clipped, so a
    perfect box (ratio 1.0000000002 from rounding) is not thrown away."""
    if x is None or not (-tol <= x <= 1.0 + tol):
        return None
    return float(min(max(x, 0.0), 1.0))


class MeshDescriptors:
    """Descriptors of ONE normalized mesh. Every value is computed on first access and
    cached in self.__dict__, so the slow repair runs at most once per mesh.
    Create a new object per mesh."""

    def __init__(self, mesh, max_hole_size=200):
        self.mesh = mesh                    # normalized, unrepaired
        self.max_hole_size = max_hole_size

    # --- shared intermediates ------------------------------------------------

    @cached_property
    def _repair(self):
        """(repaired mesh, holes filled by the fan). The slow step, runs once."""
        return repair_mesh(o3d.geometry.TriangleMesh(self.mesh), self.max_hole_size)

    @property
    def repaired(self):
        return self._repair[0]

    @cached_property
    def repaired_area(self):
        """Area of the repaired surface: the one that matches self.volume."""
        return surface_area(self.repaired)

    @cached_property
    def volume(self):
        return mesh_volume(self.repaired,
                           origin=np.asarray(self.repaired.vertices).mean(axis=0))

    # --- descriptors ---------------------------------------------------------

    @cached_property
    def area(self):
        """Area descriptor: UNrepaired mesh (the repair adds fake fan triangles)."""
        return surface_area(self.mesh)

    @cached_property
    def _compactness_raw(self):
        if self.repaired_area <= 0:
            return None
        return float(36.0 * np.pi * self.volume ** 2 / self.repaired_area ** 3)

    @cached_property
    def compactness(self):
        """36*pi*V^2 / A^3 on the repaired mesh. None if zero area or outside [0, 1]."""
        return unit_or_none(self._compactness_raw)

    @cached_property
    def quality(self):
        """closed / balanced / open, or 'zero area' / 'implausible' when compactness is None."""
        if self._compactness_raw is None:
            return "zero area"
        if self.compactness is None:
            return "implausible"
        if is_closed_and_consistent(self.repaired):
            return "closed"
        if edge_report(self.repaired)["unbalanced"] == 0:
            return "balanced"
        return "open"

    @cached_property
    def obb_volume(self):
        """Volume of Open3D's oriented bounding box of the repaired mesh. Approximate:
        the box is not guaranteed to be the minimum one. None if Open3D cannot build
        it (for example a completely flat shape)."""
        try:
            return float(self.repaired.get_oriented_bounding_box().volume())
        except RuntimeError:
            return None

    @cached_property
    def rectangularity(self):
        """Shape volume / OBB volume. 1 for a box, lower for shapes that fill their box
        poorly. None if there is no box or the ratio falls outside [0, 1]."""
        if not self.obb_volume:
            return None
        return unit_or_none(self.volume / self.obb_volume)

    @cached_property
    def hull(self):
        """Convex hull of the normalized, unrepaired mesh. Shared by diameter and
        convexity. None if Open3D cannot build it (for example a completely flat shape)."""
        try:
            hull, _ = self.mesh.compute_convex_hull()
            return hull
        except RuntimeError:
            return None

    @cached_property
    def diameter(self):
        """Largest distance between two points of the shape. The farthest pair always
        lies on the convex hull, so only hull vertices are compared. Falls back to all
        vertices when there is no hull."""
        source = self.hull if self.hull is not None else self.mesh
        return max_pairwise_distance(np.asarray(source.vertices))

    @cached_property
    def hull_volume(self):
        if self.hull is None:
            return None
        try:
            return float(mesh_volume(self.hull))
        except RuntimeError:                   # Open3D refuses a hull it deems not watertight
            return None

    @cached_property
    def convexity(self):
        """Shape volume / convex hull volume. 1 for a convex shape, lower for shapes
        with dents or holes. None if there is no hull or the ratio is outside [0, 1]."""
        if not self.hull_volume:
            return None
        return unit_or_none(self.volume / self.hull_volume)

    @cached_property
    def eccentricity(self):
        """Eccentricity of the vertex cloud. Needs no repair and no alignment: the
        eigenvalues do not change when the shape is rotated."""
        return covariance_eccentricity(self.mesh)
    """
    # oude as dict zonder histogram descriptors
    def as_dict(self):
        #All descriptors as one row. Add the new descriptors here when they exist.
        return {
            "area": self.area,
            "volume": self.volume,
            "compactness": self.compactness,
            "rectangularity": self.rectangularity,
            "convexity": self.convexity,
            "diameter": self.diameter,
            "eccentricity": self.eccentricity,
            "quality": self.quality,
        }
    """
    def as_dict(self):
        """All descriptors as one row."""
        row = {
            "area": self.area,
            "volume": self.volume,
            "compactness": self.compactness,
            "rectangularity": self.rectangularity,
            "convexity": self.convexity,
            "diameter": self.diameter,
            "eccentricity": self.eccentricity,
            "quality": self.quality,
        }
        # Histograms on the UNrepaired mesh: the repair adds fake fan triangles
        row.update(zip(histogram_feature_names(), histogram_feature_vector(self.mesh)))
        return row


def compute_descriptors(mesh, max_hole_size=200):
    return MeshDescriptors(mesh, max_hole_size).as_dict()


# ---------------------------------------------------------------------------
# Database run (same pattern as analyze_steps / normalize_steps)
# ---------------------------------------------------------------------------

#COLUMNS = ["name", "folder", "area", "volume", "compactness", "rectangularity", "convexity", "diameter", "eccentricity", "quality"]

# new colums with histogram descriptors
COLUMNS = (["name", "folder", "area", "volume", "compactness", "rectangularity",
            "convexity", "diameter", "eccentricity", "quality"]
           + histogram_feature_names())


FEATURE_COLUMNS = (["area", "compactness", "rectangularity", "convexity",
                    "diameter", "eccentricity"] + histogram_feature_names())


def feature_vector(row):
    """The numeric feature vector of one descriptor row (a dict from as_dict()).
    Missing values (None) become NaN, to be filled in over the whole database."""
    return np.array([np.nan if row.get(c) is None else float(row[c])
                     for c in FEATURE_COLUMNS])
    
    
def list_items(database_dir, limit=None):
    """All (folder, mesh name) pairs in the database. Candidate for mesh_utils.py,
    where analyze_steps, resample_steps and normalize_steps could share it."""
    items = [(f, m) for f in utils.list_subfolders(database_dir)
             for m in utils.list_meshes(os.path.join(database_dir, f))]
    return items if limit is None else items[:limit]

def descriptor_steps(database_dir, output_path="descriptors.xlsx", limit=None):
    """Generator yielding (done, total). Writes the descriptor table to output_path."""
    items = list_items(database_dir, limit)
    total = len(items)

    wb = Workbook()
    ws = wb.active
    ws.title = "Descriptors"
    ws.append(COLUMNS)

    yield 0, total
    for i, (folder, name) in enumerate(items, start=1):
        mesh = utils.load_mesh(os.path.join(database_dir, folder, name))
        if mesh is None:
            row = {"quality": "load failed"}
        else:
            try:
                row = compute_descriptors(mesh)
            except Exception as e:           # one bad mesh must not stop the whole run
                row = {"quality": f"error: {e}"}
        values = [row.get(c) for c in COLUMNS[2:]]
        # openpyxl writes a Python None as an empty cell, so write the text "None" instead
        ws.append([name, folder] + ["None" if v is None else v for v in values])
        yield i, total

    wb.save(output_path)


def descriptor_database(database_dir, output_path="descriptors.xlsx", limit=None):
    """Blocking wrapper around descriptor_steps."""
    for _ in descriptor_steps(database_dir, output_path, limit):
        pass
