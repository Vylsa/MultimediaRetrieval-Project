import os
import pymeshlab
import open3d as o3d
import numpy as np
import mesh_browser.mesh_utils as utils

from collections import defaultdict

def _run_filter(ms, new_name, old_name, **kwargs):
    fn = getattr(ms, new_name, None) or getattr(ms, old_name, None)
    if fn is None:
        raise AttributeError(f"Neither {new_name} nor {old_name} exists in your PyMeshLab version")
    fn(**kwargs)

def meshlab_close_holes(mesh, max_hole_size=200):
    """MeshLab's hole closing (handles larger and non-planar holes better than a fan).
    max_hole_size is the maximum number of edges on a hole that still gets closed."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.triangles, dtype=np.int32)))
    _run_filter(ms, "meshing_close_holes", "close_holes", maxholesize=max_hole_size)

    m = ms.current_mesh()
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(m.vertex_matrix())
    out.triangles = o3d.utility.Vector3iVector(m.face_matrix().astype(np.int32))
    return out

def fix_non_manifold(mesh):
    """Remove non-manifold edges and vertices with MeshLab. Returns a new Open3D mesh.
    Edges are repaired by removing faces (this creates holes, which fill_holes closes
    afterwards). Vertices are repaired by splitting them."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.triangles, dtype=np.int32)))

    _run_filter(ms, "meshing_repair_non_manifold_edges",
                "repair_non_manifold_edges_by_removing_faces",
                **({"method": "Remove Faces"}
                   if hasattr(ms, "meshing_repair_non_manifold_edges") else {}))
    _run_filter(ms, "meshing_repair_non_manifold_vertices",
            "repair_non_manifold_vertices_by_splitting", vertdispratio=0.0)
    _run_filter(ms, "meshing_remove_unreferenced_vertices",
                "remove_unreferenced_vertices")

    m = ms.current_mesh()
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(m.vertex_matrix())
    out.triangles = o3d.utility.Vector3iVector(m.face_matrix().astype(np.int32))
    return out

def orient_consistently(mesh):
    """Make all triangle windings agree with their neighbours, then flip each connected
    piece so it faces outward (positive signed volume). Returns a new mesh."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles).copy()
    m = len(t)

    # Which triangles use each (undirected) edge
    edge_tris = defaultdict(list)
    for i, (a, b, c) in enumerate(t):
        for u, w in ((a, b), (b, c), (c, a)):
            edge_tris[(min(u, w), max(u, w))].append(i)

    # Connect triangles only across manifold edges (exactly 2 triangles).
    # Edges shared by 3+ triangles have no consistent answer, so we skip them.
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
                # Neighbours must traverse the shared edge in opposite directions.
                # If both go the same way, flip j by reversing its vertex order.
                if has_directed(t[i], u, w) == has_directed(t[j], u, w):
                    t[j] = t[j, ::-1].copy()
                component.append(j)
                stack.append(j)

        # Outward-facing: signed volume of this piece should be positive
        idx = np.array(component)
        p0, p1, p2 = v[t[idx, 0]], v[t[idx, 1]], v[t[idx, 2]]
        if np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() < 0:
            t[idx] = t[idx, ::-1].copy()

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(v)
    out.triangles = o3d.utility.Vector3iVector(t.astype(np.int32))
    return out

def boundary_loops(mesh):
    """Find hole boundaries. Returns a list of closed vertex loops [v1, v2, ..., vn].
    An edge is a boundary edge if its reverse direction is not used by any triangle."""
    t = np.asarray(mesh.triangles)
    directed = set()
    for a, b, c in t:
        directed.update(((a, b), (b, c), (c, a)))

    nxt = defaultdict(list)
    for a, b in directed:
        if (b, a) not in directed:
            nxt[a].append(b)

    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, cur, closed = [start], start, True
        while True:
            if cur not in nxt:                 # dead end: boundary doesn't close up
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
    """Close each hole with a triangle fan around the barycentre of its boundary vertices.
    Returns (new mesh, number of holes filled)."""
    loops = boundary_loops(mesh)
    if not loops:
        return mesh, 0

    vertices = [p for p in np.asarray(mesh.vertices)]
    triangles = [list(t) for t in np.asarray(mesh.triangles)]

    for loop in loops:
        c = len(vertices)
        vertices.append(np.asarray(mesh.vertices)[loop].mean(axis=0))
        for i in range(len(loop)):
            u, w = loop[i], loop[(i + 1) % len(loop)]   # boundary edge u -> w
            triangles.append([w, u, c])                 # reversed, so winding matches

    filled = o3d.geometry.TriangleMesh()
    filled.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    filled.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    return filled, len(loops)

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

def compute_descriptors(mesh):
    """Compute a set of descriptors for the mesh."""
    surface_area = mesh.get_surface_area()
    volume = mesh.get_volume()