"""Loading meshes and preparing them for display. No GUI code here."""
import os

import numpy as np
import open3d as o3d
import threading
import pymeshlab
from openpyxl import Workbook
from .database import list_subfolders, list_meshes
from .sheet_utils import add_summary_sheet, add_shape_lists
from . import config
from collections import defaultdict

import heapq

def load_mesh(path):
    """Load a mesh from a file. Returns None if the file couldn't be loaded."""
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_triangles():
        return None
    mesh.compute_vertex_normals()
    return mesh


def mesh_stats(mesh):
    """Return (number of vertices, number of triangles)."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return vertices.shape[0], triangles.shape[0]

def mesh_bbox(mesh):
    """Axis-aligned bounding box of a mesh.
    Returns (min_x, min_y, min_z, max_x, max_y, max_z)."""
    bbox = mesh.get_axis_aligned_bounding_box()
    min_b = bbox.get_min_bound()
    max_b = bbox.get_max_bound()
    return (*map(float, min_b), *map(float, max_b))

def count_face_types(path):
    """Read the face definitions straight from the file (Open3D triangulates on load).
    Returns a dict {vertices_per_face: count}, e.g. {3: 1200, 4: 80}.
    Returns None for unsupported formats."""
    ext = os.path.splitext(path)[1].lower()
    counts = {}

    if ext == ".obj":
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("f "):
                    n = len(line.split()) - 1  # tokens after the 'f'
                    counts[n] = counts.get(n, 0) + 1
        return counts

    return None


def classify_faces(counts):
    """Turn a face count dict into a label."""
    if not counts:
        return "unknown"
    kinds = set(counts)
    if kinds == {3}:
        return "triangles"
    if kinds == {4}:
        return "quads"
    if kinds == {3, 4}:
        return "triangles + quads"
    return "mixed (n-gons)"


def make_double_sided(mesh):
    """Return a copy of the mesh where every triangle also exists facing the other way,
    so the inside is visible (e.g. through holes) and lit correctly.
    Only use this for display, not for computations."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n = len(vertices)

    # Duplicate the vertices so the flipped triangles get their own (inward-pointing) normals
    all_vertices = np.vstack([vertices, vertices])
    # Reverse the vertex order of each triangle to flip which side is the front
    flipped = np.ascontiguousarray(triangles[:, ::-1]) + n
    all_triangles = np.vstack([triangles, flipped]).astype(np.int32)

    double = o3d.geometry.TriangleMesh()
    double.vertices = o3d.utility.Vector3dVector(all_vertices)
    double.triangles = o3d.utility.Vector3iVector(all_triangles)
    double.compute_vertex_normals()
    return double

def analyze_steps(database_dir, output_path="analyze_sheet.xlsx", limit=None):
    """Generator: analyzes the meshes one at a time and yields (done, total) after each.
    The Excel file is written when the generator is exhausted."""
    items = []
    for folder in list_subfolders(database_dir):
        folder_path = os.path.join(database_dir, folder)
        for mesh_name in list_meshes(folder_path):
            items.append((folder, mesh_name))
    if limit is not None:
        items = items[:limit]
    total = len(items)

    wb = Workbook()
    ws = wb.active
    ws.title = "analyze_sheet"
    ws.append(["Mesh name", "Class", "Vertices", "Triangle faces", "Face types", 
               "Original faces", "Axis-aligned bounding box", "Min X", "Min Y", "Min Z", "Max X", "Max Y", "Max Z"])

    vertex_counts, face_counts, class_counts = [], [], {}
    records = []
    yield 0, total
    for i, (folder, mesh_name) in enumerate(items, start=1):
        path = os.path.join(database_dir, folder, mesh_name)
        mesh = load_mesh(path)
        class_counts[folder] = class_counts.get(folder, 0) + 1

        try:
            counts = count_face_types(path)
        except Exception:
            counts = None  # a malformed file shouldn't abort the whole run
        face_type = classify_faces(counts)
        n_faces = sum(counts.values()) if counts else "n/a"
        bbox = mesh_bbox(mesh) if mesh is not None else ("n/a",) * 6
        bbox_text = "(" + ", ".join(f"{v:.4f}" for v in bbox) + ")" if mesh is not None else "n/a"

        if mesh is None:
            ws.append([mesh_name, folder, "load failed", "load failed", face_type, n_faces, bbox_text, *bbox])
        else:
            n_vertices, n_triangles = mesh_stats(mesh)
            vertex_counts.append(n_vertices)
            if isinstance(n_faces, int):
                face_counts.append(n_faces)
                records.append({"name": mesh_name, "folder": folder,
                                "vertices": n_vertices, "faces": n_faces})
            ws.append([mesh_name, folder, n_vertices, n_triangles, face_type, n_faces, bbox_text, *bbox])
        yield i, total

    add_summary_sheet(wb, vertex_counts, face_counts, class_counts)
    add_shape_lists(wb, records)
    wb.save(output_path)  # overwrites an existing file


def analyze_database(database_dir, output_path="analyze_sheet.xlsx",
                     limit=None, progress_callback=None):
    """Blocking version (handy for tests). Returns the number of meshes analyzed."""
    total = 0
    for done, total in analyze_steps(database_dir, output_path, limit):
        if progress_callback:
            progress_callback(done, total)
    return total


def clean_mesh(mesh):
    """Merge duplicated vertices and remove degenerate/unused geometry (in place).
    OBJ files often duplicate vertices at UV/normal seams, which inflates the
    vertex count and causes cracks when subdividing."""
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_unreferenced_vertices()
    return mesh


'''Resample version using Open3D's built-in subdivision method "subdivide_midpoint". Works but overshoots the target vertex count. sometimes resulting
in a mesh that needs to be decimated. Using the inbuilt decimation method after this subdivide leads to errors as
the decimation will preserve flat serfaces, which is exactly what we get when we subdivide using the midpoint method.'''
'''
def resample_mesh(mesh, min_v=config.MIN_VERTICES, max_v=config.MAX_VERTICES,
                  target_v=config.TARGET_VERTICES, max_iter=10):
    """Subdivide or decimate a mesh until its vertex count is within [min_v, max_v].
    Returns (resampled mesh, list of actions taken), e.g. ["subdivide", "decimate"]."""
    actions = []
    for _ in range(max_iter):
        n_vertices = len(mesh.vertices)
        if min_v <= n_vertices <= max_v:
            break
        if n_vertices < min_v:
            # Splits every triangle into 4 using the edge midpoints, shape stays the same
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            actions.append("subdivide")
        else:
            # Decimation targets a triangle count, so estimate it from the current ratio
            ratio = len(mesh.triangles) / n_vertices
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=int(target_v * ratio))
            actions.append("decimate")
        mesh = clean_mesh(mesh)
    return mesh, actions
'''

'''resample version using the custom subdivision method "split_longest_edges"'''
def resample_mesh(mesh, min_v=config.MIN_VERTICES, max_v=config.MAX_VERTICES,
                  target_v=config.TARGET_VERTICES, max_iter=10):
    """Refine or decimate a mesh until its vertex count is within [min_v, max_v].
    Returns (resampled mesh, list of actions taken), e.g. ["split longest edges"]."""
    actions = []
    for _ in range(max_iter):
        n_vertices = len(mesh.vertices)
        if min_v <= n_vertices <= max_v:
            break
        if n_vertices < min_v:
            # Splits the longest edges first and stops exactly at target_v, so it never overshoots
            mesh = split_longest_edges(mesh, target_v)
            actions.append("split longest edges")
        else:
            # Only reached for meshes that were too dense to begin with
            ratio = len(mesh.triangles) / n_vertices
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=int(target_v * ratio))
            actions.append("decimate")
        mesh = clean_mesh(mesh)
    return mesh, actions

def split_longest_edges(mesh, target_v=config.TARGET_VERTICES):
    """Repeatedly split the longest edge of the mesh at its midpoint until the mesh
    has target_v vertices. Each split adds exactly one vertex, so the largest
    triangles are refined first and the result lands exactly on target_v.
    Expects a cleaned mesh (shared vertices), otherwise cracks appear at seams."""
    vertices = list(np.asarray(mesh.vertices))
    triangles = [list(t) for t in np.asarray(mesh.triangles)]

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def push(a, b):
        length = np.linalg.norm(vertices[a] - vertices[b])
        heapq.heappush(heap, (-length, key(a, b)))  # negative, so the longest comes out first

    # Which triangles use each edge
    edge_tris = {}
    for t, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_tris.setdefault(key(u, v), []).append(t)

    heap = []
    for a, b in edge_tris:
        push(a, b)

    while len(vertices) < target_v and heap:
        _, edge = heapq.heappop(heap)
        if edge not in edge_tris:
            continue  # this edge was already split
        a, b = edge
        m = len(vertices)
        vertices.append((vertices[a] + vertices[b]) / 2)

        for t in edge_tris.pop(edge):
            tri = triangles[t]
            # Rotate so the split edge is tri[0]-tri[1], keeping the winding order
            i = next(k for k in range(3) if key(tri[k], tri[(k + 1) % 3]) == edge)
            p, q, c = tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]

            # Split the triangle into (p, m, c) and (m, q, c)
            new_t = len(triangles)
            triangles[t] = [p, m, c]
            triangles.append([m, q, c])

            # Edge q-c now belongs to the new triangle instead of t
            qc = edge_tris[key(q, c)]
            qc[qc.index(t)] = new_t
            edge_tris.setdefault(key(p, m), []).append(t)
            edge_tris.setdefault(key(m, q), []).append(new_t)
            edge_tris.setdefault(key(m, c), []).extend([t, new_t])
            push(m, c)

        push(a, m)
        push(m, b)

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    result.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    return result

def resample_steps(database_dir, output_dir=config.REFINED_DATABASE_DIR,
                   log_path="resample_sheet.xlsx", limit=None):
    """Generator: resamples the meshes one at a time and yields (done, total) after each.
    Every mesh is written to output_dir with the same folder structure, also the ones
    that were already in range, so output_dir is a complete database.
    The Excel log is written when the generator is exhausted."""
    items = []
    for folder in list_subfolders(database_dir):
        folder_path = os.path.join(database_dir, folder)
        for mesh_name in list_meshes(folder_path):
            items.append((folder, mesh_name))
    if limit is not None:
        items = items[:limit]
    total = len(items)

    wb = Workbook()
    ws = wb.active
    ws.title = "resample_sheet"
    ws.append(["Mesh name", "Class", "Vertices before", "Vertices after",
               "Triangles before", "Triangles after", "Actions", "In range"])

    yield 0, total
    for i, (folder, mesh_name) in enumerate(items, start=1):
        path = os.path.join(database_dir, folder, mesh_name)
        mesh = load_mesh(path)

        if mesh is None:
            ws.append([mesh_name, folder, "load failed", "", "", "", "", ""])
            yield i, total
            continue

        mesh = clean_mesh(mesh)
        v_before, t_before = mesh_stats(mesh)
        mesh, actions = resample_mesh(mesh)
        v_after, t_after = mesh_stats(mesh)
        in_range = config.MIN_VERTICES <= v_after <= config.MAX_VERTICES

        out_folder = os.path.join(output_dir, folder)
        os.makedirs(out_folder, exist_ok=True)
        mesh.compute_vertex_normals()
        mesh.triangle_normals = o3d.utility.Vector3dVector()  # OBJ can't store these
        o3d.io.write_triangle_mesh(os.path.join(out_folder, mesh_name), mesh)

        ws.append([mesh_name, folder, v_before, v_after, t_before, t_after,
                   ", ".join(actions) if actions else "none", "yes" if in_range else "no"])
        yield i, total

    wb.save(log_path)  # overwrites an existing file


def resample_database(database_dir, output_dir=config.REFINED_DATABASE_DIR,
                      log_path="resample_sheet.xlsx", limit=None, progress_callback=None):
    """Blocking version (handy for tests). Returns the number of meshes processed."""
    total = 0
    for done, total in resample_steps(database_dir, output_dir, log_path, limit):
        if progress_callback:
            progress_callback(done, total)
    return total

def normalize_steps(database_dir, output_dir=config.NORMALIZED_DATABASE_DIR, limit=None):
    """Generator: normalizes the meshes one at a time and yields (done, total) after each.
    Every mesh is written to output_dir with the same folder structure."""
    items = []
    for folder in list_subfolders(database_dir):
        folder_path = os.path.join(database_dir, folder)
        for mesh_name in list_meshes(folder_path):
            items.append((folder, mesh_name))
    if limit is not None:
        items = items[:limit]
    total = len(items)

    yield 0, total
    for i, (folder, mesh_name) in enumerate(items, start=1):
        mesh = load_mesh(os.path.join(database_dir, folder, mesh_name))

        if mesh is None:
            print(f"Skipped (load failed): {folder}/{mesh_name}")
            yield i, total
            continue

        mesh = normalize_mesh(mesh)

        out_folder = os.path.join(output_dir, folder)
        os.makedirs(out_folder, exist_ok=True)
        mesh.triangle_normals = o3d.utility.Vector3dVector()  # OBJ can't store these
        o3d.io.write_triangle_mesh(os.path.join(out_folder, mesh_name), mesh)
        yield i, total


def normalize_database(database_dir, output_dir=config.NORMALIZED_DATABASE_DIR,
                       limit=None, progress_callback=None):
    """Blocking version (handy for tests). Returns the number of meshes processed."""
    total = 0
    for done, total in normalize_steps(database_dir, output_dir, limit):
        if progress_callback:
            progress_callback(done, total)
    return total

'''
# normalize mesh function from step 2, so only centering and scaling
def normalize_mesh(mesh):
    """Translate the mesh so its barycenter is at the origin, then scale it uniformly
    so its axis-aligned bounding box fits tightly in a unit cube (largest side = 1).
    Modifies the mesh in place and also returns it."""
    # 1. Barycenter -> origin (Open3D's get_center() is the mean of the vertices)
    mesh.translate(-mesh.get_center())

    # 2. Uniform scale: the longest side of the bounding box becomes exactly 1
    bbox = mesh.get_axis_aligned_bounding_box()
    longest_side = float(np.max(bbox.get_extent()))
    if longest_side > 0:
        # center=(0,0,0) so the barycenter stays at the origin while scaling
        mesh.scale(1.0 / longest_side, center=(0.0, 0.0, 0.0))
    return mesh
''' 
def center_mesh(mesh):
    """Step 1: translate the mesh so its surface barycenter is at the origin."""
    mesh.translate(-surface_barycenter(mesh))
    return mesh

def surface_barycenter(mesh):
    """Area-weighted mean of the triangle barycenters (centroid of the surface)."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    p0, p1, p2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    centroids = (p0 + p1 + p2) / 3.0

    total = areas.sum()
    if total == 0:                      # degenerate mesh, fall back to the vertex mean
        return v.mean(axis=0)
    return (centroids * areas[:, None]).sum(axis=0) / total

def principal_axes(vertices):
    """PCA of a point set. Returns (eigenvalues, eigenvectors) sorted from largest
    to smallest eigenvalue; the eigenvectors are the columns of the 3x3 matrix."""
    cov = np.cov(vertices, rowvar=False)          # 3x3 covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh: symmetric matrix, ascending order
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def align_mesh(mesh):
    """Step 2: rotate the mesh so the major eigenvector lies along x and the
    medium one along y. Expects a centered mesh."""
    _, eigvecs = principal_axes(np.asarray(mesh.vertices))
    e1, e2 = eigvecs[:, 0], eigvecs[:, 1]
    # Use the cross product instead of the third eigenvector, so the frame is
    # always right-handed (a pure rotation, never an accidental mirror)
    e3 = np.cross(e1, e2)
    # Rows are the new axes, so R @ p gives the coordinates (p.e1, p.e2, p.e3)
    R = np.vstack([e1, e2, e3])
    mesh.rotate(R, center=(0.0, 0.0, 0.0))
    return mesh


def flip_mesh(mesh):
    """Step 3: moment test. For each axis, compute
    f = sum over triangles of sign(c) * c^2, with c the triangle centroid coordinate,
    and mirror that axis if f < 0, so most of the mass ends up on the positive side."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    tri = vertices[triangles]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    centroids = tri.mean(axis=1)
    f = np.sum(areas[:, None] * np.sign(centroids) * centroids ** 2, axis=0)
    signs = np.where(f < 0, -1.0, 1.0)

    mesh.vertices = o3d.utility.Vector3dVector(vertices * signs)
    # Mirroring an odd number of axes turns the mesh inside out (normals point
    # inward), so reverse the winding order of every triangle to compensate
    if np.prod(signs) < 0:
        mesh.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(triangles[:, ::-1]).astype(np.int32))
    return mesh


def scale_mesh(mesh):
    """Step 4: scale uniformly so the longest side of the AABB becomes 1."""
    longest_side = float(np.max(mesh.get_axis_aligned_bounding_box().get_extent()))
    if longest_side > 0:
        mesh.scale(1.0 / longest_side, center=(0.0, 0.0, 0.0))
    return mesh


def normalize_mesh(mesh):
    """Four-step normalization: center, align with PCA, flip with the moment test,
    then scale to the unit cube. Modifies the mesh in place and also returns it."""
    center_mesh(mesh)
    align_mesh(mesh)
    flip_mesh(mesh)
    scale_mesh(mesh)
    mesh.compute_vertex_normals()  # flipping changed the geometry, so recompute
    return mesh


def mesh_volume(mesh):
    """Volume of a mesh, computed by summing the signed volumes of the tetrahedra
    formed by the barycenter and each triangle. Also works for meshes that are not
    watertight (unlike Open3D's get_volume): holes are effectively closed by
    'coning' them to the barycenter, which gives a good approximation."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return 0.0

    barycenter = vertices.mean(axis=0)

    # Triangle corners relative to the barycenter (the shared apex of all tetrahedra)
    a = vertices[triangles[:, 0]] - barycenter
    b = vertices[triangles[:, 1]] - barycenter
    c = vertices[triangles[:, 2]] - barycenter

    # Signed volume of each tetrahedron: (a . (b x c)) / 6
    tet_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0

    # Sum the signed volumes, then take abs so the result doesn't depend on
    # whether the triangles wind inward or outward
    return float(abs(tet_volumes.sum()))