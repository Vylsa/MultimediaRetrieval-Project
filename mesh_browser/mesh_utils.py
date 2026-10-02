"""Loading meshes and preparing them for display. No GUI code here."""
import os

import numpy as np
import open3d as o3d
import threading
from openpyxl import Workbook
from .database import list_subfolders, list_meshes


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
    ws.append(["Mesh name", "Class", "Vertices", "Triangle faces", "Face types", "Original faces"])

    yield 0, total
    for i, (folder, mesh_name) in enumerate(items, start=1):
        path = os.path.join(database_dir, folder, mesh_name)
        mesh = load_mesh(path)

        try:
            counts = count_face_types(path)
        except Exception:
            counts = None  # a malformed file shouldn't abort the whole run
        face_type = classify_faces(counts)
        n_faces = sum(counts.values()) if counts else "n/a"

        if mesh is None:
            ws.append([mesh_name, folder, "load failed", "load failed", face_type, n_faces])
        else:
            n_vertices, n_triangles = mesh_stats(mesh)
            ws.append([mesh_name, folder, n_vertices, n_triangles, face_type, n_faces])
        yield i, total

    wb.save(output_path)  # overwrites an existing file


def analyze_database(database_dir, output_path="analyze_sheet.xlsx",
                     limit=None, progress_callback=None):
    """Blocking version (handy for tests). Returns the number of meshes analyzed."""
    total = 0
    for done, total in analyze_steps(database_dir, output_path, limit):
        if progress_callback:
            progress_callback(done, total)
    return total