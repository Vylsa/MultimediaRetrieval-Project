"""Loading meshes and preparing them for display. No GUI code here."""

import numpy as np
import open3d as o3d


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