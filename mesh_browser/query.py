"""Feature extraction for a query shape (Step 8, part 1).

A query mesh can come from anywhere, so it is raw: any position, scale, orientation
and vertex count. The database features were computed on meshes that went through
clean -> resample -> normalize, so the query has to go through exactly the same
stages. Otherwise its features are not comparable to the database features.
"""
import os

import open3d as o3d

import mesh_utils as utils
from descriptors1 import compute_descriptors, feature_vector


def prepare_query_mesh(mesh):
    """Bring a raw mesh to the same state as the database meshes.
    Returns (normalized mesh, resample actions). The input mesh is not modified."""
    mesh = o3d.geometry.TriangleMesh(mesh)      # copy: normalize_mesh works in place
    mesh = utils.clean_mesh(mesh)
    if not mesh.has_triangles():
        raise ValueError("The mesh has no usable triangles after cleaning.")
    mesh, actions = utils.resample_mesh(mesh)
    mesh = utils.normalize_mesh(mesh)
    return mesh, actions


def extract_query_features(mesh):
    """Features of a raw query mesh.

    Returns (features, row, normalized_mesh):
      features         NumPy vector in the order of FEATURE_COLUMNS (None -> NaN).
                       NaNs and scaling are handled later, with database statistics.
      row              the full descriptor dict (includes 'quality' and 'volume')
      normalized_mesh  the mesh that was actually measured, handy for display
    """
    normalized, _ = prepare_query_mesh(mesh)
    row = compute_descriptors(normalized)
    return feature_vector(row), row, normalized


def extract_query_features_from_file(path):
    """Load an .obj file and extract its features. Raises ValueError if it can't be loaded."""
    mesh = utils.load_mesh(path)
    if mesh is None:
        raise ValueError(f"Could not load {os.path.basename(path)}")
    return extract_query_features(mesh)

print(extract_query_features_from_file("C:\\Users\\pjotr\\Documents\\UU\\Multimedia retrieval\\MultimediaRetrieval-Project\\ShapeDatabase_INFOMR-master\\AircraftBuoyant\\m1337.obj"))