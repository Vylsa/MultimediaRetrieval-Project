from mesh_browser import config
from mesh_browser.mesh_utils import mesh_volume
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("normalized_database/0_normalizing_tests/sphere.obj")
n = mesh_volume(mesh)
print(n)