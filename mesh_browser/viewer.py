"""The 3D view that displays a mesh."""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from . import config
from .mesh_utils import make_double_sided

MESH_NAME = "mesh"
WIREFRAME_NAME = "wireframe"


class MeshViewer:
    def __init__(self, window):
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(window.renderer)
        self.widget.scene.set_background(config.BACKGROUND_COLOR)

        # Material for the shaded surface
        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"

        # Material for the wireframe lines
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.base_color = config.WIREFRAME_COLOR
        self.line_material.line_width = config.WIREFRAME_LINE_WIDTH

        self.display_mode = config.DISPLAY_MODES[0]

    def show_mesh(self, mesh):
        """Replace whatever is shown with this mesh and point the camera at it."""
        scene = self.widget.scene
        scene.clear_geometry()

        # Add both the surface and the wireframe; the display mode decides which are visible
        scene.add_geometry(MESH_NAME, make_double_sided(mesh), self.mesh_material)
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        scene.add_geometry(WIREFRAME_NAME, wireframe, self.line_material)
        self._apply_display_mode()

        bounds = mesh.get_axis_aligned_bounding_box()
        self.widget.setup_camera(60, bounds, bounds.get_center())

    def set_display_mode(self, mode):
        """Switch between the modes in config.DISPLAY_MODES. Kept when a new mesh is loaded."""
        self.display_mode = mode
        self._apply_display_mode()

    def _apply_display_mode(self):
        scene = self.widget.scene
        if not scene.has_geometry(MESH_NAME):
            return  # nothing loaded yet

        scene.show_geometry(MESH_NAME, self.display_mode != config.DISPLAY_WIREFRAME)
        scene.show_geometry(WIREFRAME_NAME, self.display_mode != config.DISPLAY_SHADED)
        self.widget.force_redraw()