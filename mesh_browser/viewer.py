"""The 3D view that displays a mesh."""

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from . import config
from .mesh_utils import make_double_sided

MESH_NAME = "mesh"
WIREFRAME_NAME = "wireframe"
AXES_NAME = "axes"

AXIS_COLORS = [[0.9, 0.1, 0.1],     # x: red
               [0.1, 0.8, 0.1],     # y: green
               [0.1, 0.3, 0.9]]     # z: blue


def make_axes(length):
    """Three colored lines from the origin along +x, +y and +z."""
    points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=float)
    lines = np.array([[0, 1], [0, 2], [0, 3]])
    axes = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points),
                                o3d.utility.Vector2iVector(lines))
    axes.colors = o3d.utility.Vector3dVector(AXIS_COLORS)
    return axes


class MeshViewer:
    def __init__(self, window, compact=False, background=None):
        self.compact = compact
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(window.renderer)
        self.widget.scene.set_background(background or config.BACKGROUND_COLOR)
        self.widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.577, -0.577, -0.577))
        self.widget.scene.view.set_post_processing(False)
        
        # Material for the shaded surface
        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"

        # Material for the wireframe lines
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.base_color = config.WIREFRAME_COLOR
        self.line_material.line_width = config.WIREFRAME_LINE_WIDTH

        # Material for the axes: white base color, so the per-line colors are shown unchanged
        self.axes_material = rendering.MaterialRecord()
        self.axes_material.shader = "unlitLine"
        self.axes_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.axes_material.line_width = 2.0

        self.axis_labels = []
        self.display_mode = config.DISPLAY_MODES[0]

    def show_mesh(self, mesh):
        """Replace whatever is shown with this mesh and point the camera at it."""
        scene = self.widget.scene
        scene.clear_geometry()        
        
        # Add both the surface and the wireframe; the display mode decides which are visible
        scene.add_geometry(MESH_NAME, make_double_sided(mesh), self.mesh_material)
        bounds = mesh.get_axis_aligned_bounding_box()
        #scene.add_geometry(MESH_NAME, mesh, self.mesh_material)

        if not self.compact:
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            scene.add_geometry(WIREFRAME_NAME, wireframe, self.line_material)
            self._add_axes(bounds)

        self._apply_display_mode()

        # Frame the mesh and the origin, so the axes stay in view for unnormalized meshes too
        view_points = np.vstack([np.asarray(bounds.get_box_points()), np.zeros((1, 3))])
        view_bounds = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(view_points))
        self.widget.setup_camera(60, view_bounds, bounds.get_center())

    def _add_axes(self, bounds):
        """x/y/z axes through the origin, long enough to stick out of the mesh."""
        corners = np.vstack([bounds.get_min_bound(), bounds.get_max_bound()])
        length = 0.9 * max(float(np.max(bounds.get_extent())), float(np.max(np.abs(corners))))
        #self.widget.scene.add_geometry(AXES_NAME, make_axes(length), self.axes_material)
        self.widget.scene.add_geometry(AXES_NAME, make_axes(1), self.axes_material)

        # Text labels at the axis tips (widget overlays, so clear_geometry() doesn't remove them)
        for label in self.axis_labels:
            self.widget.remove_3d_label(label)
        self.axis_labels = []
        for name, direction, color in zip("XYZ", np.eye(3), AXIS_COLORS):
            label = self.widget.add_3d_label(direction * length * 1.08, name)
            label.color = gui.Color(*color)
            self.axis_labels.append(label)

    def set_display_mode(self, mode):
        """Switch between the modes in config.DISPLAY_MODES. Kept when a new mesh is loaded."""
        self.display_mode = mode
        self._apply_display_mode()

    def _apply_display_mode(self):
        if self.compact:
            return  # tiles are shaded
        scene = self.widget.scene
        if not scene.has_geometry(MESH_NAME):
            return  # nothing loaded yet

        scene.show_geometry(MESH_NAME, self.display_mode != config.DISPLAY_WIREFRAME)
        scene.show_geometry(WIREFRAME_NAME, self.display_mode != config.DISPLAY_SHADED)
        self.widget.force_redraw()

    def clear(self):
        self.widget.scene.clear_geometry()
        self.widget.force_redraw()    