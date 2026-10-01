"""The main window: connects the side panel, the 3D view and the database."""

import os

import open3d.visualization.gui as gui

from . import config
from .database import is_valid_database, list_subfolders, list_meshes
from .mesh_utils import load_mesh, mesh_stats, analyze_database
from .side_panel import SidePanel
from .viewer import MeshViewer


class MeshBrowserApp:
    def __init__(self, database_dir):
        self.database_dir = database_dir
        self.current_folder = None

        self.window = gui.Application.instance.create_window(
            config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT
        )
        em = self.window.theme.font_size

        self.viewer = MeshViewer(self.window)
        self.panel = SidePanel(
            em,
            display_modes=config.DISPLAY_MODES,
            on_folder_selected=self._on_folder_selected,
            on_mesh_selected=self._on_mesh_selected,
            on_display_mode_changed=self.viewer.set_display_mode,
            on_analyze_clicked=self._on_analyze_clicked,
        )

        self.window.add_child(self.viewer.widget)
        self.window.add_child(self.panel.widget)
        self.window.set_on_layout(self._on_layout)

        self._open_database(database_dir)

    def _on_layout(self, layout_context):
        """3D view on the left, side panel on the right."""
        r = self.window.content_rect
        panel_width = config.PANEL_WIDTH_EM * self.window.theme.font_size
        self.viewer.widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.widget.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    # --- Database ---
    def _open_database(self, path):
        self.panel.set_folders([])
        self.panel.set_meshes([])

        if not is_valid_database(path):
            self.panel.set_database_path("(not found) " + path)
            self.panel.set_info("Database folder not found.\nCheck DATABASE_DIR in config.py.")
            return

        self.database_dir = path
        self.panel.set_database_path(path)

        folders = list_subfolders(path)
        self.panel.set_folders(folders)
        if folders:
            # Filling the dropdown doesn't trigger its callback, so select the first folder ourselves
            self._on_folder_selected(folders[0])
        else:
            self.panel.set_info("No subfolders found.")

    # --- Subfolder ---
    def _on_folder_selected(self, folder_name):
        self.current_folder = os.path.join(self.database_dir, folder_name)
        meshes = list_meshes(self.current_folder)
        self.panel.set_meshes(meshes)
        self.panel.set_info(f"{len(meshes)} meshes in this folder")

    # --- Mesh ---
    def _on_mesh_selected(self, mesh_name):
        if not mesh_name:
            return
        mesh = load_mesh(os.path.join(self.current_folder, mesh_name))
        if mesh is None:
            self.panel.set_info("Could not load mesh.")
            return

        self.viewer.show_mesh(mesh)
        n_vertices, n_triangles = mesh_stats(mesh)
        self.panel.set_info(f"Vertices: {n_vertices}\nTriangles: {n_triangles}")

    # --- Shape analyzer ---
    def _on_analyze_clicked(self):
        try:
            n_rows = analyze_database(self.database_dir)
            self.panel.set_info_analyzer(f"Shape analyzer: wrote {n_rows} rows to Excel.\n")
        except Exception as e:
            self.panel.set_info_analyzer(f"Shape analyzer: error occurred - {e}\nIs excel file open? Close it and try again.\n")

def run():
    gui.Application.instance.initialize()
    MeshBrowserApp(config.DATABASE_DIR)
    gui.Application.instance.run()