# """The main window: connects the side panel, the 3D view and the database."""

# import os
# import threading

# import open3d.visualization.gui as gui

# from . import config
# from .database import is_valid_database, list_subfolders, list_meshes
# from .mesh_utils import load_mesh, mesh_stats, analyze_steps, analyze_database, normalize_database
# from .side_panel import SidePanel
# from .viewer import MeshViewer


# class MeshBrowserApp:
#     def __init__(self, database_dir):
#         self.database_dir = database_dir
#         self.current_folder = None
#         self._analyzing = False
#         self._normalizing = False

#         self.window = gui.Application.instance.create_window(
#             config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT
#         )
#         em = self.window.theme.font_size

#         self.viewer = MeshViewer(self.window)
#         self.panel = SidePanel(
#             em,
#             display_modes=config.DISPLAY_MODES,
#             on_folder_selected=self._on_folder_selected,
#             on_mesh_selected=self._on_mesh_selected,
#             on_display_mode_changed=self.viewer.set_display_mode,
#             on_analyze_clicked=self._on_analyze_clicked,
#             on_normalize_clicked=self._on_normalize_clicked,
#         )

#         self.window.add_child(self.viewer.widget)
#         self.window.add_child(self.panel.widget)
#         self.window.set_on_layout(self._on_layout)

#         self._open_database(database_dir)

#     def _on_layout(self, layout_context):
#         """3D view on the left, side panel on the right."""
#         r = self.window.content_rect
#         panel_width = config.PANEL_WIDTH_EM * self.window.theme.font_size
#         self.viewer.widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
#         self.panel.widget.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

#     # --- Database ---
#     def _open_database(self, path):
#         self.panel.set_folders([])
#         self.panel.set_meshes([])

#         if not is_valid_database(path):
#             self.panel.set_database_path("(not found) " + path)
#             self.panel.set_info("Database folder not found.\nCheck DATABASE_DIR in config.py.")
#             return

#         self.database_dir = path
#         self.panel.set_database_path(path)

#         folders = list_subfolders(path)
#         self.panel.set_folders(folders)
#         if folders:
#             # Filling the dropdown doesn't trigger its callback, so select the first folder ourselves
#             self._on_folder_selected(folders[0])
#         else:
#             self.panel.set_info("No subfolders found.")

#     # --- Subfolder ---
#     def _on_folder_selected(self, folder_name):
#         self.current_folder = os.path.join(self.database_dir, folder_name)
#         meshes = list_meshes(self.current_folder)
#         self.panel.set_meshes(meshes)
#         self.panel.set_info(f"{len(meshes)} meshes in this folder")

#     # --- Mesh ---
#     def _on_mesh_selected(self, mesh_name):
#         if not mesh_name:
#             return
#         mesh = load_mesh(os.path.join(self.current_folder, mesh_name))
#         if mesh is None:
#             self.panel.set_info("Could not load mesh.")
#             return

#         self.viewer.show_mesh(mesh)
#         n_vertices, n_triangles = mesh_stats(mesh)
#         self.panel.set_info(f"Vertices: {n_vertices}\nTriangles: {n_triangles}")

#         # --- Shape analyzer ---
#     def _on_analyze_clicked(self):
#         """Called when the user clicks the 'Analyze' button."""
#         if self._analyzing:
#             return
#         self._analyzing = True
#         self.panel.set_analyze_enabled(False)
#         self.panel.set_info_analyzer("Analyzing...")
#         threading.Thread(target=self._run_analysis, daemon=True).start()

#     def _run_analysis(self):
#         """Runs on a background thread. Must not touch any widgets directly."""
#         app = gui.Application.instance

#         def on_progress(done, total):
#             if done % 5 == 0 or done == total:
#                 app.post_to_main_thread(
#                     self.window, lambda: self._update_progress(done, total)
#                 )

#         try:
#             n = analyze_database(self.database_dir, "normalized_sheet_resampled.xlsx",
#                                  limit=None, progress_callback=on_progress)
#             message = f"Wrote {n} meshes to normalized_sheet_resampled.xlsx"
#         except PermissionError:
#             message = "Could not write normalized_sheet_resampled.xlsx.\nIs it open in Excel?"
#         except Exception as e:
#             message = f"Analysis failed: {e}"

#         app.post_to_main_thread(self.window, lambda: self._finish_analysis(message))

#     def _update_progress(self, done, total):
#         """Runs on the main thread."""
#         fraction = done / total if total else 1.0
#         filled = int(fraction * 20)
#         bar = "#" * filled + "-" * (20 - filled)
#         self.panel.set_info_analyzer(f"Analyzing...\n[{bar}] {int(fraction * 100)}%\n{done}/{total}")
#         self.window.post_redraw()

#     def _finish_analysis(self, message):
#         """Runs on the main thread."""
#         self._analyzing = False
#         self.panel.set_analyze_enabled(True)
#         self.panel.set_info_analyzer(message)
#         self.window.post_redraw()

#     # --- Normalizer ---
#     def _on_normalize_clicked(self):
#         if self._normalizing:
#             return
#         self._normalizing = True
#         self.panel.set_normalize_enabled(False)
#         self.panel.set_info_normalizer("Normalizing...")
#         threading.Thread(target=self._run_normalization, daemon=True).start()

#     def _run_normalization(self):
#         """Runs on a background thread. Must not touch any widgets directly."""
#         app = gui.Application.instance

#         def on_progress(done, total):
#             if done % 5 == 0 or done == total:
#                 app.post_to_main_thread(
#                     self.window, lambda: self._update_normalize_progress(done, total)
#                 )

#         try:
#             n = normalize_database(self.database_dir, config.NORMALIZED_DATABASE_DIR,
#                                    progress_callback=on_progress)
#             message = f"Normalized {n} meshes to {config.NORMALIZED_DATABASE_DIR}"
#         except Exception as e:
#             message = f"Normalization failed: {e}"

#         app.post_to_main_thread(self.window, lambda: self._finish_normalization(message))

#     def _update_normalize_progress(self, done, total):
#         """Runs on the main thread."""
#         fraction = done / total if total else 1.0
#         filled = int(fraction * 20)
#         bar = "#" * filled + "-" * (20 - filled)
#         self.panel.set_info_normalizer(f"Normalizing...\n[{bar}] {int(fraction * 100)}%\n{done}/{total}")
#         self.window.post_redraw()

#     def _finish_normalization(self, message):
#         """Runs on the main thread."""
#         self._normalizing = False
#         self.panel.set_normalize_enabled(True)
#         self.panel.set_info_normalizer(message)
#         self.window.post_redraw()

# def run():
#     gui.Application.instance.initialize()
#     MeshBrowserApp(config.DATABASE_DIR)
#     gui.Application.instance.run()

"""The main window: connects the side panel, the 3D view and the database."""

import os
from os import path
import threading

import open3d.visualization.gui as gui
import open3d as o3d

from . import config
from .database import is_valid_database, list_subfolders, list_meshes
from .descriptors1 import descriptor_steps
from .mesh_utils import load_mesh, mesh_stats, analyze_database, normalize_database
from .side_panel import SidePanel
from .viewer import MeshViewer
from .search import SearchIndex
from .strip import ResultsStrip
from .knn import ShapeIndex, SINGLE_VALUE_FEATURES, HISTOGRAM_NAMES
from .query import extract_query_features, N_SINGLE
from .feature_normalization import normalize_feature_vector

# ANALYZE_FILE = "sheets/normalized_sheet_resampled.xlsx"
# DESCRIPTORS_FILE = "./sheets/descriptors.xlsx"
# STATS_FILE = "feature_stats.json"
K_RESULTS = config.DEFAULT_K
R_DEFAULT = config.DEFAULT_RANGE    # default radius for range search 


class MeshBrowserApp:
    def __init__(self, database_dir):
        self.database_dir = database_dir
        self.current_folder = None
        self._index = None          # SearchIndex, built on the first search
        self._kd_index = None       # ShapeIndex (k-d tree), built on the first k-d tree search
        self.last_matches = []      # result of the last search (used by the display step)
        self._running = set()       # names of the tools that are busy right now

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
            tools=[
                ("analyze", "Shape analyzer:", "Analyze shapes -> Excel",
                 lambda: self._start_task("analyze", "Analyzing", self._work_analyze)),
                ("normalize", "Normalizer:", "Normalize shapes",
                 lambda: self._start_task("normalize", "Normalizing", self._work_normalize)),
                ("descriptors", "Descriptors:", "Compute descriptors -> Excel",
                 lambda: self._start_task("descriptors", "Computing descriptors",
                                          self._work_descriptors)),
                ("find", "Find similar shapes:", "Choose .obj and search...",
                 lambda: self._on_find_clicked(use_kd_tree=False)),
                ("find_knn", "Find similar shapes (k-d tree):", "Choose .obj and search...",
                 lambda: self._on_find_clicked(use_kd_tree=True)),
            ],
            knn_settings=(K_RESULTS, R_DEFAULT),

        )

        self.window.add_child(self.viewer.widget)
        self.window.add_child(self.panel.widget)

        self.strip = ResultsStrip(self.window)
        self.window.set_on_tick_event(self._on_tick)
        self.last_query_mesh = None

        self.window.set_on_layout(self._on_layout)

        self._open_database(database_dir)

    def _on_tick(self):
        return self.strip.sync_cameras()

    def _on_layout(self, layout_context):
        """3D view on the left, side panel on the right."""
        r = self.window.content_rect
        em = self.window.theme.font_size
        panel_width = config.PANEL_WIDTH_EM * self.window.theme.font_size
        left_width = r.width - panel_width
        strip_height = int(r.height * config.STRIP_HEIGHT_FRACTION)
        
        self.viewer.widget.frame = gui.Rect(r.x, r.y, left_width, r.height - strip_height)
        self.strip.layout(r.x, r.y + r.height - strip_height, left_width, strip_height, em)
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

    # --- Background tasks (analyze, normalize, descriptors) ---
    # Each task is a "work" function: work(on_progress) -> message to show when done.
    # on_progress(done, total) can be called from any thread.

    def _work_analyze(self, on_progress):
        n = analyze_database(self.database_dir, ANALYZE_FILE,
                             limit=None, progress_callback=on_progress)
        return f"Wrote {n} meshes to {ANALYZE_FILE}"

    def _work_normalize(self, on_progress):
        n = normalize_database(self.database_dir, config.NORMALIZED_DATABASE_DIR,
                               progress_callback=on_progress)
        return f"Normalized {n} meshes to {config.NORMALIZED_DATABASE_DIR}"

    def _work_descriptors(self, on_progress):
        """Descriptors are computed on the NORMALIZED database."""
        total = 0
        for done, total in descriptor_steps(config.NORMALIZED_DATABASE_DIR, config.DESCRIPTORS_FILE):
            on_progress(done, total)
        return f"Wrote {total} meshes to {config.DESCRIPTORS_FILE}"

    def _start_task(self, name, label, work):
        """Called on the main thread when a tool button is clicked."""
        if name in self._running:
            return
        self._running.add(name)
        self.panel.set_tool_enabled(name, False)
        self.panel.set_tool_info(name, f"{label}...")
        threading.Thread(target=self._run_task, args=(name, label, work), daemon=True).start()

    def _run_task(self, name, label, work):
        """Runs on a background thread. Must not touch any widgets directly."""
        app = gui.Application.instance

        def on_progress(done, total):
            if done % 5 == 0 or done == total:
                app.post_to_main_thread(
                    self.window, lambda: self._update_progress(name, label, done, total)
                )

        try:
            message = work(on_progress)
        except PermissionError:
            message = "Could not write the output file.\nIs it open in Excel?"
        except Exception as e:
            message = f"{label} failed: {e}"

        app.post_to_main_thread(self.window, lambda: self._finish_task(name, message))

    def _update_progress(self, name, label, done, total):
        """Runs on the main thread."""
        fraction = done / total if total else 1.0
        filled = int(fraction * 20)
        bar = "#" * filled + "-" * (20 - filled)
        self.panel.set_tool_info(name, f"{label}...\n[{bar}] {int(fraction * 100)}%\n{done}/{total}")
        self.window.post_redraw()

    def _finish_task(self, name, message):
        """Runs on the main thread."""
        self._running.discard(name)
        self.panel.set_tool_enabled(name, True)
        self.panel.set_tool_info(name, message)
        self.window.post_redraw()
        if name in ("find", "find_knn") and self.last_matches:
            self._show_results()
        
    #     # --- Open mesh from file ---
    # def _on_open_clicked(self):
    #     dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose a mesh", self.window.theme)
    #     dlg.add_filter(".obj", "Wavefront OBJ (.obj)")
    #     dlg.add_filter("", "All files")
    #     if self.database_dir and os.path.isdir(self.database_dir):
    #         dlg.set_path(self.database_dir)
    #     dlg.set_on_cancel(self._on_open_cancel)
    #     dlg.set_on_done(self._on_open_done)
    #     self.window.show_dialog(dlg)

    # def _on_open_cancel(self):
    #     self.window.close_dialog()

    # def _on_open_done(self, file_path):
    #     self.window.close_dialog()
    #     name = os.path.basename(file_path)
    #     mesh = load_mesh(file_path)
    #     if mesh is None:
    #         self.panel.set_tool_info("load", f"Could not load {name}")
    #         return
    #     self.viewer.show_mesh(mesh)
    #     n_vertices, n_triangles = mesh_stats(mesh)
    #     self.panel.set_info(f"Vertices: {n_vertices}\nTriangles: {n_triangles}")
    #     self.panel.set_tool_info("load", f"Loaded {name}")

        # --- Find: choose a file, show it, search the database with it ---
    def _on_find_clicked(self, use_kd_tree=False):
        task = "find_knn" if use_kd_tree else "find"
        if task in self._running:
            return
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose a query mesh", self.window.theme)
        dlg.add_filter(".obj", "Wavefront OBJ (.obj)")
        dlg.add_filter("", "All files")
        if self.database_dir and os.path.isdir(self.database_dir):
            dlg.set_path(self.database_dir)
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(lambda path: self._on_find_file_chosen(path, use_kd_tree))
        self.window.show_dialog(dlg)

    def _on_find_file_chosen(self, file_path, use_kd_tree=False):
        """Main thread: show the query mesh at once, then search in the background."""
        task = "find_knn" if use_kd_tree else "find"
        self.window.close_dialog()
        name = os.path.basename(file_path)
        mesh = load_mesh(file_path)
        if mesh is None:
            self.panel.set_tool_info(task, f"Could not load {name}")
            return
        self.viewer.show_mesh(mesh)
        n_vertices, n_triangles = mesh_stats(mesh)
        self.panel.set_info(f"Query: {name}\nVertices: {n_vertices}\nTriangles: {n_triangles}")

        self.last_matches = []      # forget the previous search
        self.strip.clear()          # empty the tiles while the new search runs
        if use_kd_tree:
            # Read the settings here, on the main thread (widgets can't be read from the worker)
            mode, k, r = self.panel.get_knn_settings()
            self._start_task("find_knn", "Searching (k-d tree)",
                             lambda on_progress: self._work_find_knn(mesh, name, mode, k, r))
        else:
            self._start_task("find", "Searching",
                             lambda on_progress: self._work_find(mesh, name, K_RESULTS))

    """
    def _work_find(self, mesh, k):
        #Background thread: no widget access here.
        if self._index is None:
            self._index = SearchIndex(config.DESCRIPTORS_FILE, config.STATS_FILE)   # loaded once
        matches, row, normalized = self._index.search(mesh, k=k)
        self.last_query_mesh = normalized
        self.last_matches = matches
        lines = [f"Top {len(matches)} matches:"]
        for rank, m in enumerate(matches, start=1):
            lines.append(f"{rank}. {m['class']}/{m['name']}  {m['distance']:.3f}")
        return "\n".join(lines)   
    """
    def _work_find(self, mesh, query_name, k):
        """Background thread: no widget access here."""
        if self._index is None:
            self._index = SearchIndex(config.DESCRIPTORS_FILE, config.STATS_FILE)   # loaded once
        # Ask for one extra result: if the query shape is in the database, it is its own #1
        matches, row, normalized = self._index.search(mesh, k=k + 1)
        matches = [m for m in matches if m["name"] != query_name][:k]
        self.last_query_mesh = normalized
        self.last_matches = matches
        lines = [f"Top {len(matches)} matches:"]
        for rank, m in enumerate(matches, start=1):
            lines.append(f"{rank}. {m['class']}/{m['name']}  {m['distance']:.3f}")
        return "\n".join(lines) 
    
    def _build_kd_index(self):
        """Store the normal search's normalized database vectors in a k-d tree.
        The column names are rebuilt from the vector length (6 single values followed by
        5 histograms of equal length), so search.py does not need to change."""
        idx = self._index
        n_bins = (idx.features.shape[1] - len(SINGLE_VALUE_FEATURES)) // len(HISTOGRAM_NAMES)
        feature_names = SINGLE_VALUE_FEATURES + [f"{h}_{i}" for h in HISTOGRAM_NAMES
                                                 for i in range(n_bins)]
        return ShapeIndex(idx.names, idx.classes, idx.features, feature_names, weighted=config.WEIGHTED)

    def _work_find_knn(self, mesh, query_name, mode, k, r):
        """Background thread: the query's features are computed as in the normal search,
        then the k-d tree returns either the k nearest shapes (mode "knn") or all shapes
        within distance r (mode "range")."""
        if self._index is None:
            self._index = SearchIndex(config.DESCRIPTORS_FILE, config.STATS_FILE)
        if self._kd_index is None:
            self._kd_index = self._build_kd_index()          # built once, reused afterwards

        # Query features: same pipeline and normalization as the normal search
        features, row, normalized = extract_query_features(mesh)
        query = normalize_feature_vector(features[:N_SINGLE], features[N_SINGLE:],
                                         self._index.stats)
        query = self._kd_index.prepare_query(query)          # same weighting as the tree

        # Leave the query out if it is itself in the database (it would be its own #1)
        exclude = query_name if query_name in self._kd_index.names else None
        if mode == "range":
            results = self._kd_index.range_search(query, r, exclude=exclude)
            header = f"{len(results)} shapes within R = {r:g} (k-d tree)"
            if len(results) > config.STRIP_COUNT:
                header += f", closest {config.STRIP_COUNT} shown below"
        else:
            results = self._kd_index.knn(query, k, exclude=exclude)
            header = f"Top {len(results)} matches, K = {k} (k-d tree)"

        self.last_query_mesh = normalized
        self.last_matches = [{"name": n, "class": c, "distance": d} for n, c, d in results]
        if not results:
            return header + "\nNo shapes found. Try a larger R."
        lines = [header + ":"]
        for rank, (n, c, d) in enumerate(results, start=1):
            lines.append(f"{rank}. {c}/{n}  {d:.3f}")
        return "\n".join(lines)

    def _show_results(self):
        """Main thread: fill the results strip from the last search."""
        results = []
        for rank, m in enumerate(self.last_matches[:config.STRIP_COUNT], start=1):
            result_mesh = load_mesh(os.path.join(config.NORMALIZED_DATABASE_DIR,
                                                 m["class"], m["name"]))
            caption = (f"#{rank}  {m['distance']:.3f}\n{m['class']}\n"
                       f"{os.path.splitext(m['name'])[0]}")
            results.append((result_mesh, caption))
        self.strip.show(self.last_query_mesh, results)

def run():
    gui.Application.instance.initialize()
    MeshBrowserApp(config.DATABASE_DIR)
    gui.Application.instance.run()