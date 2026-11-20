# import open3d.visualization.gui as gui

# KNN_MODES = ["K nearest neighbors", "All within distance R"]


# class SidePanel:
#     def __init__(self, em, display_modes, on_folder_selected,
#                  on_mesh_selected, on_display_mode_changed, tools, knn_settings=None):
#         """tools: list of (name, title, button_text, on_clicked). Every tool gets a
#         title, a button and an info label, addressed later by its name.
#         knn_settings: (default K, default R) to show the k-d tree search settings, or None."""
#         # Scrollable, so all tools stay reachable when the window is not very tall
#         self.widget = gui.ScrollableVert(0.5 * em, gui.Margins(em, em, em, em))
#         self._tool_buttons = {}
#         self._tool_infos = {}

#         # Database folder
#         self.widget.add_child(gui.Label("Database folder:"))
#         self.database_label = gui.Label("")
#         self.widget.add_child(self.database_label)

#         # Subfolders
#         self.widget.add_child(gui.Label("Subfolder:"))
#         self.folder_box = gui.Combobox()
#         self.folder_box.set_on_selection_changed(
#             lambda text, index: on_folder_selected(text)
#         )
#         self.widget.add_child(self.folder_box)

#         # Meshes
#         self.widget.add_child(gui.Label("Mesh:"))
#         self.mesh_list = gui.ListView()
#         self.mesh_list.set_max_visible_items(15)
#         self.mesh_list.set_on_selection_changed(
#             lambda value, is_double_click: on_mesh_selected(value)
#         )
#         self.widget.add_child(self.mesh_list)

#         # Display mode (shaded / wireframe)
#         self.widget.add_child(gui.Label("Display:"))
#         self.display_box = gui.Combobox()
#         for mode in display_modes:
#             self.display_box.add_item(mode)
#         self.display_box.set_on_selection_changed(
#             lambda text, index: on_display_mode_changed(text)
#         )
#         self.widget.add_child(self.display_box)

#         # Info text (mesh statistics, messages)
#         self.info_label = gui.Label("")
#         self.widget.add_child(self.info_label)

#         # Settings for the k-d tree search (used by the "find_knn" tool)
#         if knn_settings is not None:
#             self._add_knn_settings(em, *knn_settings)

#         # Tools (analyzer, normalizer, descriptors, ...)
#         for name, title, button_text, on_clicked in tools:
#             self._add_tool(name, title, button_text, on_clicked)

#     def _add_knn_settings(self, em, default_k, default_r):
#         """Search mode (K nearest / within R) and the values of K and R."""
#         self.widget.add_child(gui.Label(""))   # spacing
#         self.widget.add_child(gui.Label("k-d tree search settings:"))

#         self.knn_mode_box = gui.Combobox()
#         for mode in KNN_MODES:
#             self.knn_mode_box.add_item(mode)
#         self.widget.add_child(self.knn_mode_box)

#         self.k_edit = gui.NumberEdit(gui.NumberEdit.INT)
#         self.k_edit.set_limits(1, 500)
#         self.k_edit.int_value = int(default_k)
#         self.r_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
#         self.r_edit.set_limits(0.0, 100.0)
#         self.r_edit.double_value = float(default_r)

#         for label, edit in (("K:", self.k_edit), ("R:", self.r_edit)):
#             row = gui.Horiz(0.5 * em)
#             row.add_child(gui.Label(label))
#             row.add_child(edit)
#             self.widget.add_child(row)

#     def get_knn_settings(self):
#         """(mode, K, R), with mode "knn" or "range". Call on the main thread."""
#         mode = "range" if self.knn_mode_box.selected_index == 1 else "knn"
#         return mode, int(self.k_edit.int_value), float(self.r_edit.double_value)

#     def _add_tool(self, name, title, button_text, on_clicked):
#         self.widget.add_child(gui.Label(""))   # spacing
#         self.widget.add_child(gui.Label(title))
#         button = gui.Button(button_text)
#         button.set_on_clicked(on_clicked)
#         self.widget.add_child(button)
#         info = gui.Label("")
#         self.widget.add_child(info)
#         self._tool_buttons[name] = button
#         self._tool_infos[name] = info

#     def set_database_path(self, text):
#         self.database_label.text = text

#     def set_folders(self, folders):
#         self.folder_box.clear_items()
#         for folder in folders:
#             self.folder_box.add_item(folder)

#     def set_meshes(self, meshes):
#         self.mesh_list.set_items(meshes)

#     def set_info(self, text):
#         self.info_label.text = text

#     def set_tool_info(self, name, text):
#         self._tool_infos[name].text = text

#     def set_tool_enabled(self, name, enabled):
#         # Important for threading, otherwise the button can be clicked multiple times
#         self._tool_buttons[name].enabled = enabled
import open3d.visualization.gui as gui

KNN_MODES = ["K nearest neighbors", "All within distance R"]


class SidePanel:
    def __init__(self, em, display_modes, on_folder_selected,
                 on_mesh_selected, on_display_mode_changed, tools, knn_settings=None):
        """tools: list of (name, title, button_text, on_clicked). Every tool gets a
        title, a button and an info widget, addressed later by its name.
        knn_settings: (default K, default R) to show the k-d tree search settings, or None."""
        # Scrollable, so all tools stay reachable when the window is not very tall
        self.widget = gui.ScrollableVert(0.5 * em, gui.Margins(em, em, em, em))
        self._em = em
        self._tool_buttons = {}
        self._tool_infos = {}
        self._multi_line_tools = {"find", "find_knn"}

        # Database folder
        self.widget.add_child(gui.Label("Database folder:"))
        self.database_label = gui.Label("")
        self.widget.add_child(self.database_label)

        # Subfolders
        self.widget.add_child(gui.Label("Subfolder:"))
        self.folder_box = gui.Combobox()
        self.folder_box.set_on_selection_changed(
            lambda text, index: on_folder_selected(text)
        )
        self.widget.add_child(self.folder_box)

        # Meshes
        self.widget.add_child(gui.Label("Mesh:"))
        self.mesh_list = gui.ListView()
        self.mesh_list.set_max_visible_items(15)
        self.mesh_list.set_on_selection_changed(
            lambda value, is_double_click: on_mesh_selected(value)
        )
        self.widget.add_child(self.mesh_list)

        # Display mode (shaded / wireframe)
        self.widget.add_child(gui.Label("Display:"))
        self.display_box = gui.Combobox()
        for mode in display_modes:
            self.display_box.add_item(mode)
        self.display_box.set_on_selection_changed(
            lambda text, index: on_display_mode_changed(text)
        )
        self.widget.add_child(self.display_box)

        # Info text (mesh statistics, messages)
        self.info_label = gui.Label("")
        self.widget.add_child(self.info_label)

        # Settings for the k-d tree search (used by the "find_knn" tool)
        if knn_settings is not None:
            self._add_knn_settings(em, *knn_settings)

        # Tools (analyzer, normalizer, descriptors, ...)
        for name, title, button_text, on_clicked in tools:
            self._add_tool(name, title, button_text, on_clicked)

    def _add_knn_settings(self, em, default_k, default_r):
        """Search mode (K nearest / within R) and the values of K and R."""
        self.widget.add_child(gui.Label(""))   # spacing
        self.widget.add_child(gui.Label("k-d tree search settings:"))

        self.knn_mode_box = gui.Combobox()
        for mode in KNN_MODES:
            self.knn_mode_box.add_item(mode)
        self.widget.add_child(self.knn_mode_box)

        self.k_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self.k_edit.set_limits(1, 500)
        self.k_edit.int_value = int(default_k)
        self.r_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.r_edit.set_limits(0.0, 100.0)
        self.r_edit.double_value = float(default_r)

        for label, edit in (("K:", self.k_edit), ("R:", self.r_edit)):
            row = gui.Horiz(0.5 * em)
            row.add_child(gui.Label(label))
            row.add_child(edit)
            self.widget.add_child(row)

    def get_knn_settings(self):
        """(mode, K, R), with mode "knn" or "range". Call on the main thread."""
        mode = "range" if self.knn_mode_box.selected_index == 1 else "knn"
        return mode, int(self.k_edit.int_value), float(self.r_edit.double_value)

    def _add_tool(self, name, title, button_text, on_clicked):
        self.widget.add_child(gui.Label(""))   # spacing
        self.widget.add_child(gui.Label(title))
        button = gui.Button(button_text)
        button.set_on_clicked(on_clicked)
        self.widget.add_child(button)

        if name in self._multi_line_tools:
            info = gui.WidgetProxy()
            info.set_widget(gui.Vert(0.15 * self._em))
        else:
            info = gui.Label("")

        self.widget.add_child(info)
        self._tool_buttons[name] = button
        self._tool_infos[name] = info

    def set_database_path(self, text):
        self.database_label.text = text

    def set_folders(self, folders):
        self.folder_box.clear_items()
        for folder in folders:
            self.folder_box.add_item(folder)

    def set_meshes(self, meshes):
        self.mesh_list.set_items(meshes)

    def set_info(self, text):
        self.info_label.text = text

    def set_tool_info(self, name, text):
        info = self._tool_infos[name]
        if name in self._multi_line_tools:
            box = gui.Vert(0.15 * self._em)
            for line in (text or "").splitlines():
                box.add_child(gui.Label(line))
            info.set_widget(box)
        else:
            info.text = text

    def set_tool_enabled(self, name, enabled):
        # Important for threading, otherwise the button can be clicked multiple times
        self._tool_buttons[name].enabled = enabled