import open3d.visualization.gui as gui


class SidePanel:
    def __init__(self, em, display_modes, on_folder_selected,
                 on_mesh_selected, on_display_mode_changed, on_analyze_clicked):
        self.widget = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))

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

        # Shape analyzer
        self.widget.add_child(gui.Label("")) # Spacing
        self.widget.add_child(gui.Label("Shape analyzer:"))
        self.analyze_button = gui.Button("Analyze shapes -> Excel")
        self.analyze_button.set_on_clicked(on_analyze_clicked)
        self.widget.add_child(self.analyze_button)
        # self.progress_bar = gui.ProgressBar()
        # self.widget.add_child(self.progress_bar)

        # Shape analyzer info text (mesh statistics, messages)
        self.info_label_analyzer = gui.Label("")
        self.widget.add_child(self.info_label_analyzer)

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

    def set_info_analyzer(self, text):
        self.info_label_analyzer.text = text

    # def set_progress(self, fraction):
    #     self.progress_bar.set_value(fraction)

    def set_analyze_enabled(self, enabled):
        self.analyze_button.enabled = enabled # Important for threading, otherwise the button can be clicked multiple times