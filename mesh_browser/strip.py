"""A row of small 3D views under the main view: the (normalized) query on the left,
then the best matches from the search, each with a caption (rank, distance, class, name)."""
import open3d.visualization.gui as gui
import numpy as np

from . import config
from .viewer import MeshViewer

CAPTION_LINES = 3   # "#1  0.123" / class / name


class ResultsStrip:
    def __init__(self, window, count=config.STRIP_COUNT):
        """count result tiles plus one tile for the query. The tiles and captions are
        added to the window directly; the app places them with layout()."""
        self.tiles = []
        self.captions = []
        for _ in range(count + 1):
            tile = MeshViewer(window, compact=True, background=config.TILE_BACKGROUND_COLOR)
            caption = gui.Label("")
            window.add_child(tile.widget)
            window.add_child(caption)
            self.tiles.append(tile)
            self.captions.append(caption)
        self._active = [False] * len(self.tiles)   # which tiles currently hold a mesh
        self._last = None                          # view matrix of the last synced state

    def layout(self, x, y, width, height, em):
        """Spread the tiles over the given rectangle, captions below the tiles."""
        caption_height = int(em * (CAPTION_LINES * 1.5 + 0.5))
        tile_width = width // len(self.tiles)
        tile_height = max(height - caption_height, 1)
        for i, (tile, caption) in enumerate(zip(self.tiles, self.captions)):
            left = x + i * tile_width
            tile.widget.frame = gui.Rect(left, y, tile_width, tile_height)
            caption.frame = gui.Rect(left, y + tile_height, tile_width, caption_height)

    # def clear(self):
    #     for tile, caption in zip(self.tiles, self.captions):
    #         tile.clear()
    #         caption.text = ""
    def clear(self):
        for tile, caption in zip(self.tiles, self.captions):
            tile.clear()
            caption.text = ""
        self._active = [False] * len(self.tiles)
        self._last = None

    def show(self, query_mesh, results):
        """query_mesh: the normalized query. results: list of (mesh or None, caption text),
        best match first. Extra results beyond the number of tiles are ignored."""
        self.clear()
        self._set(0, query_mesh, "Query")
        for i, (mesh, text) in enumerate(results, start=1):
            if i >= len(self.tiles):
                break
            self._set(i, mesh, text)
        self._copy_camera_from(0)   # all tiles start from the query tile's camera

    def _set(self, i, mesh, text):
        if mesh is None:
            self.captions[i].text = text + "\n(could not load)"
            return
        self.tiles[i].show_mesh(mesh)
        self._active[i] = True
        self.captions[i].text = text

    def sync_cameras(self):
        """Call every frame. If the user moved one tile, copy its camera to the others.
        Returns True when something changed (the window should redraw)."""
        if not config.SYNC_CAMERAS or sum(self._active) < 2:
            return False
        for i, on in enumerate(self._active):
            if not on:
                continue
            matrix = self.tiles[i].widget.scene.camera.get_view_matrix()
            if self._last is None or not np.allclose(matrix, self._last, atol=1e-6):
                self._copy_camera_from(i)
                return True
        return False

    def _copy_camera_from(self, source):
        camera = self.tiles[source].widget.scene.camera
        self._last = np.array(camera.get_view_matrix())
        for i, on in enumerate(self._active):
            if on and i != source:
                self.tiles[i].widget.scene.camera.copy_from(camera)
                self.tiles[i].widget.force_redraw()