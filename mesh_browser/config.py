import os

# DATABASE_DIR = "ShapeDatabase_INFOMR-master"
# DATABASE_DIR = "ShapeDatabase_INFOMR-master_refined"
# DATABASE_DIR = "normalized_database"
# DATABASE_DIR = "debug_failed"

# REFINED_DATABASE_DIR = "ShapeDatabase_INFOMR-master_refined"

# NORMALIZED_DATABASE_DIR = "normalized_database"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHEETS_DIR = os.path.join(PROJECT_ROOT, "sheets")

DESCRIPTORS_FILE = os.path.join(SHEETS_DIR, "descriptors.xlsx")
STATS_FILE = os.path.join(PROJECT_ROOT, "feature_stats.json")

#DATABASE_DIR = os.path.join(PROJECT_ROOT, "ShapeDatabase_INFOMR-master")
#DATABASE_DIR = os.path.join(PROJECT_ROOT, "ShapeDatabase_INFOMR-master_refined")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "normalized_database")


REFINED_DATABASE_DIR = os.path.join(PROJECT_ROOT, "ShapeDatabase_INFOMR-master_refined")
NORMALIZED_DATABASE_DIR = os.path.join(PROJECT_ROOT, "normalized_database")

MESH_EXTENSIONS = (".obj",)
 
# Window settings
WINDOW_TITLE = "Mesh Browser"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
 
# Width of the side panel, in multiples of the font size
PANEL_WIDTH_EM = 20
 
# Background colour of the 3D view (R, G, B, A), values between 0 and 1
BACKGROUND_COLOR = [1, 1, 1, 1]

DISPLAY_SHADED = "Shaded"
DISPLAY_SHADED_WIREFRAME = "Shaded + wireframe"
DISPLAY_WIREFRAME = "Wireframe"
DISPLAY_MODES = [DISPLAY_SHADED, DISPLAY_SHADED_WIREFRAME, DISPLAY_WIREFRAME]

# Results strip under the main view
STRIP_COUNT = 5                                # best matches shown (plus the query)
STRIP_HEIGHT_FRACTION = 0.25                   # share of the window height
TILE_BACKGROUND_COLOR = [0.95, 0.95, 0.95, 1]  # light grey, so the tiles are visible

# Wireframe appearance (R, G, B, A) and line thickness in pixels
WIREFRAME_COLOR = [0, 0, 0, 1]
WIREFRAME_LINE_WIDTH = 1.0

# Super/subsampling:
MIN_VERTICES = 5000
MAX_VERTICES = 10000
TARGET_VERTICES = 7500

SYNC_CAMERAS = True    # rotate all result tiles together

# K-Nearest-Neighbors
DEFAULT_K = 6
DEFAULT_RANGE = 1.5

WEIGHTED = True               # True: every feature counts equally; False: plain L2
STANDARDIZED_FILE = os.path.join(SHEETS_DIR, "normalized_features.xlsx")
STANDARDIZED_SHEET = "zscore"  # the sheet inside the workbook with the z-scores

# Classes that are not real shape classes; removed BEFORE t-SNE.
TEST_CLASSES = ["0_normalizing_tests", "0_resampling_tests"]
