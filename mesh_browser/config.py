#DATABASE_DIR = "ShapeDatabase_INFOMR-master"
# DATABASE_DIR = "ShapeDatabase_INFOMR-master_refined"
DATABASE_DIR = "normalized_database"
# DATABASE_DIR = "debug_failed"

REFINED_DATABASE_DIR = "ShapeDatabase_INFOMR-master_refined"

NORMALIZED_DATABASE_DIR = "normalized_database"

MESH_EXTENSIONS = (".obj",)
 
# Window settings
WINDOW_TITLE = "Mesh Browser"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
 
# Width of the side panel, in multiples of the font size
PANEL_WIDTH_EM = 20
 
# Background colour of the 3D view (R, G, B, A), values between 0 and 1
BACKGROUND_COLOR = [255, 255, 255, 1]

DISPLAY_SHADED = "Shaded"
DISPLAY_SHADED_WIREFRAME = "Shaded + wireframe"
DISPLAY_WIREFRAME = "Wireframe"
DISPLAY_MODES = [DISPLAY_SHADED, DISPLAY_SHADED_WIREFRAME, DISPLAY_WIREFRAME]
 
# Wireframe appearance (R, G, B, A) and line thickness in pixels
WIREFRAME_COLOR = [0, 0, 0, 1]
WIREFRAME_LINE_WIDTH = 5.0

# Super/subsampling:
MIN_VERTICES = 5000
MAX_VERTICES = 10000
TARGET_VERTICES = 7500