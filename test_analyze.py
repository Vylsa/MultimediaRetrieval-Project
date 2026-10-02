from mesh_browser import config
from mesh_browser.mesh_utils import analyze_database

n = analyze_database(config.DATABASE_DIR, "analyze_sheet.xlsx", limit=200,
                     progress_callback=lambda d, t: print(d, t))
print("done", n)