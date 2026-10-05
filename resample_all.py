"""Resamples the entire database and writes the result to config.REFINED_DATABASE_DIR."""
import time

from mesh_browser import config
from mesh_browser.mesh_utils import resample_database

LOG_PATH = "resample_sheet.xlsx"


def print_progress(done, total):
    print(f"\rResampling: {done}/{total}", end="", flush=True)


if __name__ == "__main__":
    print(f"Input:  {config.DATABASE_DIR}")
    print(f"Output: {config.REFINED_DATABASE_DIR}")

    start = time.time()
    total = resample_database(config.DATABASE_DIR, config.REFINED_DATABASE_DIR, LOG_PATH,
                              progress_callback=print_progress)

    print(f"\nDone: {total} meshes in {time.time() - start:.1f} s")
    print(f"Log written to {LOG_PATH}")