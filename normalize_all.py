"""Normalizes the entire resampled database and writes the result to config.NORMALIZED_DATABASE_DIR."""
import time

from mesh_browser import config
from mesh_browser.mesh_utils import normalize_database


def print_progress(done, total):
    print(f"\rNormalizing: {done}/{total}", end="", flush=True)


if __name__ == "__main__":
    print(f"Input:  {config.REFINED_DATABASE_DIR}")
    print(f"Output: {config.NORMALIZED_DATABASE_DIR}")

    start = time.time()
    total = normalize_database(config.REFINED_DATABASE_DIR, config.NORMALIZED_DATABASE_DIR,
                               progress_callback=print_progress)

    print(f"\nDone: {total} meshes in {time.time() - start:.1f} s")