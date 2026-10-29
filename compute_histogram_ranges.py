"""Determines the histogram ranges for D1-D4 from the normalized database and saves them to
mesh_browser/histogram_ranges.json. After that, shape_distributions.py uses these ranges
automatically, so recompute the descriptors afterwards.

For every mesh, the raw descriptor values are computed and a random subset is kept. The
upper bound of each range is a high percentile of all those values together, so a handful
of extreme shapes don't stretch the range (values above it go into the last bin)."""
import json
import os

import numpy as np

from mesh_browser import config
from mesh_browser.mesh_utils import list_meshes, list_subfolders, load_mesh
from mesh_browser.descriptors import RANGES_PATH, distribution_values

PERCENTILE = 99.9          # upper bound = this percentile of all values in the database
VALUES_PER_MESH = 2000     # random subset kept per mesh and descriptor (keeps memory low)
DESCRIPTORS = ["D1", "D2", "D3", "D4"]   # A3 keeps its natural range [0, pi]
SEED = 0


def collect_values(database_dir):
    """A random subset of the raw D1-D4 values of every mesh, pooled per descriptor."""
    rng = np.random.default_rng(SEED)
    pooled = {name: [] for name in DESCRIPTORS}
    items = [(f, m) for f in list_subfolders(database_dir)
             for m in list_meshes(os.path.join(database_dir, f))]
    for i, (folder, name) in enumerate(items, start=1):
        mesh = load_mesh(os.path.join(database_dir, folder, name))
        if mesh is not None:
            values = distribution_values(mesh)
            for d in DESCRIPTORS:
                v = values[d]
                pooled[d].append(rng.choice(v, size=min(VALUES_PER_MESH, len(v)), replace=False))
        print(f"\rCollecting values: {i}/{len(items)}", end="", flush=True)
    print()
    return {d: np.concatenate(v) for d, v in pooled.items()}


if __name__ == "__main__":
    pooled = collect_values(config.NORMALIZED_DATABASE_DIR)
    ranges = {}
    for d in DESCRIPTORS:
        upper = float(np.percentile(pooled[d], PERCENTILE))
        ranges[d] = [0.0, upper]
        print(f"{d}: range [0, {upper:.4f}]  (absolute max {pooled[d].max():.4f})")
    with open(RANGES_PATH, "w") as f:
        json.dump(ranges, f, indent=2)
    print(f"Saved to {RANGES_PATH}. Recompute the descriptors to use the new ranges.")