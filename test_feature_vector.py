"""Test the descriptors and feature vector on a single mesh.
Set file_path below and run:  python test_descriptors.py
"""
import time

import numpy as np

from mesh_browser.mesh_utils import load_mesh
from mesh_browser.descriptors1 import MeshDescriptors, FEATURE_COLUMNS, feature_vector
from mesh_browser.descriptors import DISTRIBUTION_NAMES, N_BINS

file_path = r"normalized_database/AircraftBuoyant/m1337.obj"  


if __name__ == "__main__":
    mesh = load_mesh(file_path)
    if mesh is None:
        raise SystemExit(f"Could not load {file_path}")
    print(f"Loaded {file_path}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles\n")

    start = time.time()
    row = MeshDescriptors(mesh).as_dict()
    print(f"Computed in {time.time() - start:.1f} s\n")

    # 1. The single-value descriptors
    print("Single-value descriptors:")
    for name in ["area", "volume", "compactness", "rectangularity",
                 "convexity", "diameter", "eccentricity", "quality"]:
        value = row.get(name)
        shown = f"{value:.4f}" if isinstance(value, float) else value
        print(f"  {name:<15} {shown}")

    # 2. The histograms: each should have N_BINS values that sum to 1
    print("\nHistograms:")
    for name in DISTRIBUTION_NAMES:
        hist = np.array([row[f"{name}_{i}"] for i in range(N_BINS)])
        status = "OK" if abs(hist.sum() - 1) < 1e-6 else "SUM IS NOT 1"
        print(f"  {name}: {np.round(hist, 3)}  sum={hist.sum():.3f}  {status}")

    # 3. The feature vector
    fv = feature_vector(row)
    print(f"\nFeature vector: {len(fv)} values (expected {len(FEATURE_COLUMNS)})")
    missing = [c for c, v in zip(FEATURE_COLUMNS, fv) if np.isnan(v)]
    print(f"Missing values: {', '.join(missing) if missing else 'none'}")