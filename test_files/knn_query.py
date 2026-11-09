"""Query the shape database with the k-d tree: K-nearest neighbors and range search.

The location of the feature database is set by FEATURES_PATH in mesh_browser/knn.py.
Set the settings below and run:
    python knn_query.py
"""
from pathlib import Path
import sys
import time

import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser.knn import ShapeIndex, print_results


# --- Settings ---------------------------------------------------------

QUERY_SHAPE = "m1337.obj"   # file name of a shape in the database
K = 5                       # number of nearest neighbors
R = None                     # radius for range search (None to skip)
WEIGHTED = True             # True: every feature counts equally; False: plain L2

# ------------------------------------------------------------------------------------


def benchmark(index, n_queries=2000, k=K, seed=0):
    """Compare query time with and without the tree, and check the results are identical.
    Results count as equivalent when the distances match, even if shapes at (nearly) the
    same distance are listed in a different order."""
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(index.names), size=min(n_queries, len(index.names)), replace=False)
    same_names = same_distances = 0
    t_tree = t_brute = 0.0
    differences = []
    for q in queries:
        name = index.names[q]
        vector = index.features[q]
        start = time.perf_counter()
        a = index.knn(vector, k, exclude=name)
        t_tree += time.perf_counter() - start
        start = time.perf_counter()
        b = index.brute_force_knn(vector, k, exclude=name)
        t_brute += time.perf_counter() - start

        names_equal = [r[0] for r in a] == [r[0] for r in b]
        distances_equal = np.allclose([r[2] for r in a], [r[2] for r in b], rtol=0, atol=1e-9)
        same_names += names_equal
        same_distances += distances_equal
        if not names_equal:
            differences.append((name, a, b))

    n = len(queries)
    print(f"\nBenchmark ({n} queries, K={k}, {len(index.names)} shapes, {index.features.shape[1]} features):")
    print(f"  k-d tree:     {1000 * t_tree / n:.3f} ms per query (built in {1000 * index.build_time:.1f} ms)")
    print(f"  brute force:  {1000 * t_brute / n:.3f} ms per query")
    print(f"  identical names, in the same order: {same_names}/{n}")
    print(f"  identical distances:                {same_distances}/{n}")

    for name, a, b in differences:
        print(f"\n  Query {name}: results differ")
        for (na, ca, da), (nb, cb, db) in zip(a, b):
            mark = "" if na == nb else "   <-- different"
            print(f"    tree {ca}/{na} {da:.12f} | brute {cb}/{nb} {db:.12f}{mark}")


if __name__ == "__main__":
    index = ShapeIndex.from_file(weighted=WEIGHTED)
    print(f"Loaded {len(index.names)} shapes with {index.features.shape[1]} features each "
          f"({'weighted' if WEIGHTED else 'plain L2'})")

    if QUERY_SHAPE not in index.names:
        raise SystemExit(f"{QUERY_SHAPE} is not in the feature database. Check QUERY_SHAPE.")

    query_class = index.classes[index.names.index(QUERY_SHAPE)]
    print_results(f"{K} nearest neighbors of {QUERY_SHAPE} ({query_class}):",
                  index.knn_by_name(QUERY_SHAPE, K))

    if R is not None:
        print("\nDistance between random pairs of shapes, percentiles (to choose R):")
        print("  " + ", ".join(f"{p}%: {d:.2f}" for p, d in index.distance_percentiles().items()))
        print_results(f"All shapes within distance {R} of {QUERY_SHAPE}:",
                      index.range_by_name(QUERY_SHAPE, R))

    benchmark(index)