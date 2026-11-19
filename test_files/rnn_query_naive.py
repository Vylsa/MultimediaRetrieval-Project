"""Benchmark range search (RNN) for several radii: k-d tree vs brute force.

Brute force: compute the distance from the query to every shape, sort all distances, and
keep the shapes within radius R. The k-d tree only computes distances to the shapes in the
parts of the tree that can contain shapes within R.

Per radius we measure the time per query of both methods (median over all queries), the
average number of shapes found, how many distances the tree computes, and whether both
methods find exactly the same shapes.

The radii are chosen from the distribution of distances between random pairs of shapes
(percentiles), so they automatically fit the scale of the (weighted) feature space.

Run:
    python rnn_benchmark.py
It prints a table, writes rnn_benchmark.csv and saves the plot as rnn_benchmark.png.
"""
from pathlib import Path
import csv
import gc
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser.knn import ShapeIndex


# --- Settings ---------------------------------------------------------

WEIGHTED = True             # same setting as in the app
N_QUERIES = 2000            # queries per radius; brute force takes ~7 ms each, so 2000
                            # queries x 11 radii takes about 2.5 minutes
PERCENTILES = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10, 25, 50]   # radius = this percentile of pair distances
OUTPUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------

TREE_COLOR = "#2a78d6"      # blue
BRUTE_COLOR = "#eb6834"     # orange


# --- The two ways to answer one range query ------------------------------------------
# Both return (indices, distances) of all shapes within radius r, nearest first.
# The query shape itself (distance 0) is included by both.

def tree_range(index, q, r):
    ind, dist = index.tree.query_radius(q[None, :], r=r, return_distance=True,
                                        sort_results=True)
    return ind[0], dist[0]


def brute_force_range(X, q, r):
    """Compute the distance to every shape, sort them all, keep those within r."""
    distances = [np.sqrt(((x - q) ** 2).sum()) for x in X]
    order = sorted(range(len(X)), key=distances.__getitem__)
    inside = [i for i in order if distances[i] <= r]
    return np.array(inside, dtype=int), np.array([distances[i] for i in inside])


# --- Benchmark ----------------------------------------------------------------------

def pair_distance_percentiles(index, percentiles, n_pairs=100000, seed=1):
    """The given percentiles of the distance between random pairs of different shapes."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(index.names), size=n_pairs)
    b = rng.integers(0, len(index.names), size=n_pairs)
    keep = a != b
    d = np.linalg.norm(index.features[a[keep]] - index.features[b[keep]], axis=1)
    return [float(np.percentile(d, p)) for p in percentiles]


def run(index, percentiles=PERCENTILES, n_queries=N_QUERIES, seed=0):
    X = index.features
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(index.names), size=min(n_queries, len(index.names)), replace=False)
    radii = [round(r, 3) for r in pair_distance_percentiles(index, percentiles)]

    # Warm-up: about 1 second of realistic work, so the CPU is at full speed before measuring
    warm_r = radii[len(radii) // 2]
    end = time.perf_counter() + 1.0
    i = 0
    while time.perf_counter() < end:
        q = X[queries[i % len(queries)]]
        tree_range(index, q, warm_r)
        brute_force_range(X, q, warm_r)
        i += 1

    stats = [{"tree": [], "brute": [], "results": 0, "calls": 0,
              "same_set": 0, "same_distances": 0} for _ in radii]

    gc.disable()        # no garbage-collection pauses in the middle of a measurement
    try:
        # Queries in the outer loop, radii in the inner loop: a slow moment of the laptop
        # is spread over all radii instead of hitting one radius
        for qi in queries:
            q = X[qi]
            for s, r in zip(stats, radii):
                index.tree.reset_n_calls()
                start = time.perf_counter()
                ia, da = tree_range(index, q, r)
                s["tree"].append(time.perf_counter() - start)
                s["calls"] += index.tree.get_n_calls()

                start = time.perf_counter()
                ib, db = brute_force_range(X, q, r)
                s["brute"].append(time.perf_counter() - start)

                s["results"] += len(ia) - 1          # minus the query shape itself
                s["same_set"] += set(ia.tolist()) == set(ib.tolist())
                s["same_distances"] += len(da) == len(db) and np.allclose(
                    da, db, rtol=0, atol=1e-9)
    finally:
        gc.enable()

    n = len(queries)
    rows = []
    for p, r, s in zip(percentiles, radii, stats):
        rows.append({
            "percentile": p,
            "radius": r,
            "avg_results": s["results"] / n,
            "tree_ms": 1000 * float(np.median(s["tree"])),      # median: ignores outliers
            "brute_ms": 1000 * float(np.median(s["brute"])),
            "tree_distance_calls": s["calls"] / n,
            "same_set": s["same_set"],
            "same_distances": s["same_distances"],
            "n_queries": n,
        })
    return rows


# --- Output -------------------------------------------------------------------------

def print_table(rows, n_shapes):
    print("\nMedian time per query in ms")
    print(f"{'pct':>5} {'R':>8} {'found':>7} {'% db':>6} {'tree ms':>8} {'brute ms':>9} "
          f"{'speed-up':>8} {'tree dist.':>10} {'% db':>6} {'same set':>10}")
    for r in rows:
        print(f"{r['percentile']:>5g} {r['radius']:>8.3f} {r['avg_results']:>7.1f} "
              f"{100 * r['avg_results'] / n_shapes:>5.1f}% {r['tree_ms']:>8.3f} "
              f"{r['brute_ms']:>9.3f} {r['brute_ms'] / r['tree_ms']:>7.1f}x "
              f"{r['tree_distance_calls']:>10.0f} "
              f"{100 * r['tree_distance_calls'] / n_shapes:>5.1f}% "
              f"{r['same_set']:>4}/{r['n_queries']:<4}")
    n_wrong = sum(r["n_queries"] - r["same_distances"] for r in rows)
    print(f"\nQueries where tree and brute force differ (all radii): {n_wrong}")


def save_csv(rows, path):
    fields = ["percentile", "radius", "avg_results", "tree_ms", "brute_ms",
              "tree_distance_calls", "same_set", "same_distances", "n_queries"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fields} for r in rows])


def plot(rows, n_shapes, path):
    radius = [r["radius"] for r in rows]
    tree_ms = [r["tree_ms"] for r in rows]
    brute_ms = [r["brute_ms"] for r in rows]
    found_pct = [100 * r["avg_results"] / n_shapes for r in rows]
    calls_pct = [100 * r["tree_distance_calls"] / n_shapes for r in rows]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True)

    # Left: time per query
    ax1.plot(radius, tree_ms, color=TREE_COLOR, lw=2, marker="o", ms=5, label="k-d tree")
    ax1.plot(radius, brute_ms, color=BRUTE_COLOR, lw=2, marker="s", ms=5, label="Brute force")
    ax1.set_xlabel("Radius R")
    ax1.set_ylabel("Time per query (ms)")
    ax1.set_title("Query time", loc="left")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    # Right: why - how much of the database is found, and how much the tree had to check
    ax2.plot(radius, calls_pct, color=TREE_COLOR, lw=2, marker="o", ms=5,
             label="Distances computed by tree")
    ax2.plot(radius, found_pct, color="#52514e", lw=2, marker="^", ms=5, ls="--",
             label="Shapes within R")
    ax2.axhline(100, color="#bdbcb6", lw=1, zorder=0)
    ax2.set_xlabel("Radius R")
    ax2.set_ylabel("% of database")
    ax2.set_title("Work done by the tree", loc="left")
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\nPlot saved to {path}")


if __name__ == "__main__":
    index = ShapeIndex.from_file(weighted=WEIGHTED)
    n_shapes = len(index.names)
    print(f"Loaded {n_shapes} shapes with {index.features.shape[1]} features each "
          f"({'weighted' if WEIGHTED else 'plain L2'}); {N_QUERIES} queries per radius")

    rows = run(index)
    print_table(rows, n_shapes)
    save_csv(rows, OUTPUT_DIR / "rnn_benchmark.csv")
    plot(rows, n_shapes, OUTPUT_DIR / "rnn_benchmark.png")