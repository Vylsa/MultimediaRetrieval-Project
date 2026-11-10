"""Compare retrieval quality (class precision@K) of the custom distance and the k-d tree KNN.

For every shape in the database: take its K nearest shapes (excluding itself) and count
how many have the same class. The precision is the average fraction over all shapes.
"""
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser import config
from mesh_browser.search import SearchIndex
from mesh_browser.query import compute_group_scales, compute_distances, rank_matches  # <-- adjust module name
from mesh_browser.knn import ShapeIndex, SINGLE_VALUE_FEATURES, HISTOGRAM_NAMES

K = 5


def precision_custom(features, classes, k=K, weights=None, hist="emd"):
    """Class precision@k of the custom distance, with every database shape as query once."""
    scales = compute_group_scales(features, hist)
    hits = 0
    for i in range(len(features)):
        total, _ = compute_distances(features[i], features, scales, weights, hist)
        order, _ = rank_matches(total, k, exclude_index=i)
        hits += sum(classes[j] == classes[i] for j in order)
    return hits / (k * len(features))


if __name__ == "__main__":
    idx = SearchIndex(config.DESCRIPTORS_FILE, config.STATS_FILE)
    F, C = idx.features, idx.classes
    print(f"{len(F)} shapes, {F.shape[1]} features, K = {K}\n")

    hist_groups = ["A3", "D1", "D2", "D3", "D4"]    # must match DISTRIBUTION_NAMES
    scalars_6 = {"scalars": 6.0, **{h: 1.0 for h in hist_groups}}

    print(f"custom, default weights:      {precision_custom(F, C):.3f}")
    print(f"custom, scalars weight 6:     {precision_custom(F, C, weights=scalars_6):.3f}")
    print(f"custom, euclidean histograms: {precision_custom(F, C, hist='euclidean'):.3f}")

    n_bins = (F.shape[1] - len(SINGLE_VALUE_FEATURES)) // len(HISTOGRAM_NAMES)
    names = SINGLE_VALUE_FEATURES + [f"{h}_{i}" for h in HISTOGRAM_NAMES for i in range(n_bins)]
    for weighted in (True, False):
        p = ShapeIndex(idx.names, C, F, names, weighted=weighted).class_precision(k=K)
        print(f"knn, weighted={weighted!s:<5}:          {p:.3f}")