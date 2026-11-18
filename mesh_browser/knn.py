"""Step 5: scalable retrieval with a spatial search structure (k-d tree).

The normalized feature database (normalized_features.xlsx, sheet 'zscore') is loaded once
and stored in a k-d tree. A query is then answered without comparing it to every shape:
  * K-nearest neighbors (KNN): the K shapes with the smallest distance to the query
  * R-nearest neighbors (range search): all shapes within distance R of the query
The distance is the Euclidean (L2) distance between the normalized feature vectors.

Optionally (weighted=True) the feature groups are rescaled first, so that each of the 11
features (6 single values + 5 histograms) contributes equally to the distance on average.
Without this, the 6 standardized single values dominate the L2 distance (about 95% of it
in our database), because a histogram bin varies much less than a z-score. A k-d tree can
only use a plain Lp distance, so the weighting is applied to the vectors themselves.

We use scikit-learn's KDTree. The ANN library suggested in the assignment is also based on
k-d trees; it additionally allows an approximation factor eps, which is not needed for a
database of this size (with eps = 0, ANN gives exactly the same results as a k-d tree).
"""
import os
import time

import numpy as np
from openpyxl import load_workbook
from sklearn.neighbors import KDTree

from .config import WEIGHTED, DEFAULT_K



FEATURES_PATH = os.path.join("sheets", "normalized_features.xlsx")
# ------------------------------------------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_features_path(path=None):
    """The full path of the feature file: FEATURES_PATH by default, and relative paths
    are taken relative to the project folder."""
    path = path or FEATURES_PATH
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


SINGLE_VALUE_FEATURES = ["area", "compactness", "rectangularity",
                         "convexity", "diameter", "eccentricity"]
HISTOGRAM_NAMES = ["A3", "D1", "D2", "D3", "D4"]


def feature_groups(feature_names):
    """Column indices per feature: each single value is its own group, each histogram
    (all its bins together) is one group."""
    groups = {name: [feature_names.index(name)] for name in SINGLE_VALUE_FEATURES}
    for h in HISTOGRAM_NAMES:
        groups[h] = [i for i, f in enumerate(feature_names) if f.startswith(h + "_")]
    return groups


def group_scales(features, groups, sample=5000, seed=0):
    """Per column, the factor that makes every group contribute equally to the L2 distance:
    1 / sqrt(average squared distance of that group between random pairs of shapes).
    After scaling, each group adds on average 1 to the squared distance."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(features), size=sample)
    b = rng.integers(0, len(features), size=sample)
    diff2 = (features[a] - features[b]) ** 2
    scales = np.ones(features.shape[1])
    for cols in groups.values():
        mean_sq = diff2[:, cols].sum(axis=1).mean()
        scales[cols] = 1.0 / np.sqrt(max(mean_sq, 1e-12))
    return scales




def load_feature_database(path=None, sheet="zscore"):
    """Read the normalized feature database (FEATURES_PATH unless another path is given).
    Returns (names, classes, features, feature_names): one row of features per shape."""
    path = resolve_features_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature database not found: {path}\n"
                                f"Set FEATURES_PATH at the top of knn.py.")
    ws = load_workbook(path, read_only=True)[sheet]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    names, classes, features = [], [], []
    for row in rows:
        if row[0] is None:
            continue
        names.append(row[0])
        classes.append(row[1])
        features.append([float(v) for v in row[2:]])
    return names, classes, np.array(features), header[2:]


class ShapeIndex:
    """The feature database in a k-d tree, for fast K-nearest and range queries."""

    def __init__(self, names, classes, features, feature_names=None, weighted=WEIGHTED,
                 leaf_size=30):
        self.names = list(names)
        self.classes = list(classes)
        self.weighted = weighted
        features = np.asarray(features, dtype=float)
        if weighted:
            self.scales = group_scales(features, feature_groups(feature_names))
        else:
            self.scales = np.ones(features.shape[1])
        self.features = features * self.scales     # the vectors the tree searches in
        start = time.perf_counter()
        self.tree = KDTree(self.features, leaf_size=leaf_size, metric="euclidean")
        self.build_time = time.perf_counter() - start
        self._row = {name: i for i, name in enumerate(self.names)}

    @classmethod
    def from_file(cls, path=None, weighted=WEIGHTED):
        """Build the index from the feature file (FEATURES_PATH unless another path is given)."""
        names, classes, features, feature_names = load_feature_database(path)
        return cls(names, classes, features, feature_names, weighted)
    

    def prepare_query(self, normalized_vector):
        """Apply the same weighting to a query's normalized feature vector (for a query
        shape that is not in the database), so it can be searched in the tree."""
        return np.asarray(normalized_vector, dtype=float) * self.scales

    def vector_of(self, name):
        """The (weighted) feature vector of a shape that is in the database, by file name."""
        return self.features[self._row[name]]

    def class_precision(self, k=6):
        """Average fraction of the k nearest neighbors that have the same class as the
        query, over all shapes in the database: a simple measure of retrieval quality."""
        hits = 0
        for name, cls in zip(self.names, self.classes):
            hits += sum(r[1] == cls for r in self.knn_by_name(name, k))
        return hits / (k * len(self.names))

    def _results(self, indices, distances, exclude):
        """List of (name, class, distance), nearest first, without the excluded shape."""
        results = [(self.names[i], self.classes[i], float(d))
                   for i, d in zip(indices, distances) if self.names[i] != exclude]
        return sorted(results, key=lambda r: r[2])

    def knn(self, query_vector, k=DEFAULT_K, exclude=None):
        """The k shapes nearest to the query vector. If the query shape is itself in the
        database, pass its name as exclude, so it is not returned as its own neighbor."""
        k_search = min(k + (1 if exclude else 0), len(self.names))
        distances, indices = self.tree.query(np.asarray(query_vector)[None, :], k=k_search)
        return self._results(indices[0], distances[0], exclude)[:k]

    def range_search(self, query_vector, radius, exclude=None):
        """All shapes within distance radius of the query vector, nearest first."""
        indices, distances = self.tree.query_radius(np.asarray(query_vector)[None, :],
                                                    r=radius, return_distance=True,
                                                    sort_results=True)
        return self._results(indices[0], distances[0], exclude)

    def knn_by_name(self, name, k=DEFAULT_K):
        """KNN for a shape from the database (the shape itself is left out)."""
        return self.knn(self.vector_of(name), k, exclude=name)

    def range_by_name(self, name, radius):
        """Range search for a shape from the database (the shape itself is left out)."""
        return self.range_search(self.vector_of(name), radius, exclude=name)

    def brute_force_knn(self, query_vector, k=DEFAULT_K, exclude=None):
        """The same KNN query by computing the distance to every shape (no tree).
        Used to check that the tree gives identical results, and to compare speed."""
        distances = np.linalg.norm(self.features - np.asarray(query_vector), axis=1)
        order = np.argsort(distances)[:k + (1 if exclude else 0)]
        return self._results(order, distances[order], exclude)[:k]

    def distance_percentiles(self, sample=500, seed=0):
        """Percentiles of the distance between random pairs of shapes, to help choose a
        sensible radius R for range search."""
        rng = np.random.default_rng(seed)
        a = rng.integers(0, len(self.names), size=sample)
        b = rng.integers(0, len(self.names), size=sample)
        d = np.linalg.norm(self.features[a] - self.features[b], axis=1)
        return {p: float(np.percentile(d[a != b], p)) for p in (1, 5, 10, 25, 50)}


def print_results(title, results):
    print(f"\n{title}")
    if not results:
        print("  (no shapes found)")
    for rank, (name, cls, dist) in enumerate(results, start=1):
        print(f"  {rank:>3}. {cls:<20} {name:<20} distance {dist:.3f}")