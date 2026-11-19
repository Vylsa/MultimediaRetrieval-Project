"""Step 5: scalable retrieval with a spatial search structure (k-d tree).

The normalized feature database (normalized_features.xlsx, sheet 'zscore') is loaded once
(excluding TEST_CLASSES) and stored in a k-d tree. A query is then answered without
comparing it to every shape:
  * K-nearest neighbors (KNN): the K shapes with the smallest distance to the query
  * R-nearest neighbors (range search): all shapes within distance R of the query
The distance is the Euclidean (L2) distance in the shared transformed feature space.

Without group scaling, the 6 standardized single values dominate the L2 distance (over 99%
of the squared distance in our database), because a histogram bin varies much less than a
z-score. A k-d tree can only use a plain Lp distance, so the Hellinger square-root and
group multipliers are applied to the vectors themselves via `apply_feature_transform`.

We use scikit-learn's KDTree (eps=0, exact nearest neighbors matching brute_force_knn).
"""
import os
import time

import numpy as np
from openpyxl import load_workbook
from sklearn.neighbors import KDTree

from .config import DEFAULT_K, TEST_CLASSES, WEIGHTED
from .dr_features import (apply_feature_transform, fit_feature_transform,
                          load_feature_transform, save_feature_transform)
from .dr_paths import Variant, feature_transform_path

FEATURES_PATH = os.path.join("sheets", "normalized_features.xlsx")
DEFAULT_VARIANT = Variant(hellinger=True, best_shares=True)
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


def load_feature_database(path=None, sheet="zscore", exclude_classes=TEST_CLASSES):
    """Read the normalized feature database (excluding non-shape test classes).
    Returns (names, classes, features, feature_names): one row of features per shape."""
    path = resolve_features_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature database not found: {path}\n"
                                f"Set FEATURES_PATH at the top of knn.py.")
    excluded = set(exclude_classes or ())
    ws = load_workbook(path, read_only=True)[sheet]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    names, classes, features = [], [], []
    for row in rows:
        if row[0] is None or row[1] in excluded:
            continue
        names.append(row[0])
        classes.append(row[1])
        features.append([float(v) for v in row[2:]])
    return names, classes, np.array(features), header[2:]


class ShapeIndex:
    """The feature database in a k-d tree, for fast K-nearest and range queries."""

    def __init__(self, names, classes, features, feature_names=None,
                 weighted=WEIGHTED, transform_spec=None,
                 variant=DEFAULT_VARIANT, leaf_size=30):
        self.names = list(names)
        self.classes = list(classes)
        self.weighted = weighted
        raw_features = np.asarray(features, dtype=float)

        if transform_spec is not None:
            self.transform_spec = transform_spec
        elif not weighted:
            self.transform_spec = fit_feature_transform(raw_features, Variant())
        else:
            spec_path = feature_transform_path(variant)
            if os.path.isfile(spec_path):
                self.transform_spec = load_feature_transform(spec_path)
            else:
                self.transform_spec = fit_feature_transform(raw_features, variant)
                save_feature_transform(self.transform_spec, path=spec_path,
                                       variant=variant)

        self.scales = np.asarray(self.transform_spec["multipliers"], dtype=float)
        self.features = apply_feature_transform(raw_features, self.transform_spec)

        start = time.perf_counter()
        self.tree = KDTree(self.features, leaf_size=leaf_size, metric="euclidean")
        self.build_time = time.perf_counter() - start
        self._row = {name: i for i, name in enumerate(self.names)}

    @classmethod
    def from_file(cls, path=None, weighted=WEIGHTED, transform_spec=None,
                  variant=DEFAULT_VARIANT, exclude_classes=TEST_CLASSES):
        """Build the index from the feature file (excluding TEST_CLASSES)."""
        names, classes, features, feature_names = load_feature_database(
            path, exclude_classes=exclude_classes
        )
        return cls(names, classes, features, feature_names, weighted=weighted,
                   transform_spec=transform_spec, variant=variant)

    def prepare_query(self, normalized_vector):
        """Apply the shared Hellinger + group-multiplier transform to a query vector."""
        return apply_feature_transform(normalized_vector, self.transform_spec)

    def vector_of(self, name):
        """The transformed feature vector of a shape that is in the database, by file name."""
        return self.features[self._row[name]]

    def class_precision(self, k=10):
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
        Used as the custom linear-scan engine and to verify identical results."""
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