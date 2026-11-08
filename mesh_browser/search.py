# """Raw mesh -> features -> normalized vector -> ranked matches."""
# import numpy as np

# from .feature_normalization import (load_descriptor_table, load_stats,
#                                     standardize, normalize_feature_vector)
# from .query import extract_query_features, compute_group_scales, find_matches, N_SINGLE


# class SearchIndex:
#     def __init__(self, descriptor_path="sheets/descriptors.xlsx", stats_path="feature_stats.json",
#                  hist_distance="emd", weights=None):
#         names, classes, single, hists, _ = load_descriptor_table(descriptor_path)
#         self.stats = load_stats(stats_path)       # must match descriptors.xlsx (same run)
#         self.names, self.classes = names, classes
#         self.features = np.hstack([standardize(single, self.stats), hists])
#         self.hist_distance = hist_distance
#         self.weights = weights
#         self.scales = compute_group_scales(self.features, hist_distance)

#     def search(self, mesh, k=10, threshold=None):
#         """mesh: a raw query mesh (any position, scale, orientation).
#         Returns (matches, row, normalized_mesh); see matching.find_matches for the
#         match format."""
#         features, row, normalized = extract_query_features(mesh)
#         query = normalize_feature_vector(features[:N_SINGLE], features[N_SINGLE:], self.stats)
#         matches = find_matches(query, self.features, self.names, self.classes, self.scales,
#                                self.weights, self.hist_distance, k, threshold)
#         return matches, row, normalized

"""Step 8, part 3: the search index. Loads the standardized database features once,
then answers queries: raw mesh -> features -> normalized vector -> ranked matches."""
import numpy as np
from openpyxl import load_workbook

from .descriptors import DISTRIBUTION_NAMES, N_BINS
# NOTE: adjust the module name to wherever the Step 4a normalization code lives.
from .feature_normalization import load_stats, normalize_feature_vector
from .query import extract_query_features, compute_group_scales, find_matches, N_SINGLE

N_FEATURES = N_SINGLE + len(DISTRIBUTION_NAMES) * N_BINS   # 6 + 5 * N_BINS


def load_standardized_table(path, sheet):
    """Read the standardized feature sheet (columns: name, class, then the features).
    Returns (names, classes, matrix with one row per shape)."""
    ws = load_workbook(path, read_only=True, data_only=True)[sheet]
    rows = ws.iter_rows(values_only=True)
    header = [h for h in next(rows) if h is not None]
    n_features = len(header) - 2
    if n_features != N_FEATURES:
        raise ValueError(
            f"{sheet} has {n_features} feature columns, but N_BINS={N_BINS} in "
            f"descriptors.py gives {N_FEATURES}. The sheet and N_BINS must match.")

    names, classes, features = [], [], []
    for row in rows:
        if row[0] is None:
            continue
        names.append(row[0])
        classes.append(row[1])
        features.append(row[2:2 + n_features])
    return names, classes, np.array(features, dtype=float)


class SearchIndex:
    def __init__(self, standardized_path, sheet, stats_path, exclude_classes=(),
                 hist_distance="emd", weights=None):
        """standardized_path/sheet: the z-score sheet of the normalized features.
        stats_path: feature_stats.json, needed to standardize the QUERY the same way.
        exclude_classes: classes that are not real shapes (e.g. the test folders)."""
        names, classes, features = load_standardized_table(standardized_path, sheet)
        keep = [i for i, c in enumerate(classes) if c not in set(exclude_classes)]
        self.names = [names[i] for i in keep]
        self.classes = [classes[i] for i in keep]
        self.features = features[keep]
        self.stats = load_stats(stats_path)
        self.hist_distance = hist_distance
        self.weights = weights
        self.scales = compute_group_scales(self.features, hist_distance)

    def search(self, mesh, k=10, threshold=None):
        """mesh: a raw query mesh (any position, scale, orientation).
        Returns (matches, row, normalized_mesh); see matching.find_matches for the
        match format."""
        features, row, normalized = extract_query_features(mesh)
        query = normalize_feature_vector(features[:N_SINGLE], features[N_SINGLE:], self.stats)
        matches = find_matches(query, self.features, self.names, self.classes, self.scales,
                               self.weights, self.hist_distance, k, threshold)
        return matches, row, normalized