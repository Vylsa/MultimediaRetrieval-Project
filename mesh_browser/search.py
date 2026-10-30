"""Raw mesh -> features -> normalized vector -> ranked matches."""
import numpy as np

from .feature_normalization import (load_descriptor_table, load_stats,
                                    standardize, normalize_feature_vector)
from .query import extract_query_features, compute_group_scales, find_matches, N_SINGLE


class SearchIndex:
    def __init__(self, descriptor_path="sheets/descriptors.xlsx", stats_path="feature_stats.json",
                 hist_distance="emd", weights=None):
        names, classes, single, hists, _ = load_descriptor_table(descriptor_path)
        self.stats = load_stats(stats_path)       # must match descriptors.xlsx (same run)
        self.names, self.classes = names, classes
        self.features = np.hstack([standardize(single, self.stats), hists])
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