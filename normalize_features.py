"""Normalizes the feature database (descriptors.xlsx) and writes:
  normalized_features.xlsx  the normalized database, one sheet per version, to inspect
  feature_stats.json        the database statistics, used later to normalize query shapes
Run after the descriptors have been computed for the whole database."""
from mesh_browser.feature_normalization import normalize_database_features

DESCRIPTOR_PATH = "descriptors.xlsx"
OUTPUT_PATH = "normalized_features.xlsx"
STATS_PATH = "feature_stats.json"

if __name__ == "__main__":
    names, classes, features = normalize_database_features(DESCRIPTOR_PATH, OUTPUT_PATH, STATS_PATH)
    print(f"Normalized {len(names)} shapes, {features.shape[1]} features each")
    print(f"Written to {OUTPUT_PATH} and {STATS_PATH}")