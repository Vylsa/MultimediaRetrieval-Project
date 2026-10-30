import sys
from pathlib import Path

# Adds the parent project root directory to Python's path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser.query import extract_query_features_from_file, compute_group_scales, find_matches
from mesh_browser.feature_normalization import normalize_database_features, normalize_feature_vector, load_stats


path = "C:\\Users\\pjotr\\Documents\\UU\\Multimedia retrieval\\MultimediaRetrieval-Project\\ShapeDatabase_INFOMR-master\\AircraftBuoyant\\m1337.obj"

features, row, normalized_mesh = extract_query_features_from_file(path)   # 56 raw values
stats = load_stats("feature_stats.json")
query = normalize_feature_vector(features[:6], features[6:], stats)       # log + z-score

names, classes, db = normalize_database_features()                        # (N, 56) matrix
scales = compute_group_scales(db)                                          # once, keep it
print ("scales:", scales)

matches = find_matches(query, db, names, classes, scales, k=10)

for match in matches:
    print(f"Index: {match['index']}, Name: {match['name']}, Class: {match['class']}, Distance: {match['distance']:.4f}")