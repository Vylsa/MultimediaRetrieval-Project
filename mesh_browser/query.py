"""Feature extraction for a query shape (Step 8, part 1).

A query mesh can come from anywhere, so it is raw: any position, scale, orientation
and vertex count. The database features were computed on meshes that went through
clean -> resample -> normalize, so the query has to go through exactly the same
stages. Otherwise its features are not comparable to the database features.
"""
import os

import open3d as o3d
import numpy as np

from . import mesh_utils as utils
from .descriptors1 import compute_descriptors, feature_vector
from .descriptors import DISTRIBUTION_NAMES, N_BINS

N_SINGLE = 6  # number of single-value features at the start of the vector

# Distance function used for the histograms
HIST_DISTANCE = "emd" 


# Slice of the feature vector that belongs to each group
GROUPS = {"scalars": slice(0, N_SINGLE)}
for _i, _name in enumerate(DISTRIBUTION_NAMES):
    GROUPS[_name] = slice(N_SINGLE + _i * N_BINS, N_SINGLE + (_i + 1) * N_BINS)

DEFAULT_WEIGHTS = {group: 1.0 for group in GROUPS}

def prepare_query_mesh(mesh):
    """Bring a raw mesh to the same state as the database meshes.
    Returns (normalized mesh, resample actions). The input mesh is not modified."""
    mesh = o3d.geometry.TriangleMesh(mesh)      # copy: normalize_mesh works in place
    mesh = utils.clean_mesh(mesh)
    if not mesh.has_triangles():
        raise ValueError("The mesh has no usable triangles after cleaning.")
    mesh, actions = utils.resample_mesh(mesh)
    mesh = utils.normalize_mesh(mesh)
    return mesh, actions


def extract_query_features(mesh):
    """Features of a raw query mesh.

    Returns (features, row, normalized_mesh):
      features         NumPy vector in the order of FEATURE_COLUMNS (None -> NaN).
                       NaNs and scaling are handled later, with database statistics.
      row              the full descriptor dict (includes 'quality' and 'volume')
      normalized_mesh  the mesh that was actually measured, handy for display
    """
    normalized, _ = prepare_query_mesh(mesh)
    row = compute_descriptors(normalized)
    return feature_vector(row), row, normalized


def extract_query_features_from_file(path):
    """Load an .obj file and extract its features. Raises ValueError if it can't be loaded."""
    mesh = utils.load_mesh(path)
    if mesh is None:
        raise ValueError(f"Could not load {os.path.basename(path)}")
    return extract_query_features(mesh)

"""Step 4, part 2: distances between a query shape and all database shapes.

Both the query and the database use the normalized feature vector:
  [6 standardized single values] + [A3 (10 bins)] + [D1] + [D2] + [D3] + [D4]
The vector is split into 6 groups. Each group gets its own distance, the group
distances are put on a common scale, and then combined as a weighted sum:
  scalars        Euclidean distance over the 6 standardized values
  A3 ... D4      Earth Mover's Distance (or L1/Euclidean) over the 10 bins
"""

# --- Distance functions (work on one pair or on a whole matrix) ------------------------

def euclidean(a, b):
    return np.linalg.norm(a - b, axis=-1)


def l1(a, b):
    return np.abs(a - b).sum(axis=-1)


def emd(a, b):
    """Earth Mover's Distance between ordered 1D histograms with equal total mass:
    the L1 distance between the cumulative sums. Unlike L1, it knows that bin 3 is
    closer to bin 4 than to bin 9."""
    return np.abs(np.cumsum(a - b, axis=-1)).sum(axis=-1)


HISTOGRAM_DISTANCES = {"emd": emd, "l1": l1, "euclidean": euclidean}


# --- Group distances ------------------------------------------------------------------------

def group_distances(query, database, hist_distance=HISTOGRAM_DISTANCES):
    """Distance per group. query: (56,), database: (n, 56) (or a second (56,) vector).
    Returns {group: array of n distances}."""
    hist_fn = HISTOGRAM_DISTANCES[hist_distance]
    result = {}
    for group, cols in GROUPS.items():
        fn = euclidean if group == "scalars" else hist_fn
        result[group] = fn(database[..., cols], query[cols])
    return result


def compute_group_scales(database, hist_distance=HISTOGRAM_DISTANCES, n_pairs=5000, seed=0):
    """Typical distance per group: the mean over random pairs of database shapes.
    Dividing by it makes the groups comparable (otherwise the group with the largest
    numbers decides the ranking). Compute once per database and keep the result."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(database), size=n_pairs)
    j = rng.integers(0, len(database), size=n_pairs)
    keep = i != j
    d = group_distances_pairs(database[i[keep]], database[j[keep]], hist_distance)
    return {group: max(float(values.mean()), 1e-12) for group, values in d.items()}



def group_distances_pairs(a, b, hist_distance=HISTOGRAM_DISTANCES):
    """Like group_distances, but for two matrices of equal shape (row against row)."""
    hist_fn = HISTOGRAM_DISTANCES[hist_distance]
    return {group: (euclidean if group == "scalars" else hist_fn)(a[:, cols], b[:, cols])
            for group, cols in GROUPS.items()}


# --- Combining and ranking ---------------------------------------------------------------
def compute_distances(query, database, scales, weights=None, hist_distance=HISTOGRAM_DISTANCES):
    """Combined distance of the query to every database shape.
    Returns (total, parts): total is an array of n distances, parts the scaled
    per-group distances (useful to see WHY a shape matched)."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    raw = group_distances(np.asarray(query, dtype=float), database, hist_distance)
    parts = {group: d / scales[group] for group, d in raw.items()}
    total = sum(weights.get(group, 0.0) * d for group, d in parts.items())
    return total, parts


def rank_matches(distances, k=None, threshold=None, exclude_index=None):
    """Indices sorted from low to high distance, plus those distances.
    k: keep the K best. threshold: keep only distances <= threshold (the combined
    distance is relative to the average pair of shapes, so 1.0 is about 'typical').
    exclude_index: leave one database shape out (the query itself, if it is in there)."""
    order = np.argsort(distances)
    if exclude_index is not None:
        order = order[order != exclude_index]
    if threshold is not None:
        order = order[distances[order] <= threshold]
    if k is not None:
        order = order[:k]
    return order, distances[order]


def find_matches(query, database, names, classes, scales, weights=None,
                 hist_distance=HISTOGRAM_DISTANCES, k=10, threshold=None, exclude_index=None):
    """Everything in one call. Returns a list of dicts, best match first:
    {"index", "name", "class", "distance", "parts"}."""
    total, parts = compute_distances(query, database, scales, weights, hist_distance)
    order, dists = rank_matches(total, k, threshold, exclude_index)
    return [{"index": int(i), "name": names[i], "class": classes[i],
             "distance": float(d), "parts": {g: float(p[i]) for g, p in parts.items()}}
            for i, d in zip(order, dists)]