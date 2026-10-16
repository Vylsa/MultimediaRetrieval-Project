"""Shape-property distribution descriptors (Step 3.2): A3, D1, D2, D3, D4.

Each descriptor is computed for many random samples on a shape and reduced to a
normalized histogram of N_BINS values (the bins sum to 1).

Expects normalized meshes (barycenter at the origin, inside the unit cube), because
the histogram ranges in DISTRIBUTION_RANGES are fixed for that case.
"""
import numpy as np

# --- Settings ------------------------------------------------------------------

N_SAMPLES = 100_000  # random samples per descriptor
N_BINS = 10          # B, the number of histogram bins per descriptor
SEED = 42            # fixed seed, so the histograms are reproducible

# "vertices": random mesh vertices, as in the assignment
# "surface":  random points spread uniformly over the surface (area-weighted),
#             independent of how the mesh is triangulated
SAMPLE_SOURCE = "vertices"
N_SURFACE_POINTS = 10_000  # only used when SAMPLE_SOURCE == "surface"

# Fixed histogram ranges, the same for every shape (valid for shapes in the unit cube).
# Values outside the range are clipped into the first/last bin.
DISTRIBUTION_RANGES = {
    "A3": (0.0, np.pi),          # angle
    "D1": (0.0, np.sqrt(3)),     # at most the diagonal of the unit cube
    "D2": (0.0, np.sqrt(3)),
    "D3": (0.0, 1.0),            # sqrt of a triangle area inside the unit cube
    "D4": (0.0, np.cbrt(1 / 3)), # largest tetrahedron in the unit cube has volume 1/3
}
DISTRIBUTION_NAMES = list(DISTRIBUTION_RANGES)

EPS = 1e-12  # guards against division by zero for degenerate samples


# --- Descriptor values -------------------------------------------------------

def sample_points(points, k, n, rng):
    """n random samples of k distinct points. Returns a list of k arrays of shape (<=n, 3);
    samples where the same point is picked twice are dropped (they would give
    zero lengths, areas and undefined angles)."""
    idx = rng.integers(0, len(points), size=(n, k))
    if k > 1:
        s = np.sort(idx, axis=1)
        idx = idx[np.all(s[:, 1:] != s[:, :-1], axis=1)]
    return [points[idx[:, j]] for j in range(k)]


def a3_values(points, n, rng):
    """A3: angle at b between (a - b) and (c - b), for random point triples."""
    a, b, c = sample_points(points, 3, n, rng)
    u, w = a - b, c - b
    norms = np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)
    cos = np.einsum("ij,ij->i", u, w) / np.maximum(norms, EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))  # clip: rounding can give |cos| > 1


def d1_values(points, barycenter, n, rng):
    """D1: distance between the barycenter and random points. With few points
    (at most n) all of them are used, which gives the exact distribution."""
    if len(points) > n:
        points = points[rng.integers(0, len(points), size=n)]
    return np.linalg.norm(points - barycenter, axis=1)


def d2_values(points, n, rng):
    """D2: distance between random point pairs."""
    a, b = sample_points(points, 2, n, rng)
    return np.linalg.norm(a - b, axis=1)


def d3_values(points, n, rng):
    """D3: square root of the area of the triangle of random point triples."""
    a, b, c = sample_points(points, 3, n, rng)
    return np.sqrt(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1))


def d4_values(points, n, rng):
    """D4: cube root of the volume of the tetrahedron of random point quadruples."""
    a, b, c, d = sample_points(points, 4, n, rng)
    volumes = np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))) / 6.0
    return np.cbrt(volumes)


def normalized_histogram(values, value_range, n_bins=N_BINS):
    """Histogram over a fixed range, divided by the number of values so the bins sum to 1."""
    lo, hi = value_range
    hist, _ = np.histogram(np.clip(values, lo, hi), bins=n_bins, range=value_range)
    return hist / max(len(values), 1)


def sampling_points(mesh, source=SAMPLE_SOURCE):
    """The points the distributions are computed from (see SAMPLE_SOURCE)."""
    if source == "surface":
        return np.asarray(mesh.sample_points_uniformly(number_of_points=N_SURFACE_POINTS).points)
    return np.asarray(mesh.vertices)


def distribution_values(mesh, n_samples=N_SAMPLES, seed=SEED, source=SAMPLE_SOURCE):
    """The raw values of the five distributions as a dict {name: 1D array}.
    Handy for plotting or for choosing the histogram ranges."""
    points = sampling_points(mesh, source)
    barycenter = np.asarray(mesh.vertices).mean(axis=0)
    rng = np.random.default_rng(seed)
    return {
        "A3": a3_values(points, n_samples, rng),
        "D1": d1_values(points, barycenter, n_samples, rng),
        "D2": d2_values(points, n_samples, rng),
        "D3": d3_values(points, n_samples, rng),
        "D4": d4_values(points, n_samples, rng),
    }


def distribution_descriptors(mesh, n_samples=N_SAMPLES, n_bins=N_BINS, seed=SEED,
                             source=SAMPLE_SOURCE):
    """The five normalized histograms as a dict {name: array of n_bins values}."""
    values = distribution_values(mesh, n_samples, seed, source)
    return {name: normalized_histogram(vals, DISTRIBUTION_RANGES[name], n_bins)
            for name, vals in values.items()}


# --- Histogram feature vector ---------------------------------------------------

def histogram_feature_names(n_bins=N_BINS):
    """Names for every entry of the histogram vector: 'A3_0', ..., 'D4_<n_bins-1>'."""
    return [f"{name}_{i}" for name in DISTRIBUTION_NAMES for i in range(n_bins)]


def histogram_feature_vector(mesh, n_samples=N_SAMPLES, n_bins=N_BINS, seed=SEED,
                             source=SAMPLE_SOURCE):
    """The five histograms concatenated (5 * n_bins values, in the order of
    histogram_feature_names())."""
    hists = distribution_descriptors(mesh, n_samples, n_bins, seed, source)
    return np.concatenate([hists[name] for name in DISTRIBUTION_NAMES])