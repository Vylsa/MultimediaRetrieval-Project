"""Step 4a: normalize the feature database so all features can be combined in one distance.

Following the technical tips:
  * single-value features: standardization (z-score) per feature, using the mean and
    standard deviation over the whole database. Min-max normalization is computed too,
    for comparison.
  * histogram features: already normalized by their area (every histogram sums to 1),
    so they are copied unchanged.

The database statistics are saved to a JSON file, so a query shape that is not in the
database can be normalized with exactly the same numbers later.
"""
import json

import numpy as np
from openpyxl import Workbook, load_workbook

SINGLE_VALUE_FEATURES = ["area", "compactness", "rectangularity",
                         "convexity", "diameter", "eccentricity"]
HISTOGRAM_PREFIXES = ("A3_", "D1_", "D2_", "D3_", "D4_")

EPS = 1e-12  # guards against dividing by a zero standard deviation or range


# --- Reading the descriptor table ----------------------------------------------------

def _to_float(value):
    """Cell value -> float, with missing values ('None', empty, text) as NaN."""
    if value is None or value == "None":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_descriptor_table(path="descriptors.xlsx"):
    """Read the table written by descriptor_steps.
    Returns (names, classes, single, hists, hist_columns):
      names, classes: lists of strings, one per mesh
      single: (n_meshes, 6) array of single-value features, NaN where missing
      hists:  (n_meshes, n_bins_total) array of histogram bins
      hist_columns: the histogram column names, e.g. ['A3_0', ..., 'D4_9']"""
    ws = load_workbook(path, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    col = {name: i for i, name in enumerate(header)}
    hist_columns = [h for h in header if isinstance(h, str) and h.startswith(HISTOGRAM_PREFIXES)]

    names, classes, single, hists = [], [], [], []
    for row in rows:
        if row[col["name"]] is None:
            continue
        values_single = [_to_float(row[col[f]]) for f in SINGLE_VALUE_FEATURES]
        values_hist = [_to_float(row[col[h]]) for h in hist_columns]
        if np.all(np.isnan(values_hist)):
            continue  # failed mesh (load failed / error): no usable features
        names.append(row[col["name"]])
        classes.append(row[col["folder"]])
        single.append(values_single)
        hists.append(values_hist)
    return names, classes, np.array(single), np.array(hists), hist_columns


# --- Database statistics ----------------------------------------------------------------

def compute_stats(single):
    """Per single-value feature: mean, std, min, max and median over the database
    (ignoring missing values). Returns a dict {feature: {stat: value}}."""
    stats = {}
    for j, name in enumerate(SINGLE_VALUE_FEATURES):
        column = single[:, j]
        stats[name] = {
            "mean": float(np.nanmean(column)),
            "std": float(np.nanstd(column)),
            "min": float(np.nanmin(column)),
            "max": float(np.nanmax(column)),
            "median": float(np.nanmedian(column)),
            "missing": int(np.isnan(column).sum()),
        }
    return stats


def save_stats(stats, path="feature_stats.json"):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path="feature_stats.json"):
    with open(path) as f:
        return json.load(f)


# --- Normalization -------------------------------------------------------------------------

def standardize(single, stats):
    """z = (x - mean) / std per feature. Missing values become 0 (= the database mean),
    so they add nothing to a distance. Works on one vector (6,) or a matrix (n, 6)."""
    single = np.asarray(single, dtype=float)
    mean = np.array([stats[f]["mean"] for f in SINGLE_VALUE_FEATURES])
    std = np.array([stats[f]["std"] for f in SINGLE_VALUE_FEATURES])
    z = (single - mean) / np.maximum(std, EPS)
    return np.nan_to_num(z, nan=0.0)


def min_max(single, stats):
    """(x - min) / (max - min) per feature, so the database spans [0, 1]. A query outside
    the database range can fall outside [0, 1]. Missing values become 0.5."""
    single = np.asarray(single, dtype=float)
    lo = np.array([stats[f]["min"] for f in SINGLE_VALUE_FEATURES])
    hi = np.array([stats[f]["max"] for f in SINGLE_VALUE_FEATURES])
    scaled = (single - lo) / np.maximum(hi - lo, EPS)
    return np.nan_to_num(scaled, nan=0.5)


def normalize_feature_vector(single, hists, stats, method="zscore"):
    """The full normalized feature vector of one shape (database or query):
    the 6 normalized single values followed by the histograms (unchanged)."""
    scaled = standardize(single, stats) if method == "zscore" else min_max(single, stats)
    return np.concatenate([scaled, np.asarray(hists, dtype=float)])


# --- Storing and displaying the normalized database --------------------------------------

def _write_table(ws, header, names, classes, matrix, digits=4):
    ws.append(["name", "class"] + header)
    for name, cls, row in zip(names, classes, matrix):
        ws.append([name, cls] + [round(float(v), digits) for v in row])


def write_normalized_database(names, classes, single, hists, hist_columns, stats,
                              path="normalized_features.xlsx"):
    """Excel file with one sheet per version, so the normalization can be inspected:
      stats    mean, std, min, max, median, missing count per feature
      zscore   standardized single values + histograms (the one we use)
      minmax   min-max single values + histograms (for comparison)
      raw      the original values (-1 where missing)"""
    wb = Workbook()
    ws = wb.active
    ws.title = "stats"
    ws.append(["feature", "mean", "std", "min", "max", "median", "missing"])
    for f in SINGLE_VALUE_FEATURES:
        s = stats[f]
        ws.append([f, s["mean"], s["std"], s["min"], s["max"], s["median"], s["missing"]])

    header = SINGLE_VALUE_FEATURES + hist_columns
    _write_table(wb.create_sheet("zscore"), header, names, classes,
                 np.hstack([standardize(single, stats), hists]))
    _write_table(wb.create_sheet("minmax"), header, names, classes,
                 np.hstack([min_max(single, stats), hists]))
    _write_table(wb.create_sheet("raw"), header, names, classes,
                 np.hstack([np.nan_to_num(single, nan=-1), hists]))
    wb.save(path)


def normalize_database_features(descriptor_path="descriptors.xlsx",
                                output_path="normalized_features.xlsx",
                                stats_path="feature_stats.json"):
    """Full step: read the descriptors, compute and save the statistics, write the
    normalized database. Returns (names, classes, zscore_matrix) for direct use."""
    names, classes, single, hists, hist_columns = load_descriptor_table(descriptor_path)
    stats = compute_stats(single)
    save_stats(stats, stats_path)
    write_normalized_database(names, classes, single, hists, hist_columns, stats, output_path)
    features = np.hstack([standardize(single, stats), hists])
    return names, classes, features