"""Feature table and feature-vector transforms for the t-SNE and KNN parts.

The standardized sheet has name, class, 6 z-scored scalars and 5 histograms
of 50 bins (256 feature columns). This module loads it, defines the column
groups and the group shares, and provides a shared fit/apply transform so
t-SNE, k-d tree queries, and evaluation use the exact same feature space.
"""

import json
import os

import numpy as np
import pandas as pd

from . import config
from .dr_paths import Variant, best_shares_path, feature_transform_path

STANDARDIZED_FILE = os.path.join(config.SHEETS_DIR, "normalized_features.xlsx")
STANDARDIZED_SHEET = "zscore"  # sheet inside the workbook with the z-scores

TEST_CLASSES = config.TEST_CLASSES

SCALAR_NAMES = ["area", "compactness", "rectangularity",
                "convexity", "diameter", "eccentricity"]
N_SCALARS = len(SCALAR_NAMES)
HISTOGRAMS = ["A3", "D1", "D2", "D3", "D4"]
N_BINS = 50
EXPECTED_FEATURES = N_SCALARS + len(HISTOGRAMS) * N_BINS  # 256

# 6-group partition (scalars together + 5 histograms) used by DR & querying.
GROUPS = {"scalars": slice(0, N_SCALARS)}
GROUPS.update({
    name: slice(N_SCALARS + i * N_BINS, N_SCALARS + (i + 1) * N_BINS)
    for i, name in enumerate(HISTOGRAMS)
})

# 11-group partition (each scalar individually + 5 histograms) for baseline comparison.
GROUPS_11 = {name: slice(i, i + 1) for i, name in enumerate(SCALAR_NAMES)}
GROUPS_11.update({
    name: slice(N_SCALARS + i * N_BINS, N_SCALARS + (i + 1) * N_BINS)
    for i, name in enumerate(HISTOGRAMS)
})

SCALAR_SHARE = 1 / 3  # default share of the scalars; histograms split the rest
N_PAIRS = 5000        # random pairs used to estimate group scales


# ---------------------------------------------------------------- shares

def default_shares(groups=GROUPS):
    """Share of the total squared distance per group (adds up to 1)."""
    if groups is not GROUPS or "scalars" not in groups:
        return {name: 1.0 / len(groups) for name in groups}
    rest = (1 - SCALAR_SHARE) / (len(groups) - 1)
    return {name: SCALAR_SHARE if name == "scalars" else rest
            for name in groups}


def get_shares(variant):
    """The shares for a variant: saved --optimize result or the defaults."""
    if not variant.best_shares:
        return default_shares()
    with open(best_shares_path(variant.hellinger)) as f:
        saved = json.load(f)
    total = sum(saved[name] for name in GROUPS)
    return {name: saved[name] / total for name in GROUPS}


def share_weights(shares, groups=GROUPS):
    """Column multiplier per group that gives it its share.

    After scaling, every group has squared distance ~1, so the multiplier is
    sqrt(share). A common constant factor does not matter for t-SNE or kNN, so
    equal shares give 1.0.
    """
    return {name: (len(groups) * shares[name]) ** 0.5 for name in groups}


# ---------------------------------------------------------------- loading

def load_feature_table(path=STANDARDIZED_FILE, sheet=STANDARDIZED_SHEET,
                       drop_classes=TEST_CLASSES):
    """Load the standardized sheet.

    Returns (info, X): info has the columns name and class, X is a float
    matrix (rows = shapes, columns = the 256 features).
    """
    df = pd.read_excel(path, sheet_name=sheet)
    feature_cols = [c for c in df.columns if c not in ("name", "class")]

    # Decimal commas stored as text -> numbers.
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    if drop_classes:
        df = df[~df["class"].isin(drop_classes)].reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype=float)

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} feature columns, found {X.shape[1]}."
        )
    if np.isnan(X).any():
        bad = np.unique(np.argwhere(np.isnan(X))[:, 1])
        raise ValueError(
            "NaN values in columns: " + ", ".join(feature_cols[i] for i in bad)
        )
    return df[["name", "class"]], X


# ------------------------------------------------------------- transforms

def group_scales(X, n_pairs=N_PAIRS, seed=0, groups=GROUPS):
    """Mean Euclidean distance between random pairs, per column group."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return {
        name: float(np.linalg.norm(X[i][:, sl] - X[j][:, sl], axis=1).mean())
        for name, sl in groups.items()
    }


def group_report(X, label=""):
    """Print each group's share of the squared distance of a random pair."""
    scales = group_scales(X)
    sq = {name: s ** 2 for name, s in scales.items()}  # approx. mean sq. dist
    total = sum(sq.values())
    print(f"share of squared distance per group {label}:")
    for name in GROUPS:
        print(f"  {name:8s} {100 * sq[name] / total:6.2f} %")


def fit_feature_transform(X, variant=Variant(), shares=None,
                          groups=GROUPS, n_pairs=N_PAIRS, seed=0):
    """Fit the feature transform on the database matrix X.

    Returns a JSON-serializable dict with:
      - hellinger: bool (whether to take sqrt of histogram bins first)
      - multipliers: list of length 256 (weight / scale per column)
      - scales: per-group mean pair distances on the database
      - shares: per-group shares used
      - variant: variant name tag
    """
    X_work = np.asarray(X, dtype=float).copy()
    if X_work.ndim != 2 or X_work.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected shape (N, {EXPECTED_FEATURES}), got {X_work.shape}."
        )

    if variant.hellinger:
        X_work[:, N_SCALARS:] = np.sqrt(np.maximum(X_work[:, N_SCALARS:], 0.0))

    multipliers = np.ones(EXPECTED_FEATURES, dtype=float)
    scales = {}
    used_shares = {}

    if variant.scaled:
        used_shares = shares if shares is not None else (
            get_shares(variant) if groups is GROUPS else default_shares(groups)
        )
        weights = share_weights(used_shares, groups=groups)
        scales = group_scales(X_work, n_pairs=n_pairs, seed=seed, groups=groups)
        for name, sl in groups.items():
            multipliers[sl] = weights[name] / max(scales[name], 1e-12)

    return {
        "variant": variant.name,
        "hellinger": bool(variant.hellinger),
        "scaled": bool(variant.scaled),
        "n_scalars": N_SCALARS,
        "n_features": EXPECTED_FEATURES,
        "shares": {k: float(v) for k, v in used_shares.items()},
        "scales": {k: float(v) for k, v in scales.items()},
        "multipliers": multipliers.tolist(),
    }


def save_feature_transform(spec, path=None, variant=None):
    """Save a fitted transform spec to JSON and return the path."""
    if path is None:
        v = variant or Variant()
        path = feature_transform_path(v)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    return path


def load_feature_transform(path=None, variant=None):
    """Load a fitted transform spec from JSON."""
    if path is None:
        v = variant or Variant(hellinger=True, best_shares=True)
        path = feature_transform_path(v)
    with open(path) as f:
        return json.load(f)


def apply_feature_transform(X, spec):
    """Apply a fitted transform spec to a single vector (D,) or matrix (N, D)."""
    arr = np.asarray(X, dtype=float)
    is_1d = arr.ndim == 1
    out = arr[None, :].copy() if is_1d else arr.copy()

    n_scalars = int(spec.get("n_scalars", N_SCALARS))
    if spec.get("hellinger", False):
        out[:, n_scalars:] = np.sqrt(np.maximum(out[:, n_scalars:], 0.0))

    multipliers = np.asarray(spec["multipliers"], dtype=float)
    out *= multipliers
    return out[0] if is_1d else out


def transform_features(X, variant, shares=None, save_path=None):
    """Fit and apply the transform on X (optionally saving the fitted spec)."""
    spec = fit_feature_transform(X, variant=variant, shares=shares)
    if save_path is not None:
        save_feature_transform(spec, path=save_path, variant=variant)
    return apply_feature_transform(X, spec)