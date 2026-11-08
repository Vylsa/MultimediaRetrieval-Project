"""Feature table and feature-vector transforms for the t-SNE part.

The standardized sheet has name, class, 6 z-scored scalars and 5 histograms
of 50 bins (256 feature columns). This module loads it, defines the column
groups and the group shares, and turns the raw table into the vector that
t-SNE and the scores work on (Hellinger and/or block scaling).
"""

import json
import os

import numpy as np
import pandas as pd

from . import config
from .dr_paths import best_shares_path

STANDARDIZED_FILE = os.path.join(config.SHEETS_DIR, "normalized_features.xlsx")
STANDARDIZED_SHEET = "zscore"  # sheet inside the workbook with the z-scores

# Classes that are not real shape classes; removed BEFORE t-SNE.
# TODO: add the second test class.
TEST_CLASSES = config.TEST_CLASSES

N_SCALARS = 6
HISTOGRAMS = ["A3", "D1", "D2", "D3", "D4"]
N_BINS = 50
EXPECTED_FEATURES = N_SCALARS + len(HISTOGRAMS) * N_BINS  # 256

# Column groups inside the 256-value vector: scalars first, then histograms.
GROUPS = {"scalars": slice(0, N_SCALARS)}
GROUPS.update({
    name: slice(N_SCALARS + i * N_BINS, N_SCALARS + (i + 1) * N_BINS)
    for i, name in enumerate(HISTOGRAMS)
})

SCALAR_SHARE = 1 / 3  # default share of the scalars; histograms split the rest
N_PAIRS = 5000        # random pairs used to estimate group scales


# ---------------------------------------------------------------- shares

def default_shares():
    """Share of the total squared distance per group (adds up to 1)."""
    rest = (1 - SCALAR_SHARE) / (len(GROUPS) - 1)
    return {name: SCALAR_SHARE if name == "scalars" else rest
            for name in GROUPS}


def get_shares(variant):
    """The shares for a variant: saved --optimize result or the defaults."""
    if not variant.best_shares:
        return default_shares()
    with open(best_shares_path(variant.hellinger)) as f:
        saved = json.load(f)
    total = sum(saved[name] for name in GROUPS)
    return {name: saved[name] / total for name in GROUPS}


def share_weights(shares):
    """Column multiplier per group that gives it its share.

    After scaling, every group has squared distance ~1, so the multiplier is
    sqrt(share). A common constant factor does not matter for t-SNE, so equal
    shares give 1.0.
    """
    return {name: (len(GROUPS) * shares[name]) ** 0.5 for name in GROUPS}


# ---------------------------------------------------------------- loading

def load_feature_table(path=STANDARDIZED_FILE, drop_classes=TEST_CLASSES):
    """Load the standardized sheet.

    Returns (info, X): info has the columns name and class, X is a float
    matrix (rows = shapes, columns = the 256 features).
    """
    print("classes to be dropped before t-SNE:", drop_classes)
    df = pd.read_excel(path, sheet_name=STANDARDIZED_SHEET)
    feature_cols = [c for c in df.columns if c not in ("name", "class")]

    # Decimal commas stored as text -> numbers.
    for col in feature_cols:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

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

def group_scales(X, n_pairs=N_PAIRS, seed=0):
    """Mean Euclidean distance between random pairs, per column group."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return {
        name: np.linalg.norm(X[i][:, sl] - X[j][:, sl], axis=1).mean()
        for name, sl in GROUPS.items()
    }


def group_report(X, label=""):
    """Print each group's share of the squared distance of a random pair."""
    scales = group_scales(X)
    sq = {name: s ** 2 for name, s in scales.items()}  # approx. mean sq. dist
    total = sum(sq.values())
    print(f"share of squared distance per group {label}:")
    for name in GROUPS:
        print(f"  {name:8s} {100 * sq[name] / total:6.2f} %")


def transform_features(X, variant, shares=None):
    """The vector t-SNE and the scores work on, for a given variant.

    Hellinger: sqrt of every histogram bin (Euclidean distance between the
    results is the Hellinger distance between the histograms).
    Scaled: every group divided by its typical pair distance and multiplied by
    its weight, so the groups contribute according to `shares`.
    """
    X = X.copy()
    if variant.hellinger:
        for name, sl in GROUPS.items():
            if name != "scalars":
                X[:, sl] = np.sqrt(np.maximum(X[:, sl], 0))
    if variant.scaled:
        weights = share_weights(shares or get_shares(variant))
        scales = group_scales(X)
        for name, sl in GROUPS.items():
            X[:, sl] *= weights[name] / scales[name]
    return X