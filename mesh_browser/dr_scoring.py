"""Distances and scores for the DR part.

kNN class purity: share of each shape's K nearest neighbours with the same
class. Mean Average Precision comes from map_eval. Also the per-group
evaluation table and the single-shape explanation.
"""

import os

import numpy as np
import pandas as pd

from . import config
from .dr_features import (GROUPS, GROUPS_11, N_SCALARS, apply_feature_transform,
                          default_shares, fit_feature_transform, group_scales,
                          transform_features)
from .dr_paths import Variant, class_purity_path, tsne_path
from .map_eval import mean_average_precision

K_EVAL = 10  # neighbours used by the kNN class purity


# ------------------------------------------------------------- distances

def squared_distances(A):
    """Matrix of squared Euclidean distances between all rows of A."""
    sq = (A ** 2).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2 * A @ A.T
    return np.maximum(D, 0).astype(np.float32)


def group_distance_matrices(X, hellinger=False):
    """Squared distance matrix per group, divided by the group's scale^2.

    Combining them as sum(share * matrix) equals the squared distance of the
    block-scaled vector (up to one constant), so shares can be tried quickly.
    """
    Xt = transform_features(X, Variant(hellinger=hellinger))
    scales = group_scales(Xt)
    return {name: squared_distances(Xt[:, sl]) / scales[name] ** 2
            for name, sl in GROUPS.items()}


def combine(mats, shares):
    """Weighted sum of the per-group squared distance matrices."""
    return sum(shares[name] * mats[name] for name in GROUPS)


# ----------------------------------------------------------------- purity

def per_shape_purity(D, labels, k=K_EVAL):
    """For every shape: share of its k nearest neighbours with the same class.

    D is an (n, n) distance matrix, labels an integer array of length n.
    """
    D = D.copy()
    np.fill_diagonal(D, np.inf)  # a shape is not its own neighbour
    idx = np.argpartition(D, kth=range(k), axis=1)[:, :k]
    return (labels[idx] == labels[:, None]).mean(axis=1)


def knn_purity(D, labels, k=K_EVAL):
    """Mean kNN class purity over all shapes."""
    return float(per_shape_purity(D, labels, k).mean())


def chance_purity(labels):
    """Expected purity if neighbours were random (the baseline to beat)."""
    p = np.bincount(labels) / len(labels)
    return float((p ** 2).sum())


def class_table(labels, names, **per_shape):
    """Per-class mean of per-shape values, sorted by the first column."""
    sizes = np.bincount(labels)
    data = {"class": np.asarray(names), "size": sizes}
    for key, vals in per_shape.items():
        data[key] = np.bincount(labels, weights=vals) / sizes
    return pd.DataFrame(data).sort_values(list(per_shape)[0], ascending=False)


# ------------------------------------------------------------ evaluation

def evaluate_groups(X, labels, names, shares):
    """Print kNN purity per group, the 11-group baseline, and 6-group variants."""
    print(f"kNN purity (K={K_EVAL}), {len(X)} shapes, "
          f"{labels.max() + 1} classes; chance level {chance_purity(labels):.3f}")
    variants = {
        "raw": group_distance_matrices(X),
        "Hellinger": group_distance_matrices(X, hellinger=True),
    }
    print(f"  {'group':8s}" + "".join(f"{v:>15s}" for v in variants))
    for name in GROUPS:
        row = "".join(f"{knn_purity(m[name], labels):15.3f}"
                      for m in variants.values())
        print(f"  {name:8s}{row}")

    # Baseline comparison: 11 equal groups (6 individual scalars + 5 raw histograms)
    spec_11 = fit_feature_transform(
        X, variant=Variant(scale_blocks=True, hellinger=False),
        shares=default_shares(GROUPS_11), groups=GROUPS_11
    )
    D_11 = squared_distances(apply_feature_transform(X, spec_11))
    print(f"\n  11-group equal (raw, knn.py baseline): "
          f"purity {knn_purity(D_11, labels):.3f}, "
          f"mAP {mean_average_precision(D_11, labels):.4f}")

    equal = {name: 1 / len(GROUPS) for name in GROUPS}
    for label, mats in variants.items():
        D_eq = combine(mats, equal)
        D_cur = combine(mats, shares)
        print(f"  6-group {label:9s}: "
              f"equal shares purity {knn_purity(D_eq, labels):.3f} "
              f"(mAP {mean_average_precision(D_eq, labels):.4f}), "
              f"current shares purity {knn_purity(D_cur, labels):.3f} "
              f"(mAP {mean_average_precision(D_cur, labels):.4f})")

    # per class, for the Hellinger vector with the current shares
    ps = per_shape_purity(combine(variants["Hellinger"], shares), labels)
    table = class_table(labels, names, purity=ps)
    os.makedirs(config.SHEETS_DIR, exist_ok=True)
    out = class_purity_path()
    table.to_csv(out, index=False)
    print("\nbest 10 classes:")
    print(table.head(10).to_string(index=False))
    print("worst 10 classes:")
    print(table.tail(10).to_string(index=False))
    print("saved", out)


def evaluate_map(X, labels, variant, shares, perplexity):
    """mAP of the saved 2D map if it exists, else of the feature space."""
    csv_path = tsne_path(perplexity, variant)
    if os.path.exists(csv_path):
        Y = pd.read_csv(csv_path)[["x", "y"]].to_numpy()
        value = mean_average_precision(squared_distances(Y), labels)
        return (f"mAP in 2D t-SNE space (perplexity {perplexity}, "
                f"variant '{variant.name}'): {value:.4f}")
    Xt = transform_features(X, variant, shares)
    value = mean_average_precision(squared_distances(Xt), labels)
    return f"mAP in feature space (variant '{variant.name}'): {value:.4f}"


def explain_shape(X, info, name, variant, shares, k=K_EVAL):
    """Print a shape's nearest neighbours and what drives each distance.

    For every neighbour the share of the total (scaled, squared) distance
    that comes from each group is shown, plus the rank of the first
    same-class shapes. This shows which group pulls a shape away from its class.
    """
    names = np.array(info["name"].astype(str).tolist(), dtype=str)
    classes = np.array(info["class"].astype(str).tolist(), dtype=str)
    hits = np.flatnonzero(names == name)
    if len(hits) == 0:
        hits = np.flatnonzero(np.char.find(names, name) >= 0)  # substring
    if len(hits) == 0:
        raise ValueError(f"No shape named like '{name}' in the sheet.")
    i = int(hits[0])

    mats = group_distance_matrices(X, hellinger=variant.hellinger)
    D = combine(mats, shares)
    d = D[i].copy()
    d[i] = np.inf
    order = np.argsort(d)
    groups = list(GROUPS)

    def parts(j):
        return [100 * shares[g] * mats[g][i, j] / D[i, j] for g in groups]

    print(f"\nshape {names[i]}  class {classes[i]}")
    print("scalars (z-scores): " + "  ".join(
        f"{v:6.2f}" for v in X[i, :N_SCALARS]))
    print(f"\n{'rank':>4s} {'name':14s} {'class':22s} {'dist':>6s}  "
          + " ".join(f"{g:>7s}" for g in groups) + "   (% of distance)")
    for rank, j in enumerate(order[:k], start=1):
        mark = "*" if classes[j] == classes[i] else " "
        print(f"{rank:4d}{mark}{names[j]:14s} {classes[j]:22s} {d[j]:6.2f}  "
              + " ".join(f"{p:7.1f}" for p in parts(j)))
    same = np.flatnonzero(classes[order] == classes[i])[:5]
    print("\n(* = same class)  ranks of the first same-class shapes:",
          [int(r) + 1 for r in same])
    for r in same[:3]:
        j = order[r]
        print(f"  rank {r + 1:4d} {names[j]:14s} dist {d[j]:6.2f}  "
              + " ".join(f"{p:7.1f}" for p in parts(j)))