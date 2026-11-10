"""Running t-SNE and tuning it: group shares, seeds, perplexities.

t-SNE is stochastic, so `sweep` runs several seeds per perplexity, scores
every map and keeps the best one. Scores:
    purity_2d        kNN class purity measured in the 2D map (uses the labels)
    trustworthiness  share of 2D neighbours that are also input-space
                     neighbours (no labels; 1.0 = perfect)
    map              Mean Average Precision of the 2D distances
    kl               t-SNE's own objective (only comparable at equal perplexity)
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness

from . import config
from .dr_features import (GROUPS, STANDARDIZED_FILE, group_report,
                          load_feature_table, transform_features)
from .dr_paths import (best_shares_path, class_purity_path, sweep_path,
                       tsne_path)
from .dr_scoring import (K_EVAL, chance_purity, class_table, combine,
                         knn_purity, per_shape_purity, squared_distances)
from .map_eval import mean_average_precision

RANDOM_STATE = 0
METRICS = {"purity": "purity_2d", "trust": "trustworthiness", "map": "map"}


# ----------------------------------------------------------------- t-SNE

def run_tsne(X, perplexity, random_state=RANDOM_STATE, return_kl=False):
    """2D t-SNE embedding of X (Euclidean distance, PCA initialisation)."""
    if perplexity >= len(X):
        raise ValueError(f"Perplexity {perplexity} must be below {len(X)} shapes.")
    tsne = TSNE(
        n_components=2,
        metric="euclidean",
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_state,
    )
    Y = tsne.fit_transform(X)
    return (Y, float(tsne.kl_divergence_)) if return_kl else Y


def write_map_csv(info, Y, perplexity, variant, seed):
    """Write one 2D map (name, class, x, y, ...) and return the file path."""
    path = tsne_path(perplexity, variant)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old_seed = old["seed"].iloc[0] if "seed" in old else "unknown"
        print(f"  replacing the existing map (seed {old_seed})")
    out = info.copy()
    out["x"] = Y[:, 0]
    out["y"] = Y[:, 1]
    out["perplexity"] = perplexity
    out["variant"] = variant.name
    out["seed"] = seed
    out["source"] = os.path.basename(STANDARDIZED_FILE)
    os.makedirs(config.SHEETS_DIR, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def sweep(perplexities, n_seeds, variant, shares, metric="purity_2d"):
    """Run t-SNE for every perplexity and seed, keep the best map per perplexity.

    The best map of each perplexity is saved (it replaces an earlier map of
    the same perplexity and variant). Returns the best perplexity overall.
    """
    info, X_raw = load_feature_table()
    group_report(X_raw, "BEFORE transform (raw)")
    X = transform_features(X_raw, variant, shares)
    group_report(X, "AFTER transform (this is what t-SNE sees)")

    labels, names = pd.factorize(info["class"])
    ps_input = per_shape_purity(squared_distances(X), labels)
    print(f"\n{len(X)} shapes, {X.shape[1]} features, variant '{variant.name}'")
    print(f"input-space kNN purity {ps_input.mean():.3f} "
          f"(chance {chance_purity(labels):.3f}); selecting by {metric}")

    rows, best_per_p = [], {}
    for p in perplexities:
        for seed in range(n_seeds):
            Y, kl = run_tsne(X, p, random_state=seed, return_kl=True)
            D2 = squared_distances(Y)
            row = {
                "perplexity": p,
                "seed": seed,
                "purity_2d": knn_purity(D2, labels),
                "trustworthiness": float(trustworthiness(X, Y, n_neighbors=K_EVAL)),
                "map": mean_average_precision(D2, labels),
                "kl": kl,
            }
            rows.append(row)
            print(f"  perplexity {p:3d} seed {seed}: purity {row['purity_2d']:.3f}"
                  f"  trust {row['trustworthiness']:.3f}  mAP {row['map']:.3f}"
                  f"  KL {kl:.3f}")
            if p not in best_per_p or row[metric] > best_per_p[p][0]:
                best_per_p[p] = (row[metric], seed, Y)

    df = pd.DataFrame(rows)
    os.makedirs(config.SHEETS_DIR, exist_ok=True)
    df.to_csv(sweep_path(variant), index=False)
    if n_seeds > 1:
        print("\nmean / std over seeds, per perplexity:")
        columns = ["purity_2d", "trustworthiness", "map", "kl"]
        print(df.groupby("perplexity")[columns].agg(["mean", "std"])
              .round(3).to_string())

    print()
    for p, (score, seed, Y) in best_per_p.items():
        path = write_map_csv(info, Y, p, variant, seed)
        table = class_table(labels, names, purity_input=ps_input,
                            purity_2d=per_shape_purity(squared_distances(Y), labels))
        table.to_csv(class_purity_path(p, variant), index=False)
        print(f"best map for perplexity {p}: seed {seed}, {metric} {score:.3f}"
              f" -> {path}")
    best_p = max(best_per_p, key=lambda p: best_per_p[p][0])
    print(f"overall best: perplexity {best_p}, seed {best_per_p[best_p][1]}"
          f" ({metric} {best_per_p[best_p][0]:.3f})")
    return best_p


# ---------------------------------------------------------------- shares

def optimize_shares(mats, labels, n_random=200, n_refine=300, seed=0):
    """Search for group shares that maximize kNN purity.

    Random shares first, then hill climbing from the best one. Returns the
    best shares (summing to 1) and their purity.
    """
    rng = np.random.default_rng(seed)
    names = list(GROUPS)

    def score(vec):
        return knn_purity(combine(mats, dict(zip(names, vec))), labels)

    best_vec = np.full(len(names), 1 / len(names))
    best = score(best_vec)
    print(f"start (equal shares): {best:.3f}")
    for _ in range(n_random):
        vec = rng.dirichlet(np.ones(len(names)))
        sc = score(vec)
        if sc > best:
            best, best_vec = sc, vec
    print(f"after random search: {best:.3f}")
    for _ in range(n_refine):
        vec = best_vec * np.exp(rng.normal(0, 0.3, len(names)))
        vec /= vec.sum()
        sc = score(vec)
        if sc > best:
            best, best_vec = sc, vec
    print(f"after refinement:    {best:.3f}")
    return dict(zip(names, (float(v) for v in best_vec))), best


def save_best_shares(shares, hellinger=False):
    """Write the shares found by optimize_shares; returns the file path."""
    path = best_shares_path(hellinger)
    os.makedirs(config.SHEETS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(shares, f, indent=2)
    return path