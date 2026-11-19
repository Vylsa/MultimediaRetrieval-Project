"""Step 9: evaluate the shape search engine (query-by-example distance).

Every database shape is used as a query against all OTHER database shapes, using
the exact same feature transform and distance metric as t-SNE and ShapeIndex.

Metrics per query (the query itself is never in its own ranking):
  AUROC        probability that a random same-class shape is ranked closer
               than a random different-class shape (0.5 = chance, 1 = perfect).
               Uses the whole ranking, so it does not depend on K.
  precision@K  fraction of the K nearest shapes that share the query's class
               (equals kNN purity when K = 10).

Aggregation: per class (mean over its shapes), micro (every shape counts once)
and macro (every class counts once).

Run:  python -m mesh_browser.evaluate [--k 10] [--no-plot]
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config
from .dr_features import (GROUPS_11, default_shares, fit_feature_transform,
                          load_feature_table)
from .dr_paths import Variant
from .knn import ShapeIndex, SINGLE_VALUE_FEATURES, HISTOGRAM_NAMES
from .query import compute_distances
from .search import SearchIndex

FEATURE_FILE = config.STANDARDIZED_FILE
SHEET = config.STANDARDIZED_SHEET
STATS_FILE = config.STATS_FILE

FPR_GRID = np.linspace(0, 1, 201)


# ----------------------------------------------------------------- distances
def full_distance_matrix(index):
    """N x N combined distances, one legacy EMD engine query per database shape."""
    n = len(index.names)
    dist = np.empty((n, n))
    for i in range(n):
        dist[i], _ = compute_distances(index.features[i], index.features,
                                       index.scales, index.weights,
                                       index.hist_distance)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    return dist


def bruteforce_distance_matrix(tree):
    """N x N Euclidean distances on the shared transformed feature matrix."""
    sq = (tree.features ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * tree.features @ tree.features.T, 0.0)
    return np.sqrt(d2)


def kdtree_distance_matrix(names, classes, raw_features, transform_spec):
    """N x N distances from the k-d tree engine (built with the shared transform)."""
    n = len(names)
    n_bins = (raw_features.shape[1] - len(SINGLE_VALUE_FEATURES)) // len(HISTOGRAM_NAMES)
    feature_names = SINGLE_VALUE_FEATURES + [f"{h}_{i}" for h in HISTOGRAM_NAMES
                                             for i in range(n_bins)]
    tree = ShapeIndex(names, classes, raw_features, feature_names,
                      weighted=config.WEIGHTED, transform_spec=transform_spec)

    position = {(c, nm): j for j, (nm, c) in enumerate(zip(tree.names, tree.classes))}
    if len(position) != n:
        raise ValueError("Shape names are not unique within their class; "
                         "cannot map k-d tree results back to rows.")

    dist = np.full((n, n), np.inf)
    for i in range(n):
        query = tree.prepare_query(raw_features[i])
        for name, cls, d in tree.knn(query, n):         # (name, class, distance)
            dist[i, position[(cls, name)]] = d
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    missing = np.isinf(dist).sum()
    if missing:
        print(f"WARNING: {missing} (query, shape) pairs were not returned by the "
              "k-d tree; AUROC is only valid for a complete ranking.")
    return dist, tree


# ----------------------------------------------------------------- metrics
def query_metrics(d, same, k):
    """AUROC, precision@K and top-K indices for one query.
    d: distances to all other shapes, same: True where the class matches."""
    n_pos = int(same.sum())
    n_neg = len(same) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.array([], dtype=int)
    ranks = rankdata(-d)                            # closest = highest rank
    auroc = (ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    top = np.argsort(d, kind="stable")[:k]
    return auroc, same[top].mean(), top


def evaluate(names, labels, dist, k):
    """Per-shape table plus, for every class, the wrong classes in its top K."""
    n = len(names)
    rows = []
    confusion = {c: Counter() for c in np.unique(labels)}
    for i in range(n):
        idx = np.delete(np.arange(n), i)            # leave the query out
        same = labels[idx] == labels[i]
        auroc, prec, top = query_metrics(dist[i, idx], same, k)
        rows.append({"name": names[i], "class": labels[i], "auroc": auroc,
                     "precision_at_k": prec, "chance_precision": same.mean()})
        for j in top:
            if not same[j]:
                confusion[labels[i]][labels[idx][j]] += 1
    return pd.DataFrame(rows), confusion


def class_table(per_shape, confusion):
    """Aggregate per class, with the most frequent wrong class in the top K."""
    table = per_shape.groupby("class").agg(
        n=("name", "size"),
        auroc=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        precision_at_k=("precision_at_k", "mean"),
        chance_precision=("chance_precision", "mean"))
    table["top_confusion"] = [
        confusion[c].most_common(1)[0][0] if confusion[c] else ""
        for c in table.index]
    return table.sort_values("auroc", ascending=False).reset_index()


def overall(per_shape, table, k):
    """Grand aggregates: micro (per shape) and macro (per class)."""
    return pd.DataFrame([
        {"aggregate": "micro (every shape counts once)",
         "auroc": per_shape["auroc"].mean(),
         f"precision@{k}": per_shape["precision_at_k"].mean()},
        {"aggregate": "macro (every class counts once)",
         "auroc": table["auroc"].mean(),
         f"precision@{k}": table["precision_at_k"].mean()},
    ])


# ------------------------------------------------------------- ROC curves
def roc_curves(labels, dist):
    """Mean ROC curve per class."""
    n = len(labels)
    sums = {c: np.zeros(len(FPR_GRID)) for c in np.unique(labels)}
    counts = Counter()
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        same = labels[idx] == labels[i]
        n_pos = int(same.sum())
        n_neg = len(same) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        hit = same[np.argsort(dist[i, idx], kind="stable")]
        tpr = np.concatenate([[0], np.cumsum(hit) / n_pos])
        fpr = np.concatenate([[0], np.cumsum(~hit) / n_neg])
        sums[labels[i]] += np.interp(FPR_GRID, fpr, tpr)
        counts[labels[i]] += 1
    return {c: sums[c] / counts[c] for c in sums if counts[c]}


def curve_area(tpr):
    """Trapezoid area under a curve sampled on FPR_GRID."""
    return float(np.sum(np.diff(FPR_GRID) * (tpr[1:] + tpr[:-1]) / 2))


def save_roc_plot(curves, path, aucs, macro_auc, micro_auc):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    macro = np.mean(list(curves.values()), axis=0)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "ROC curves", "Same, log FPR axis (top of the ranking)"))
    fig.update_layout(title=f"AUROC: macro {macro_auc:.3f}, micro {micro_auc:.3f} "
                            "(shaded: area under the macro-average curve)")
    for col in (1, 2):
        for c, tpr in curves.items():
            fig.add_trace(go.Scatter(x=FPR_GRID, y=tpr, mode="lines",
                                     name=f"{c} (AUROC {aucs[c]:.2f})",
                                     legendgroup=c, showlegend=(col == 1),
                                     opacity=0.35, line=dict(width=1)),
                          row=1, col=col)
        fig.add_trace(go.Scatter(x=FPR_GRID, y=macro, mode="lines",
                                 name=f"macro average (AUROC {macro_auc:.3f})",
                                 legendgroup="macro", showlegend=(col == 1),
                                 fill="tozeroy", fillcolor="rgba(0,0,0,0.12)",
                                 line=dict(width=4, color="black")),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                 legendgroup="chance", showlegend=(col == 1),
                                 line=dict(width=2, dash="dash", color="grey")),
                      row=1, col=col)
    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_xaxes(title_text="False positive rate (log)", type="log",
                     range=[np.log10(0.005), 0], row=1, col=2)
    fig.update_yaxes(title_text="True positive rate (recall)")
    fig.write_html(path)


# ------------------------------------------------------------------ output
def save_plot(table, k, path):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_bar(x=table["class"], y=table["auroc"], name="AUROC")
    fig.add_bar(x=table["class"], y=table["precision_at_k"],
                name=f"precision@{k}")
    fig.add_scatter(x=table["class"], y=table["chance_precision"],
                    mode="markers", name=f"chance precision@{k}")
    fig.update_layout(barmode="group", xaxis_tickangle=-60,
                      title="Retrieval quality per class (sorted by AUROC)")
    fig.write_html(path)


def print_extremes(table, count=10):
    cols = ["class", "n", "auroc", "precision_at_k", "chance_precision",
            "top_confusion"]
    with pd.option_context("display.width", 140, "display.float_format",
                           "{:.3f}".format):
        print(f"\nBest {count} classes (by AUROC):")
        print(table[cols].head(count).to_string(index=False))
        print(f"\nWorst {count} classes (by AUROC):")
        print(table[cols].tail(count).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--k", type=int, default=10, help="K for precision@K")
    parser.add_argument("--features", default=FEATURE_FILE)
    parser.add_argument("--sheet", default=SHEET)
    parser.add_argument("--stats", default=STATS_FILE)
    parser.add_argument("--hist-distance", default="emd",
                        choices=["emd", "l1", "euclidean"])
    parser.add_argument("--engine", default="kdtree",
                        choices=["kdtree", "bruteforce", "emd"],
                        help="kdtree: k-d tree on shared transform; "
                             "bruteforce: linear scan on shared transform; "
                             "emd: legacy EMD comparison")
    parser.add_argument("--baseline-11", action="store_true",
                        help="use 11 equal groups (6 scalars + 5 raw histograms)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("Loading feature table (excluding TEST_CLASSES)...")
    info, raw_features = load_feature_table(args.features, sheet=args.sheet)
    names = info["name"].astype(str).to_numpy()
    labels = info["class"].astype(str).to_numpy()
    print(f"{len(names)} shapes, {len(np.unique(labels))} classes")

    if args.engine == "emd":
        print("Loading legacy EMD search index...")
        index = SearchIndex(args.features, args.sheet, args.stats,
                            exclude_classes=config.TEST_CLASSES,
                            hist_distance=args.hist_distance)
        dist = full_distance_matrix(index)
        tag = "_emd" if args.hist_distance == "emd" else f"_emd_{args.hist_distance}"
    else:
        if args.baseline_11:
            spec = fit_feature_transform(
                raw_features,
                variant=Variant(scale_blocks=True, hellinger=False),
                shares=default_shares(GROUPS_11),
                groups=GROUPS_11,
            )
            tag = "_baseline11"
        else:
            spec = None  # ShapeIndex loads/fits Variant(hellinger=True, best_shares=True)
            tag = "" if args.engine == "kdtree" else f"_{args.engine}"

        print(f"Computing the distance matrix with the {args.engine} engine...")
        dist, tree = kdtree_distance_matrix(names, labels, raw_features, spec)
        if args.engine == "bruteforce":
            dist = bruteforce_distance_matrix(tree)

    print("Evaluating every shape as a query...")
    per_shape, confusion = evaluate(names, labels, dist, args.k)
    table = class_table(per_shape, confusion)
    summary = overall(per_shape, table, args.k)

    out = config.SHEETS_DIR
    per_shape.to_csv(os.path.join(out, f"eval_per_shape{tag}.csv"), index=False)
    table.to_csv(os.path.join(out, f"eval_per_class{tag}.csv"), index=False)
    summary.to_csv(os.path.join(out, f"eval_summary{tag}.csv"), index=False)
    if not args.no_plot:
        save_plot(table, args.k, os.path.join(out, f"eval_per_class{tag}.html"))

    curves = roc_curves(labels, dist)
    macro_area = curve_area(np.mean(list(curves.values()), axis=0))
    if not args.no_plot:
        save_roc_plot(curves, os.path.join(out, f"eval_roc_curves{tag}.html"),
                      dict(zip(table["class"], table["auroc"])),
                      table["auroc"].mean(), per_shape["auroc"].mean())

    print("\nGrand aggregates:")
    print(summary.to_string(index=False, float_format="{:.3f}".format))
    print_extremes(table)
    print(f"\nArea under the macro-average ROC curve: {macro_area:.3f} "
          "(should be close to the macro AUROC above)")
    print(f"\nSaved to {out}: eval_*{tag}.csv" + ("" if args.no_plot else f" and eval_*{tag}.html"))


if __name__ == "__main__":
    main()