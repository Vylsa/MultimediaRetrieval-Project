"""Step 9: evaluate the shape search engine (query-by-example distance).

Every database shape is used as a query against all OTHER database shapes, with
the engine itself (SearchIndex + query.compute_distances), so the numbers describe
the real search. The query's stored (already standardized) feature vector is used,
which is exactly the row the engine would compute for that shape.

Metrics per query (the query itself is never in its own ranking):
  AUROC        probability that a random same-class shape is ranked closer
               than a random different-class shape (0.5 = chance, 1 = perfect).
               Uses the whole ranking, so it does not depend on K.
  precision@K  fraction of the K nearest shapes that share the query's class.

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
from .query import compute_distances
from .search import SearchIndex

# Same files as the GUI (see _work_find in app.py).
FEATURE_FILE = config.STANDARDIZED_FILE
SHEET = config.STANDARDIZED_SHEET
STATS_FILE = config.STATS_FILE

# Common false-positive-rate grid, so ROC curves of different queries can be averaged.
FPR_GRID = np.linspace(0, 1, 201)


# ----------------------------------------------------------------- distances
def full_distance_matrix(index):
    """N x N combined distances, one engine query per database shape."""
    n = len(index.names)
    dist = np.empty((n, n))
    for i in range(n):
        dist[i], _ = compute_distances(index.features[i], index.features,
                                       index.scales, index.weights,
                                       index.hist_distance)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    return dist


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
    """Mean ROC curve per class. One query's curve is obtained by walking down its
    ranking (K = 1, 2, ..., N-1): TPR = share of the same-class shapes found so far,
    FPR = share of the other-class shapes returned so far. Curves of the queries of
    one class are averaged on a common FPR grid."""
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
    """Left: ROC curves (linear axes). Right: same with a log FPR axis, which
    zooms in on the top of the ranking, where a user actually looks."""
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
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("Loading the search index...")
    index = SearchIndex(args.features, args.sheet, args.stats,
                        exclude_classes=config.TEST_CLASSES,
                        hist_distance=args.hist_distance)
    names = np.array(index.names, dtype=str)
    labels = np.array(index.classes, dtype=str)
    print(f"{len(names)} shapes, {len(np.unique(labels))} classes")

    print("Computing the distance matrix with the engine...")
    dist = full_distance_matrix(index)

    print("Evaluating every shape as a query...")
    per_shape, confusion = evaluate(names, labels, dist, args.k)
    table = class_table(per_shape, confusion)
    summary = overall(per_shape, table, args.k)

    out = config.SHEETS_DIR
    per_shape.to_csv(os.path.join(out, "eval_per_shape.csv"), index=False)
    table.to_csv(os.path.join(out, "eval_per_class.csv"), index=False)
    summary.to_csv(os.path.join(out, "eval_summary.csv"), index=False)
    if not args.no_plot:
        save_plot(table, args.k, os.path.join(out, "eval_per_class.html"))

    curves = roc_curves(labels, dist)
    macro_area = curve_area(np.mean(list(curves.values()), axis=0))
    if not args.no_plot:
        save_roc_plot(curves, os.path.join(out, "eval_roc_curves.html"),
                      dict(zip(table["class"], table["auroc"])),
                      table["auroc"].mean(), per_shape["auroc"].mean())

    print("\nGrand aggregates:")
    print(summary.to_string(index=False, float_format="{:.3f}".format))
    print_extremes(table)
    print(f"\nArea under the macro-average ROC curve: {macro_area:.3f} "
          "(should be close to the macro AUROC above)")
    print(f"\nSaved to {out}: eval_per_shape.csv, eval_per_class.csv, "
          "eval_summary.csv" + ("" if args.no_plot else
          ", eval_per_class.html, eval_roc_curves.html"))


if __name__ == "__main__":
    main()