# """Dimensionality reduction (t-SNE) of the standardized feature database,
# plus the interactive scatterplot of the result.

# Reads the standardized sheet (name, class, 6 standardized scalars, 5
# histograms of 50 bins), drops the test classes, runs t-SNE to 2D and saves
# the coordinates to the sheets folder. The scatterplot (Plotly) is built from
# those saved coordinates: points coloured by class, hover shows name and class.

# Usage (from the project root):
#     python -m mesh_browser.dr 5 30 50              # compute + plot each
#     python -m mesh_browser.dr 30 --plot-only       # plot saved results only
#     python -m mesh_browser.dr 30 --no-browser      # write the HTML, do not open

# Variants (each gets its own tag in the output file names):
#     --scale-blocks   scale the scalars and each histogram to equal weight
#     --cumulative     use cumulative histograms (L1/Euclidean ~ EMD)
#     --only scalars   t-SNE on the 6 scalars only   (or: --only hists)

#     --hellinger      square root of the histogram bins (Hellinger distance)

# Evaluation without t-SNE (kNN class purity in the full 256D space):
#     python -m mesh_browser.dr --evaluate
#     python -m mesh_browser.dr --optimize [--cumulative]   # saves best shares
#     python -m mesh_browser.dr 30 --use-best-shares [--cumulative]
#     python -m mesh_browser.dr 5 15 30 50 --sweep --seeds 5 --use-best-shares --hellinger
#                                                      # score many t-SNE runs, keep best
#     python -m mesh_browser.dr --explain D00681.obj   # why does this shape get
#                                                      # these neighbours?
# """

# import argparse
# import json
# import os
# import webbrowser

# import numpy as np
# import pandas as pd
# import plotly.express as px
# from sklearn.manifold import TSNE, trustworthiness

# from . import config
# from .map_eval import mean_average_precision

# STANDARDIZED_FILE = os.path.join(config.SHEETS_DIR, "normalized_features.xlsx")
# STANDARDIZED_SHEET = "zscore"  # the sheet inside the workbook with the z-scores

# # Classes that are not real shape classes; removed BEFORE t-SNE.
# # TODO: add the second test class.
# TEST_CLASSES = ["0_normalizing_tests", "0_resampling_tests"]

# N_SCALARS = 6
# N_HISTOGRAMS = 5  # A3, D1, D2, D3, D4
# N_BINS = 50
# EXPECTED_FEATURES = N_SCALARS + N_HISTOGRAMS * N_BINS  # 256
# RANDOM_STATE = 0
# DEFAULT_PERPLEXITIES = [30]

# # Column groups inside the 256-value vector: scalars first, then 5 histograms.
# GROUPS = {"scalars": slice(0, N_SCALARS)}
# for _i, _name in enumerate(["A3", "D1", "D2", "D3", "D4"]):
#     _start = N_SCALARS + _i * N_BINS
#     GROUPS[_name] = slice(_start, _start + N_BINS)

# # Share of the total squared distance each group should get with
# # --scale-blocks. Must add up to 1. Equal shares = 1/6 each.
# # Example: SHARES["scalars"] = 1/3, the five histograms split the other 2/3.
# SHARES = {name: 1 / len(GROUPS) for name in GROUPS}
# SHARES["scalars"] = 1 / 3
# _rest = (1 - SHARES["scalars"]) / (len(GROUPS) - 1)
# for _name in GROUPS:
#     if _name != "scalars":
#         SHARES[_name] = _rest
# assert abs(sum(SHARES.values()) - 1) < 1e-9, "SHARES must add up to 1"

# # After scaling every group has squared distance ~1, so the multiplier that
# # gives a group its share is sqrt(share). Multiplying all of them by the same
# # constant (sqrt(6) here) changes nothing for t-SNE, so equal shares -> 1.0.
# WEIGHTS = {name: (len(GROUPS) * SHARES[name]) ** 0.5 for name in GROUPS}
# N_PAIRS = 5000  # random pairs used to estimate group scales
# K_EVAL = 10     # neighbours used by the kNN class purity

# # Enough distinct colours for many classes; cycled if there are more.
# PALETTE = (
#     px.colors.qualitative.Alphabet
#     + px.colors.qualitative.Dark24
#     + px.colors.qualitative.Light24
# )


# # ----------------------------------------------------------------- t-SNE

# def best_shares_path(hellinger=False):
#     """File with the shares found by --optimize (separate per variant)."""
#     suffix = "_hel" if hellinger else ""
#     return os.path.join(config.SHEETS_DIR, f"best_shares{suffix}.json")


# def tsne_path(perplexity, tag=""):
#     """Output CSV for a given perplexity (and variant tag)."""
#     return os.path.join(config.SHEETS_DIR, f"tsne_2d_p{perplexity}{tag}.csv")


# def html_path(perplexity, tag=""):
#     """Output HTML (scatterplot) for a given perplexity (and variant tag)."""
#     return os.path.join(config.SHEETS_DIR, f"tsne_map_p{perplexity}{tag}.html")


# def variant_tag(scale_blocks=False, cumulative=False, only="all",
#                 best_shares=False, hellinger=False):
#     """File name suffix describing the variant ("" for the plain one)."""
#     tag = ""
#     if hellinger:
#         tag += "_hel"
#     if cumulative:
#         tag += "_cum"
#     if scale_blocks:
#         tag += "_scaled"
#     if best_shares:
#         tag += "_opt"
#     if only != "all":
#         tag += f"_{only}"
#     return tag


# def load_feature_table(path=STANDARDIZED_FILE, drop_classes=TEST_CLASSES):
#     """Load the standardized sheet.

#     Returns (info, X): info has the columns name and class, X is a float
#     matrix (rows = shapes, columns = the 256 features).
#     """
#     df = pd.read_excel(path, sheet_name=STANDARDIZED_SHEET)
#     feature_cols = [c for c in df.columns if c not in ("name", "class")]

#     # Decimal commas stored as text -> numbers.
#     for col in feature_cols:
#         if df[col].dtype == object:
#             df[col] = pd.to_numeric(
#                 df[col].astype(str).str.replace(",", ".", regex=False),
#                 errors="coerce",
#             )

#     df = df[~df["class"].isin(drop_classes)].reset_index(drop=True)
#     X = df[feature_cols].to_numpy(dtype=float)

#     if X.shape[1] != EXPECTED_FEATURES:
#         raise ValueError(
#             f"Expected {EXPECTED_FEATURES} feature columns, found {X.shape[1]}."
#         )
#     if np.isnan(X).any():
#         bad = np.unique(np.argwhere(np.isnan(X))[:, 1])
#         raise ValueError(
#             "NaN values in columns: " + ", ".join(feature_cols[i] for i in bad)
#         )
#     return df[["name", "class"]], X


# def group_scales(X, n_pairs=N_PAIRS, seed=0):
#     """Mean Euclidean distance between random pairs, per column group."""
#     rng = np.random.default_rng(seed)
#     i = rng.integers(0, len(X), n_pairs)
#     j = rng.integers(0, len(X), n_pairs)
#     keep = i != j
#     i, j = i[keep], j[keep]
#     return {
#         name: np.linalg.norm(X[i][:, sl] - X[j][:, sl], axis=1).mean()
#         for name, sl in GROUPS.items()
#     }


# def group_report(X, label=""):
#     """Print each group's share of the squared distance of a random pair."""
#     scales = group_scales(X)
#     sq = {name: s ** 2 for name, s in scales.items()}  # approx. mean sq. dist
#     total = sum(sq.values())
#     print(f"share of squared distance per group {label}:")
#     for name in GROUPS:
#         print(f"  {name:8s} {100 * sq[name] / total:6.2f} %")


# def transform_features(X, scale_blocks=False, cumulative=False, only="all",
#                        hellinger=False):
#     """Optionally cumulate histograms, scale the groups, select columns."""
#     X = X.copy()
#     if hellinger:
#         # sqrt of a proportion histogram: Euclidean distance between the
#         # results is the Hellinger distance between the histograms.
#         for name, sl in GROUPS.items():
#             if name != "scalars":
#                 X[:, sl] = np.sqrt(np.maximum(X[:, sl], 0))
#     if cumulative:
#         for name, sl in GROUPS.items():
#             if name != "scalars":
#                 X[:, sl] = np.cumsum(X[:, sl], axis=1)
#     if scale_blocks:
#         scales = group_scales(X)
#         for name, sl in GROUPS.items():
#             X[:, sl] *= WEIGHTS[name] / scales[name]
#     return select_columns(X, only)


# def select_columns(X, only="all"):
#     """Keep all columns, only the scalars, or only the histograms."""
#     if only == "scalars":
#         return X[:, GROUPS["scalars"]]
#     if only == "hists":
#         return X[:, N_SCALARS:]
#     return X


# def run_tsne(X, perplexity, random_state=RANDOM_STATE, return_kl=False):
#     """2D t-SNE embedding of X (Euclidean distance, PCA initialisation)."""
#     if perplexity >= len(X):
#         raise ValueError(f"Perplexity {perplexity} must be below {len(X)} shapes.")
#     tsne = TSNE(
#         n_components=2,
#         metric="euclidean",
#         init="pca",
#         learning_rate="auto",
#         perplexity=perplexity,
#         random_state=random_state,
#     )
#     Y = tsne.fit_transform(X)
#     return (Y, float(tsne.kl_divergence_)) if return_kl else Y


# def compute_and_save(perplexity, path=STANDARDIZED_FILE, scale_blocks=False,
#                      cumulative=False, only="all", best_shares=False,
#                      hellinger=False):
#     """Run t-SNE for one perplexity and write the CSV. Returns the file path."""
#     info, X = load_feature_table(path)
#     X = transform_features(X, scale_blocks, cumulative, "all", hellinger)
#     group_report(X, "AFTER transform (this is what t-SNE sees)")
#     X = select_columns(X, only)
#     tag = variant_tag(scale_blocks, cumulative, only, best_shares, hellinger)
#     print(f"{len(X)} shapes, {X.shape[1]} features, perplexity {perplexity}, "
#           f"variant '{tag or 'plain'}' ...")
#     Y = run_tsne(X, perplexity)

#     # How well did the 2D map keep the class neighbourhoods?
#     labels, names = pd.factorize(info["class"])
#     ps_high = per_shape_purity(squared_distances(X), labels)
#     ps_2d = per_shape_purity(squared_distances(Y), labels)
#     print(f"kNN purity (K={K_EVAL}): input space {ps_high.mean():.3f} -> "
#           f"2D map {ps_2d.mean():.3f} (chance {chance_purity(labels):.3f})")
#     table = class_table(labels, names, purity_input=ps_high, purity_2d=ps_2d)
#     table_path = os.path.join(config.SHEETS_DIR, f"class_purity_p{perplexity}{tag}.csv")
#     os.makedirs(config.SHEETS_DIR, exist_ok=True)
#     table.to_csv(table_path, index=False)
#     print("saved", table_path)

#     out = info.copy()
#     out["x"] = Y[:, 0]
#     out["y"] = Y[:, 1]
#     out["perplexity"] = perplexity
#     out["variant"] = tag or "plain"
#     out["source"] = os.path.basename(path)

#     os.makedirs(config.SHEETS_DIR, exist_ok=True)
#     out_path = tsne_path(perplexity, tag)
#     out.to_csv(out_path, index=False)
#     print("saved", out_path)
#     return out_path


# # ------------------------------------------------------------ evaluation
# # Quick, t-SNE-free quality check in the full 256D space: kNN class purity =
# # the share of each shape's K nearest neighbours that have the same class.

# def set_shares(shares):
#     """Replace SHARES and WEIGHTS in place (used for --use-best-shares)."""
#     total = sum(shares.values())
#     for name in GROUPS:
#         SHARES[name] = shares[name] / total
#         WEIGHTS[name] = (len(GROUPS) * SHARES[name]) ** 0.5


# def squared_distances(A):
#     """Matrix of squared Euclidean distances between all rows of A."""
#     sq = (A ** 2).sum(axis=1)
#     D = sq[:, None] + sq[None, :] - 2 * A @ A.T
#     return np.maximum(D, 0).astype(np.float32)


# def l1_distances(A, chunk=50):
#     """Matrix of L1 (city block) distances between all rows of A."""
#     n = len(A)
#     D = np.empty((n, n), dtype=np.float32)
#     for start in range(0, n, chunk):
#         diff = A[start:start + chunk, None, :] - A[None, :, :]
#         D[start:start + chunk] = np.abs(diff).sum(axis=2)
#     return D


# def group_distance_matrices(X, cumulative=False, emd=False, hellinger=False):
#     """Squared distance matrix per group, divided by the group's scale^2.

#     Combining them as sum(share * matrix) equals the squared distance of the
#     block-scaled vector (up to one constant), so shares can be tried quickly.
#     With emd=True the histogram groups use the Earth Mover's Distance (L1 on
#     cumulative histograms) instead of Euclidean; the scalars stay Euclidean.
#     """
#     Xt = transform_features(X, scale_blocks=False, cumulative=cumulative or emd,
#                             hellinger=hellinger)
#     if not emd:
#         scales = group_scales(Xt)
#         return {name: squared_distances(Xt[:, sl]) / scales[name] ** 2
#                 for name, sl in GROUPS.items()}
#     n = len(Xt)
#     mats = {}
#     for name, sl in GROUPS.items():
#         if name == "scalars":
#             d = np.sqrt(squared_distances(Xt[:, sl]))
#         else:
#             d = l1_distances(Xt[:, sl])
#         scale = d.sum() / (n * (n - 1))  # mean distance over all pairs
#         mats[name] = (d / scale) ** 2
#     return mats


# def combine(mats, shares):
#     """Weighted sum of the per-group squared distance matrices."""
#     return sum(shares[name] * mats[name] for name in GROUPS)


# def per_shape_purity(D, labels, k=K_EVAL):
#     """For every shape: share of its k nearest neighbours with the same class.

#     D is an (n, n) distance matrix, labels an integer array of length n.
#     """
#     D = D.copy()
#     np.fill_diagonal(D, np.inf)  # a shape is not its own neighbour
#     idx = np.argpartition(D, k, axis=1)[:, :k]
#     return (labels[idx] == labels[:, None]).mean(axis=1)


# def knn_purity(D, labels, k=K_EVAL):
#     """Mean kNN class purity over all shapes."""
#     return float(per_shape_purity(D, labels, k).mean())


# def class_table(labels, names, **per_shape):
#     """Per-class mean of per-shape values, sorted by the first column."""
#     sizes = np.bincount(labels)
#     data = {"class": np.asarray(names), "size": sizes}
#     for key, vals in per_shape.items():
#         data[key] = np.bincount(labels, weights=vals) / sizes
#     return pd.DataFrame(data).sort_values(list(per_shape)[0], ascending=False)


# def chance_purity(labels):
#     """Expected purity if neighbours were random (the baseline to beat)."""
#     p = np.bincount(labels) / len(labels)
#     return float((p ** 2).sum())


# def evaluate_groups(X, labels, names, all_variants=False):
#     """Print kNN purity per group (raw, cumulative, EMD) and per class."""
#     print(f"kNN purity (K={K_EVAL}), {len(X)} shapes, "
#           f"{labels.max() + 1} classes; chance level {chance_purity(labels):.3f}")
#     variants = {
#         "raw": group_distance_matrices(X),
#         "Hellinger": group_distance_matrices(X, hellinger=True),
#     }
#     if all_variants:
#         variants["cumulative"] = group_distance_matrices(X, cumulative=True)
#         variants["EMD (L1 cum.)"] = group_distance_matrices(X, emd=True)
#     print(f"  {'group':8s}" + "".join(f"{v:>15s}" for v in variants))
#     for name in GROUPS:
#         row = "".join(f"{knn_purity(m[name], labels):15.3f}"
#                       for m in variants.values())
#         print(f"  {name:8s}{row}")
#     equal = {name: 1 / len(GROUPS) for name in GROUPS}
#     for label, mats in variants.items():
#         print(f"  all groups, {label}: equal shares "
#               f"{knn_purity(combine(mats, equal), labels):.3f}, current SHARES "
#               f"{knn_purity(combine(mats, SHARES), labels):.3f}")

#     # per class, for the raw vector with the current shares
#     ps = per_shape_purity(combine(variants["raw"], SHARES), labels)
#     table = class_table(labels, names, purity=ps)
#     os.makedirs(config.SHEETS_DIR, exist_ok=True)
#     out = os.path.join(config.SHEETS_DIR, "class_purity.csv")
#     table.to_csv(out, index=False)
#     print("\nbest 10 classes:")
#     print(table.head(10).to_string(index=False))
#     print("worst 10 classes:")
#     print(table.tail(10).to_string(index=False))
#     print("saved", out)


# def explain_shape(X, info, name, k=K_EVAL, hellinger=False):
#     """Print a shape's nearest neighbours and what drives each distance.

#     For every neighbour the share of the total (scaled, squared) distance
#     that comes from each group is shown, plus the rank of the first
#     same-class shapes. This shows which group pulls a shape away from its class.
#     """
#     names = np.array(info["name"].astype(str).tolist(), dtype=str)
#     classes = np.array(info["class"].astype(str).tolist(), dtype=str)
#     hits = np.flatnonzero(names == name)
#     if len(hits) == 0:
#         hits = np.flatnonzero(np.char.find(names, name) >= 0)  # substring
#     if len(hits) == 0:
#         raise ValueError(f"No shape named like '{name}' in the sheet.")
#     i = int(hits[0])

#     mats = group_distance_matrices(X, hellinger=hellinger)
#     D = combine(mats, SHARES)
#     d = D[i].copy()
#     d[i] = np.inf
#     order = np.argsort(d)

#     print(f"\nshape {names[i]}  class {classes[i]}")
#     print("scalars (z-scores): " + "  ".join(
#         f"{v:6.2f}" for v in X[i, :N_SCALARS]))
#     groups = list(GROUPS)
#     print(f"\n{'rank':>4s} {'name':14s} {'class':22s} {'dist':>6s}  "
#           + " ".join(f"{g:>7s}" for g in groups) + "   (% of distance)")
#     for rank, j in enumerate(order[:k], start=1):
#         parts = [100 * SHARES[g] * mats[g][i, j] / D[i, j] for g in groups]
#         mark = "*" if classes[j] == classes[i] else " "
#         print(f"{rank:4d}{mark}{names[j]:14s} {classes[j]:22s} {d[j]:6.2f}  "
#               + " ".join(f"{p:7.1f}" for p in parts))
#     same = np.flatnonzero(classes[order] == classes[i])[:5]
#     print("\n(* = same class)  ranks of the first same-class shapes:",
#           [int(r) + 1 for r in same])
#     for r in same[:3]:
#         j = order[r]
#         parts = [100 * SHARES[g] * mats[g][i, j] / D[i, j] for g in groups]
#         print(f"  rank {r + 1:4d} {names[j]:14s} dist {d[j]:6.2f}  "
#               + " ".join(f"{p:7.1f}" for p in parts))


# def optimize_shares(mats, labels, n_random=200, n_refine=300, seed=0):
#     """Search for group shares that maximize kNN purity.

#     Random shares first, then hill climbing from the best one. Returns the
#     best shares (summing to 1) and their purity.
#     """
#     rng = np.random.default_rng(seed)
#     names = list(GROUPS)

#     def score(vec):
#         shares = dict(zip(names, vec))
#         return knn_purity(combine(mats, shares), labels)

#     best_vec = np.full(len(names), 1 / len(names))
#     best = score(best_vec)
#     print(f"start (equal shares): {best:.3f}")
#     for _ in range(n_random):
#         vec = rng.dirichlet(np.ones(len(names)))
#         sc = score(vec)
#         if sc > best:
#             best, best_vec = sc, vec
#     print(f"after random search: {best:.3f}")
#     for _ in range(n_refine):
#         vec = best_vec * np.exp(rng.normal(0, 0.3, len(names)))
#         vec /= vec.sum()
#         sc = score(vec)
#         if sc > best:
#             best, best_vec = sc, vec
#     print(f"after refinement:    {best:.3f}")
#     return dict(zip(names, (float(v) for v in best_vec))), best


# # ------------------------------------------------------------------ sweep
# # t-SNE is stochastic, so run several seeds per perplexity, score every map
# # and keep the best. Scores:
# #   purity_2d        kNN class purity measured in the 2D map (uses the labels)
# #   trustworthiness  share of 2D neighbours that are also input-space
# #                    neighbours (no labels; 1.0 = perfect)
# #   kl               t-SNE's own objective (only comparable at equal perplexity)

# def write_map_csv(info, Y, perplexity, tag, seed):
#     """Write one 2D map (name, class, x, y, ...) and return the file path."""
#     out = info.copy()
#     out["x"] = Y[:, 0]
#     out["y"] = Y[:, 1]
#     out["perplexity"] = perplexity
#     out["variant"] = tag or "plain"
#     out["seed"] = seed
#     out["source"] = os.path.basename(STANDARDIZED_FILE)
#     os.makedirs(config.SHEETS_DIR, exist_ok=True)
#     path = tsne_path(perplexity, tag)
#     out.to_csv(path, index=False)
#     return path


# def sweep(perplexities, n_seeds=5, scale_blocks=False, cumulative=False,
#           only="all", best_shares=False, hellinger=False, metric="purity_2d"):
#     """Run t-SNE for every perplexity and seed, keep the best map per perplexity.

#     Returns (best perplexity, best seed) over all runs for the chosen metric.
#     """
#     info, X = load_feature_table()
#     X = transform_features(X, scale_blocks, cumulative, "all", hellinger)
#     X = select_columns(X, only)
#     tag = variant_tag(scale_blocks, cumulative, only, best_shares, hellinger)
#     labels, _ = pd.factorize(info["class"])
#     print(f"{len(X)} shapes, {X.shape[1]} features, variant '{tag or 'plain'}'")
#     print(f"input-space kNN purity {knn_purity(squared_distances(X), labels):.3f}"
#           f" (chance {chance_purity(labels):.3f}); selecting by {metric}")

#     rows, best_per_p = [], {}
#     for p in perplexities:
#         for seed in range(n_seeds):
#             Y, kl = run_tsne(X, p, random_state=seed, return_kl=True)
#             row = {
#                 "perplexity": p,
#                 "seed": seed,
#                 "purity_2d": knn_purity(squared_distances(Y), labels),
#                 "trustworthiness": float(trustworthiness(X, Y, n_neighbors=K_EVAL)),
#                 "map": mean_average_precision(squared_distances(Y), labels),
#                 "kl": kl,
#             }
#             rows.append(row)
#             print(f"  perplexity {p:3d} seed {seed}: purity {row['purity_2d']:.3f}"
#                   f"  trust {row['trustworthiness']:.3f}  KL {kl:.3f}")
#             if p not in best_per_p or row[metric] > best_per_p[p][0]:
#                 best_per_p[p] = (row[metric], seed, Y)

#     df = pd.DataFrame(rows)
#     os.makedirs(config.SHEETS_DIR, exist_ok=True)
#     sweep_file = os.path.join(config.SHEETS_DIR, f"tsne_sweep{tag}.csv")
#     df.to_csv(sweep_file, index=False)

#     print("\nmean / std over seeds, per perplexity:")
#     summary = df.groupby("perplexity")[["purity_2d", "trustworthiness", "kl"]]
#     print(summary.agg(["mean", "std"]).round(3).to_string())

#     for p, (score, seed, Y) in best_per_p.items():
#         path = write_map_csv(info, Y, p, tag, seed)
#         print(f"best map for perplexity {p}: seed {seed}, {metric} {score:.3f}"
#               f" -> {path}")
#     best_p = max(best_per_p, key=lambda p: best_per_p[p][0])
#     print(f"overall best: perplexity {best_p}, seed {best_per_p[best_p][1]}"
#           f" ({metric} {best_per_p[best_p][0]:.3f})")
#     print("saved", sweep_file)
#     return best_p, best_per_p[best_p][1]


# # ----------------------------------------------------------- scatterplot

# def build_figure(csv_path):
#     """Build the Plotly figure from a t-SNE CSV (name, class, x, y)."""
#     df = pd.read_csv(csv_path)
#     classes = sorted(df["class"].unique())
#     colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}

#     perplexity = df["perplexity"].iloc[0] if "perplexity" in df else "?"
#     variant = df["variant"].iloc[0] if "variant" in df else "plain"
#     if "seed" in df:
#         variant += f", seed {df['seed'].iloc[0]}"
#     fig = px.scatter(
#         df,
#         x="x",
#         y="y",
#         color="class",
#         color_discrete_map=colors,
#         category_orders={"class": classes},
#         hover_name="name",
#         hover_data={"class": True, "x": False, "y": False},
#         render_mode="webgl",
#         title=f"t-SNE map of the shape database "
#         f"({len(df)} shapes, perplexity {perplexity}, {variant})",
#     )
#     fig.update_traces(marker=dict(size=7, opacity=0.85))
#     # t-SNE axes have no meaning, so hide them and keep distances undistorted.
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
#     fig.update_layout(legend_title_text="Class", height=800)
#     return fig


# def show_map(perplexity=30, open_browser=True, tag=""):
#     """Write the scatterplot HTML for one perplexity/variant and open it."""
#     csv_path = tsne_path(perplexity, tag)
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(
#             f"{csv_path} not found. Run t-SNE first (python -m mesh_browser.dr)."
#         )
#     out = html_path(perplexity, tag)
#     build_figure(csv_path).write_html(out, include_plotlyjs=True)
#     print("saved", out)
#     if open_browser:
#         webbrowser.open("file://" + os.path.abspath(out))
#     return out


# # ------------------------------------------------------------------- CLI

# def main():
#     parser = argparse.ArgumentParser(description="t-SNE map of the database")
#     parser.add_argument("perplexities", nargs="*", type=int,
#                         default=DEFAULT_PERPLEXITIES)
#     parser.add_argument("--plot-only", action="store_true",
#                         help="only plot already computed results")
#     parser.add_argument("--no-browser", action="store_true",
#                         help="write the HTML but do not open it")
#     parser.add_argument("--scale-blocks", action="store_true",
#                         help="scale scalars and each histogram to equal weight")
#     parser.add_argument("--cumulative", action="store_true",
#                         help="use cumulative histograms")
#     parser.add_argument("--only", choices=["all", "scalars", "hists"],
#                         default="all", help="restrict the features used")
#     parser.add_argument("--evaluate", action="store_true",
#                         help="print kNN class purity per group and exit")
#     parser.add_argument("--optimize", action="store_true",
#                         help="search group shares for best kNN purity and exit")
#     parser.add_argument("--use-best-shares", action="store_true",
#                         help="use the shares saved by --optimize (implies "
#                              "--scale-blocks)")
#     parser.add_argument("--hellinger", action="store_true",
#                         help="square root of the histogram bins")
#     parser.add_argument("--evaluate-all", action="store_true",
#                         help="--evaluate also tests cumulative and EMD")
#     parser.add_argument("--sweep", action="store_true",
#                         help="run several seeds per perplexity, keep the best")
#     parser.add_argument("--seeds", type=int, default=5,
#                         help="seeds per perplexity for --sweep (default 5)")
#     parser.add_argument("--metric", choices=["purity", "trust", "map"], default="purity",
#                         help="score used by --sweep to pick the best map")
#     parser.add_argument("--explain", metavar="NAME",
#                         help="show the neighbours of one shape and why")
#     parser.add_argument("--evaluate-map", action="store_true",
#                         help="print Mean Average Precision (mAP) for the chosen variant or saved 2D map")
#     args = parser.parse_args()

#     if args.evaluate or args.evaluate_all or args.optimize or args.explain or args.evaluate_map:
#         info, X = load_feature_table()
#         labels, names = pd.factorize(info["class"])
#         if args.use_best_shares:
#             with open(best_shares_path(args.hellinger)) as f:
#                 set_shares(json.load(f))
#         if args.evaluate or args.evaluate_all:
#             evaluate_groups(X, labels, names, all_variants=args.evaluate_all)
#         if args.explain:
#             explain_shape(X, info, args.explain, hellinger=args.hellinger)
#         if args.evaluate_map:
#             tag = variant_tag(args.scale_blocks, args.cumulative, args.only,
#                               args.use_best_shares, args.hellinger)
#             p = args.perplexities[0] if args.perplexities else DEFAULT_PERPLEXITIES[0]
#             csv_path = tsne_path(p, tag)
            
#             # If a saved 2D t-SNE CSV exists for this configuration, evaluate on the 2D coordinates.
#             # Otherwise, evaluate directly on the high-dimensional transformed feature space.
#             if os.path.exists(csv_path):
#                 df_tsne = pd.read_csv(csv_path)
#                 Y = df_tsne[["x", "y"]].to_numpy()
#                 D = squared_distances(Y)
#                 print(f"mAP in 2D t-SNE space (perplexity {p}, variant '{tag or 'plain'}'): {mean_average_precision(D, labels):.4f}")
#             else:
#                 Xt = transform_features(X, args.scale_blocks, args.cumulative, args.only, args.hellinger)
#                 Xt = select_columns(Xt, args.only)
#                 D = squared_distances(Xt)
#                 print(f"mAP in feature space (variant '{tag or 'plain'}'): {mean_average_precision(D, labels):.4f}")
#         if args.optimize:
#             mats = group_distance_matrices(X, cumulative=args.cumulative,
#                                            hellinger=args.hellinger)
#             best_shares, best = optimize_shares(mats, labels)
#             with open(best_shares_path(args.hellinger), "w") as f:
#                 json.dump(best_shares, f, indent=2)
#             print("best shares:", {k: round(v, 3) for k, v in best_shares.items()})
#             print("saved", best_shares_path(args.hellinger))
#         return

#     if args.use_best_shares:
#         with open(best_shares_path(args.hellinger)) as f:
#             set_shares(json.load(f))
#         args.scale_blocks = True
#         print("using shares:", {k: round(v, 3) for k, v in SHARES.items()})

#     tag = variant_tag(args.scale_blocks, args.cumulative, args.only,
#                       args.use_best_shares, args.hellinger)
#     if args.sweep:
#         if args.metric == "purity":
#             metric = "purity_2d"
#         elif args.metric == "trust":
#             metric = "trustworthiness"
#         else:
#             metric = "map"
        
#         best_p, _ = sweep(args.perplexities, args.seeds, args.scale_blocks,
#                           args.cumulative, args.only, args.use_best_shares,
#                           args.hellinger, metric)
#         show_map(best_p, open_browser=not args.no_browser, tag=tag)
#         return

#     if not args.plot_only:
#         group_report(load_feature_table()[1], "BEFORE transform (raw)")
#     for p in args.perplexities:
#         if not args.plot_only:
#             compute_and_save(p, scale_blocks=args.scale_blocks,
#                              cumulative=args.cumulative, only=args.only,
#                              best_shares=args.use_best_shares,
#                              hellinger=args.hellinger)
#         show_map(p, open_browser=not args.no_browser, tag=tag)


# if __name__ == "__main__":
#     main()
"""t-SNE map of the standardized feature database (command line entry).

Modules:
    dr_paths      file names and the Variant (which vector variant)
    dr_features   loading the sheet, groups, shares, Hellinger / block scaling
    dr_scoring    distances, kNN purity, mAP, evaluation tables, explain
    dr_tuning     t-SNE runs, seed/perplexity sweep, share optimization
    dr_plotting   the Plotly scatterplot

Usage (from the project root):
    python -m mesh_browser.dr 5 30 50                 # t-SNE + plot each
    python -m mesh_browser.dr 30 --plot-only          # plot saved maps only
    python -m mesh_browser.dr 5 30 --seeds 5          # 5 seeds each, keep best

Variant flags (each variant has its own file names, e.g. _hel_scaled_opt):
    --hellinger        square root of the histogram bins
    --scale-blocks     scale the scalars and each histogram to their shares
    --use-best-shares  shares saved by --optimize (implies --scale-blocks)

Evaluation without t-SNE:
    python -m mesh_browser.dr --evaluate
    python -m mesh_browser.dr --optimize [--hellinger]       # saves best shares
    python -m mesh_browser.dr 30 --evaluate-map --hellinger --use-best-shares
    python -m mesh_browser.dr --explain D00681.obj           # neighbours of a shape

Example final run:
    python -m mesh_browser.dr 5 15 30 50 --seeds 5 --hellinger --use-best-shares --metric map
"""

import argparse

from .dr_paths import Variant

DEFAULT_PERPLEXITIES = [30]


def build_parser():
    parser = argparse.ArgumentParser(description="t-SNE map of the database")
    parser.add_argument("perplexities", nargs="*", type=int,
                        default=DEFAULT_PERPLEXITIES)
    parser.add_argument("--plot-only", action="store_true",
                        help="only plot already computed maps")
    parser.add_argument("--no-browser", action="store_true",
                        help="write the HTML but do not open it")
    parser.add_argument("--seeds", type=int, default=1,
                        help="t-SNE seeds per perplexity; the best map is kept")
    parser.add_argument("--metric", choices=["purity", "trust", "map"],
                        default="purity",
                        help="score used to pick the best seed")
    parser.add_argument("--hellinger", action="store_true",
                        help="square root of the histogram bins")
    parser.add_argument("--scale-blocks", action="store_true",
                        help="scale scalars and each histogram to their shares")
    parser.add_argument("--use-best-shares", action="store_true",
                        help="use the shares saved by --optimize")
    parser.add_argument("--evaluate", action="store_true",
                        help="print kNN purity per group and per class, exit")
    parser.add_argument("--optimize", action="store_true",
                        help="search group shares for best purity, exit")
    parser.add_argument("--evaluate-map", action="store_true",
                        help="print mAP of the saved 2D map (or feature space)")
    parser.add_argument("--explain", metavar="NAME",
                        help="show the neighbours of one shape and why")
    return parser


def run_analysis(args, variant):
    """The modes that print numbers and exit (no t-SNE, no plot)."""
    import pandas as pd

    from .dr_features import get_shares, load_feature_table
    from .dr_scoring import (evaluate_groups, evaluate_map, explain_shape,
                             group_distance_matrices)

    info, X = load_feature_table()
    labels, names = pd.factorize(info["class"])
    shares = get_shares(variant)

    if args.evaluate:
        evaluate_groups(X, labels, names, shares)
    if args.explain:
        explain_shape(X, info, args.explain, variant, shares)
    if args.evaluate_map:
        p = args.perplexities[0]
        print(evaluate_map(X, labels, variant, shares, p))
    if args.optimize:
        from .dr_tuning import optimize_shares, save_best_shares
        mats = group_distance_matrices(X, hellinger=args.hellinger)
        best_shares, _ = optimize_shares(mats, labels)
        path = save_best_shares(best_shares, args.hellinger)
        print("best shares:", {k: round(v, 3) for k, v in best_shares.items()})
        print("saved", path)


def main():
    args = build_parser().parse_args()
    variant = Variant(scale_blocks=args.scale_blocks,
                      hellinger=args.hellinger,
                      best_shares=args.use_best_shares)

    if args.evaluate or args.optimize or args.evaluate_map or args.explain:
        run_analysis(args, variant)
        return

    if not args.plot_only:
        from .dr_features import get_shares
        from .dr_tuning import METRICS, sweep
        sweep(args.perplexities, args.seeds, variant, get_shares(variant),
              METRICS[args.metric])

    from .dr_plotting import show_map
    for p in args.perplexities:
        show_map(p, variant, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()