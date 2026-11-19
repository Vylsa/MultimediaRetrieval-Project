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