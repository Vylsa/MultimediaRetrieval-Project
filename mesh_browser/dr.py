"""Dimensionality reduction (t-SNE) of the standardized feature database,
plus the interactive scatterplot of the result.

Reads the standardized sheet (name, class, 6 standardized scalars, 5
histograms of 50 bins), drops the test classes, runs t-SNE to 2D and saves
the coordinates to the sheets folder. The scatterplot (Plotly) is built from
those saved coordinates: points coloured by class, hover shows name and class.

Usage (from the project root):
    python -m mesh_browser.dr 5 30 50              # compute + plot each
    python -m mesh_browser.dr 30 --plot-only       # plot saved results only
    python -m mesh_browser.dr 30 --no-browser      # write the HTML, do not open
"""

import argparse
import os
import webbrowser

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

import config

N_SCALARS = 6
N_HISTOGRAMS = 5  # A3, D1, D2, D3, D4
N_BINS = 50
EXPECTED_FEATURES = N_SCALARS + N_HISTOGRAMS * N_BINS  # 256
RANDOM_STATE = 0
DEFAULT_PERPLEXITIES = [30]

# Enough distinct colours for many classes; cycled if there are more.
PALETTE = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
)


# ----------------------------------------------------------------- t-SNE

def tsne_path(perplexity):
    """Output CSV for a given perplexity."""
    return os.path.join(config.SHEETS_DIR, f"tsne_2d_p{perplexity}.csv")


def html_path(perplexity):
    """Output HTML (scatterplot) for a given perplexity."""
    return os.path.join(config.SHEETS_DIR, f"tsne_map_p{perplexity}.html")


def load_feature_table(path=config.STANDARDIZED_FILE, drop_classes=config.TEST_CLASSES):
    """Load the standardized sheet.

    Returns (info, X): info has the columns name and class, X is a float
    matrix (rows = shapes, columns = the 256 features).
    """
    df = pd.read_excel(path, sheet_name=config.STANDARDIZED_SHEET)
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


def run_tsne(X, perplexity, random_state=RANDOM_STATE):
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
    return tsne.fit_transform(X)


def compute_and_save(perplexity, path=config.STANDARDIZED_FILE):
    """Run t-SNE for one perplexity and write the CSV. Returns the file path."""
    info, X = load_feature_table(path)
    print(f"{len(X)} shapes, {X.shape[1]} features, perplexity {perplexity} ...")
    Y = run_tsne(X, perplexity)

    out = info.copy()
    out["x"] = Y[:, 0]
    out["y"] = Y[:, 1]
    out["perplexity"] = perplexity
    out["source"] = os.path.basename(path)

    os.makedirs(config.SHEETS_DIR, exist_ok=True)
    out_path = tsne_path(perplexity)
    out.to_csv(out_path, index=False)
    print("saved", out_path)
    return out_path


# ----------------------------------------------------------- scatterplot

def build_figure(csv_path):
    """Build the Plotly figure from a t-SNE CSV (name, class, x, y)."""
    df = pd.read_csv(csv_path)
    classes = sorted(df["class"].unique())
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}

    perplexity = df["perplexity"].iloc[0] if "perplexity" in df else "?"
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="class",
        color_discrete_map=colors,
        category_orders={"class": classes},
        hover_name="name",
        hover_data={"class": True, "x": False, "y": False},
        render_mode="webgl",
        title=f"t-SNE map of the shape database "
        f"({len(df)} shapes, perplexity {perplexity})",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    # t-SNE axes have no meaning, so hide them and keep distances undistorted.
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(legend_title_text="Class", height=800)
    return fig


def show_map(perplexity=30, open_browser=True):
    """Write the scatterplot HTML for one perplexity and open it."""
    csv_path = tsne_path(perplexity)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run: python -m mesh_browser.dr {perplexity}"
        )
    out = html_path(perplexity)
    build_figure(csv_path).write_html(out, include_plotlyjs=True)
    print("saved", out)
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out))
    return out


# ------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="t-SNE map of the database")
    parser.add_argument("perplexities", nargs="*", type=int,
                        default=DEFAULT_PERPLEXITIES)
    parser.add_argument("--plot-only", action="store_true",
                        help="only plot already computed results")
    parser.add_argument("--no-browser", action="store_true",
                        help="write the HTML but do not open it")
    args = parser.parse_args()

    for p in args.perplexities:
        if not args.plot_only:
            compute_and_save(p)
        show_map(p, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()