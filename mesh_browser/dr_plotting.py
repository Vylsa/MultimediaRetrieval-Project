"""Interactive Plotly scatterplot of a saved 2D t-SNE map.

Every point is one shape, coloured by class. Hovering shows the shape name
and class; the legend toggles classes (click = hide/show, double click =
isolate); zoom and pan come with Plotly.
"""

import os
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.express as px

from .dr_paths import Variant, html_path, tsne_path

# Enough distinct colours for many classes; cycled if there are more.
PALETTE = (
    px.colors.qualitative.Alphabet
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
)


def build_figure(csv_path):
    """Build the Plotly figure from a t-SNE CSV (name, class, x, y)."""
    df = pd.read_csv(csv_path)
    classes = sorted(df["class"].unique())
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}

    perplexity = df["perplexity"].iloc[0] if "perplexity" in df else "?"
    variant = df["variant"].iloc[0] if "variant" in df else "plain"
    if "seed" in df:
        variant += f", seed {df['seed'].iloc[0]}"
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
        f"({len(df)} shapes, perplexity {perplexity}, {variant})",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    # t-SNE axes have no meaning, so hide them and keep distances undistorted.
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(legend_title_text="Class", height=800)
    return fig


def show_map(perplexity=30, variant=Variant(), open_browser=True):
    """Write the scatterplot HTML for one perplexity/variant and open it."""
    csv_path = tsne_path(perplexity, variant)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run t-SNE first (python -m mesh_browser.dr)."
        )
    out = html_path(perplexity, variant)
    build_figure(csv_path).write_html(out, include_plotlyjs=True)
    print("saved", out)
    if open_browser:
        webbrowser.open(Path(out).resolve().as_uri())  # handles spaces, backslashes
    return out