"""
Plot how many shapes in the database have each vertex count.

Usage:
    python plot_vertex_counts.py                       # uses the defaults below
    python plot_vertex_counts.py path/to/sheet.xlsx    # override the input file
    python plot_vertex_counts.py in.xlsx out.png       # override input and output

Requires: pandas, openpyxl, matplotlib, numpy
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Settings - change these to adapt the script
# ---------------------------------------------------------------------------
INPUT_FILE = SCRIPT_DIR.parent / "sheets" / "analyze_sheet_resampled.xlsx"     # path to the Excel file
SHEET_NAME = "analyze_sheet"           # sheet holding one row per shape
VERTEX_COLUMN = "Vertices"             # column with the vertex counts
OUTPUT_FILE = "vertex_count_distribution.png"

LOG_X = False        # log-spaced bins: the counts span 16 .. ~73k, so this reads best
N_BINS = 100         # number of histogram bins
DPI = 200

# Colours (light chart surface, single blue series)
BAR_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# ---------------------------------------------------------------------------


def load_vertex_counts(path, sheet, column):
    df = pd.read_excel(path, sheet_name=sheet)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
    return pd.to_numeric(df[column], errors="coerce").dropna().astype(int).to_numpy()


def thousands(x, _pos):
    if x >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"


def plot(counts, out_path):
    lo, hi = counts.min(), counts.max()
    if LOG_X:
        bins = np.logspace(np.log10(lo), np.log10(hi), N_BINS + 1)
    else:
        bins = np.linspace(lo, hi, N_BINS + 1)

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Bars with a thin surface-coloured gap between them
    heights, _, _ = ax.hist(counts, bins=bins, color=BAR_COLOR,
                            edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_ylim(0, heights.max() * 1.15)   # headroom for the line labels

    if LOG_X:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))

    # Median and mean reference lines, labelled directly
    median, mean = np.median(counts), counts.mean()
    ymax = ax.get_ylim()[1]
    for value, label, style in [(median, "median", "-"), (mean, "mean", "--")]:
        ax.axvline(value, color=TEXT_SECONDARY, linestyle=style,
                   linewidth=1.2, zorder=4)
        ax.text(value * (1.04 if LOG_X else 1), ymax * 0.99,
                f"{label} {value:,.0f}", color=TEXT_SECONDARY,
                fontsize=9.5, va="top", ha="left", zorder=5)

    # Titles and labels
    fig.text(0.07, 0.95, "Vertex count per shape", fontsize=15,
             fontweight="bold", color=TEXT_PRIMARY, ha="left")
    fig.text(0.07, 0.905,
             f"{len(counts):,} shapes  ·  range {lo:,} – {hi:,} vertices"
             + ("  ·  log-scaled x-axis" if LOG_X else ""),
             fontsize=10, color=TEXT_SECONDARY, ha="left")
    ax.set_xlabel("Number of vertices", color=TEXT_SECONDARY, fontsize=10.5)
    ax.set_ylabel("Number of shapes", color=TEXT_SECONDARY, fontsize=10.5)

    # Recessive axes and grid
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5, length=0, which="both")
    ax.tick_params(axis="x", length=4, which="major")
    ax.set_xlim(bins[0], bins[-1])

    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.11)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved plot to {Path(out_path).resolve()}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    output_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE
    counts = load_vertex_counts(input_file, SHEET_NAME, VERTEX_COLUMN)
    plot(counts, output_file)