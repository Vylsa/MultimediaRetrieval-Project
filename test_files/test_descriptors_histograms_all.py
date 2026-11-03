"""Plot the A3, D1, D2, D3, D4 histograms of ALL shapes in one class, to see whether
shapes of the same class have similar distributions.

Every shape is drawn as a thin, semi-transparent line; the class average is drawn on
top as a thick line. Optionally a second class is drawn in another color for contrast.

Set class_name below and run the file:
    python show_class_histograms.py
"""
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser.mesh_utils import load_mesh, list_meshes, list_subfolders
from mesh_browser.descriptors import (DISTRIBUTION_NAMES, DISTRIBUTION_RANGES, N_BINS,
                                      N_SAMPLES, SAMPLE_SOURCE, distribution_descriptors)
from mesh_browser.config import NORMALIZED_DATABASE_DIR

# --- Settings: edit these ---------------------------------------------------------

class_name = "Chess"        # class (folder) to plot; upper/lower case doesn't matter
compare_class = None        # e.g. "Chess" to draw a second class in another color, or None
max_shapes = None           # e.g. 30 to limit the number of shapes per class, None for all

save_path = None            # e.g. "sword_histograms.png" to also save the figure
n_bins = N_BINS
sample_source = SAMPLE_SOURCE

# ------------------------------------------------------------------------------------

CLASS_COLORS = ["#2a78d6", "#eb6834"]   # first class blue, comparison class orange

TITLES = {
    "A3": "A3: angle of 3 random points",
    "D1": "D1: barycenter to random point",
    "D2": "D2: distance between 2 points",
    "D3": "D3: √ area of triangle",
    "D4": "D4: ∛ volume of tetrahedron",
}
X_LABELS = {"A3": "angle (rad)", "D1": "distance", "D2": "distance",
            "D3": "√ area", "D4": "∛ volume"}


def style_axes(ax):
    """Recessive axes: light grid behind the data, no top/right spines."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8b7b0")
    ax.tick_params(colors="#55554f", labelsize=8)


def find_class_folder(name):
    """The folder in the normalized database matching name, ignoring upper/lower case."""
    for folder in list_subfolders(NORMALIZED_DATABASE_DIR):
        if folder.lower() == name.lower():
            return folder
    return None


def class_histograms(name):
    """(folder name, list of histogram dicts) for every shape in the class."""
    folder = find_class_folder(name)
    if folder is None:
        print(f"Class '{name}' not found in {NORMALIZED_DATABASE_DIR}")
        return None, []
    folder_path = os.path.join(NORMALIZED_DATABASE_DIR, folder)
    files = list_meshes(folder_path)
    if max_shapes is not None:
        files = files[:max_shapes]

    hists = []
    for i, file in enumerate(files, start=1):
        mesh = load_mesh(os.path.join(folder_path, file))
        if mesh is not None:
            hists.append(distribution_descriptors(mesh, n_bins=n_bins, source=sample_source))
        print(f"\r{folder}: {i}/{len(files)}", end="", flush=True)
    print()
    return folder, hists


def spread(hists, name):
    """Average L1 distance of the shapes' histograms to the class average: 0 means all
    shapes in the class have exactly the same histogram, 2 is the maximum possible."""
    matrix = np.array([h[name] for h in hists])
    return float(np.abs(matrix - matrix.mean(axis=0)).sum(axis=1).mean())


def plot_classes(classes):
    """classes: list of (folder name, list of histogram dicts)."""
    fig, axes = plt.subplots(1, len(DISTRIBUTION_NAMES), figsize=(4 * len(DISTRIBUTION_NAMES), 3.8),
                             sharey=True, constrained_layout=True)
    alpha = 0.25 if max(len(h) for _, h in classes) < 30 else 0.12

    for ax, name in zip(axes, DISTRIBUTION_NAMES):
        lo, hi = DISTRIBUTION_RANGES[name]
        edges = np.linspace(lo, hi, n_bins + 1)
        for (folder, hists), color in zip(classes, CLASS_COLORS):
            for h in hists:                                   # every shape: thin, transparent
                ax.stairs(h[name], edges, color=color, linewidth=1, alpha=alpha)
            mean = np.mean([h[name] for h in hists], axis=0)  # class average: thick
            ax.stairs(mean, edges, color=color, linewidth=2.5,
                      label=f"{folder} average ({len(hists)} shapes)")
        ax.set_title(TITLES[name], fontsize=10, color="#22221f", loc="left")
        ax.set_xlabel(X_LABELS[name], fontsize=9, color="#55554f")
        ax.set_xlim(lo, hi)
        style_axes(ax)

    axes[0].set_ylabel("fraction of samples", fontsize=9, color="#55554f")
    axes[0].set_ylim(bottom=0)
    title = " vs ".join(folder for folder, _ in classes)
    fig.suptitle(f"Shape distributions per class: {title}", fontsize=12, x=0.01, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels),
               frameon=False, fontsize=9)
    info = f"{N_SAMPLES:,} samples per descriptor, {n_bins} bins, sampled from {sample_source}"
    fig.text(0.99, 0.995, info, ha="right", va="top", fontsize=8, color="#7a7a73")
    return fig


def main():
    classes = []
    for name in [class_name] + ([compare_class] if compare_class else []):
        folder, hists = class_histograms(name)
        if hists:
            classes.append((folder, hists))
    if not classes:
        print("No shapes to show. Check class_name at the top of this file.")
        return

    # How similar are the shapes within each class? (lower = more similar)
    print("\nAverage L1 distance to the class average (0 = identical, 2 = maximum):")
    print("  " + " ".join(f"{name:>6}" for name in DISTRIBUTION_NAMES) + "   class")
    for folder, hists in classes:
        print("  " + " ".join(f"{spread(hists, name):6.3f}" for name in DISTRIBUTION_NAMES)
              + f"   {folder}")

    fig = plot_classes(classes)
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()