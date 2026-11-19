"""File names and the Variant description for the DR part.

Standard library only (plus config), so the GUI can import this cheaply to
check whether a saved map exists, without loading pandas, scikit-learn or
plotly.
"""

import os
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Variant:
    """Which feature-vector variant a map or score belongs to."""

    scale_blocks: bool = False  # scale the scalars and each histogram
    hellinger: bool = False     # square root of the histogram bins
    best_shares: bool = False   # use the shares saved by --optimize

    @property
    def scaled(self):
        """True if block scaling is applied (best shares imply it)."""
        return self.scale_blocks or self.best_shares

    @property
    def tag(self):
        """File name suffix, e.g. '_hel_scaled_opt' ('' for the plain one)."""
        tag = ""
        if self.hellinger:
            tag += "_hel"
        if self.scaled:
            tag += "_scaled"
        if self.best_shares:
            tag += "_opt"
        return tag

    @property
    def name(self):
        return self.tag or "plain"


def best_shares_path(hellinger=False):
    """File with the shares found by --optimize (one per histogram variant)."""
    suffix = "_hel" if hellinger else ""
    return os.path.join(config.SHEETS_DIR, f"best_shares{suffix}.json")


def feature_transform_path(variant=Variant()):
    """JSON file storing the fitted per-column multipliers and Hellinger flag."""
    return os.path.join(config.SHEETS_DIR,
                        f"feature_transform{variant.tag}.json")


def tsne_path(perplexity, variant=Variant()):
    """CSV with the 2D map (name, class, x, y, ...)."""
    return os.path.join(config.SHEETS_DIR,
                        f"tsne_2d_p{perplexity}{variant.tag}.csv")


def html_path(perplexity, variant=Variant()):
    """HTML file with the Plotly scatterplot."""
    return os.path.join(config.SHEETS_DIR,
                        f"tsne_map_p{perplexity}{variant.tag}.html")


def sweep_path(variant=Variant()):
    """CSV with the score of every (perplexity, seed) run."""
    return os.path.join(config.SHEETS_DIR, f"tsne_sweep{variant.tag}.csv")


def class_purity_path(perplexity=None, variant=Variant()):
    """Per-class purity table (input space only, or input space vs 2D map)."""
    if perplexity is None:
        return os.path.join(config.SHEETS_DIR, "class_purity.csv")
    return os.path.join(config.SHEETS_DIR,
                        f"class_purity_p{perplexity}{variant.tag}.csv")