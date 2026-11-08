"""What the GUI button calls to show the t-SNE map.

Standard library only (plus config and dr_paths), so importing it in app.py
does not load pandas, scikit-learn or plotly.

The map shown is the one chosen after the sweep: perplexity and variant are
set below. The button opens the saved HTML if it exists; if only the CSV
exists, a separate process rebuilds the HTML and opens it.
"""

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from . import config
from .dr_paths import Variant, html_path, tsne_path

# The final map: perplexity 15, Hellinger histograms, optimized group shares
# (files tsne_2d_p15_hel_scaled_opt.csv / tsne_map_p15_hel_scaled_opt.html).
TSNE_PERPLEXITY = 15
TSNE_VARIANT = Variant(hellinger=True, best_shares=True)


def open_tsne_map(perplexity=TSNE_PERPLEXITY, variant=TSNE_VARIANT):
    """Show the saved t-SNE scatterplot in the browser.

    Returns a short status message for the GUI's info label. Never raises.
    """
    html = html_path(perplexity, variant)
    csv = tsne_path(perplexity, variant)

    if os.path.exists(html):
        webbrowser.open(Path(html).resolve().as_uri())
        return f"Opened t-SNE map (perplexity {perplexity}, {variant.name})."

    if os.path.exists(csv):
        # Rebuild the HTML in a separate process: plotly and scikit-learn stay
        # out of the GUI process. cwd must be the project root, because the
        # file dialog changes the working directory.
        cmd = [sys.executable, "-m", f"{__package__}.dr", str(perplexity),
               "--plot-only"]
        if variant.hellinger:
            cmd.append("--hellinger")
        if variant.scale_blocks:
            cmd.append("--scale-blocks")
        if variant.best_shares:
            cmd.append("--use-best-shares")
        try:
            subprocess.Popen(cmd, cwd=config.PROJECT_ROOT)
        except OSError as error:
            return f"Could not start the plot process: {error}"
        return "Building the t-SNE map, it opens in the browser shortly."

    return (f"No saved t-SNE map for perplexity {perplexity} "
            f"({variant.name}). Run python -m {__package__}.dr first.")