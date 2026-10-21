"""Mesh defect table (Table tab:diagnosis) before and after the repair pipeline.

For a random sample of the normalized database it counts how many meshes have each defect
    before = after merging duplicate vertices (clean_mesh), as in the original table
    after  = after the full repair_mesh pipeline, i.e. the mesh the volume is computed on
and prints a text table plus LaTeX code for the report (also saved to defects_table.tex).

Run from the project root:  python test_defects.py
Slow: the repair takes seconds per mesh and the self-intersection test is expensive, so
lower N_SAMPLES for a quick try.
"""
import os
import random

import open3d as o3d

from mesh_browser import config
import mesh_browser.mesh_utils as utils
from mesh_browser.descriptors1 import list_items, repair_mesh   # or descriptors1

N_SAMPLES = 200
SEED = 0

# (label, test that returns True when the mesh HAS the defect)
DEFECTS = [
    ("Self-intersecting", lambda m: m.is_self_intersecting()),
    ("Non-manifold edges", lambda m: not m.is_edge_manifold(allow_boundary_edges=True)),
    ("Non-orientable", lambda m: not m.is_orientable()),
    ("Non-manifold vertices", lambda m: not m.is_vertex_manifold()),
    ("Boundary edges or non-manifold edges",
     lambda m: not m.is_edge_manifold(allow_boundary_edges=False)),
]

db = config.NORMALIZED_DATABASE_DIR
items = list_items(db)
sample = random.Random(SEED).sample(items, min(N_SAMPLES, len(items)))

before = [0] * len(DEFECTS)
after = [0] * len(DEFECTS)
n = 0

for k, (folder, name) in enumerate(sample):
    mesh = utils.load_mesh(os.path.join(db, folder, name))
    if mesh is None:
        continue
    n += 1
    cleaned = utils.clean_mesh(o3d.geometry.TriangleMesh(mesh))
    repaired, _ = repair_mesh(o3d.geometry.TriangleMesh(cleaned))
    for i, (label, has_defect) in enumerate(DEFECTS):
        before[i] += int(has_defect(cleaned))
        after[i] += int(has_defect(repaired))
    print(f"\r{k + 1}/{len(sample)}", end="", flush=True)


def share(count):
    return f"{100.0 * count / n:.1f}"


print(f"\n\n{n} meshes\n")
print(f"{'Defect':40}{'before':>16}{'after':>16}")
for i, (label, _) in enumerate(DEFECTS):
    print(f"{label:40}{before[i]:>6} ({share(before[i]):>5}%){after[i]:>6} ({share(after[i]):>5}%)")

rows = []
for i, (label, _) in enumerate(DEFECTS):
    rows.append("        " + f"{label:38} & {before[i]:3d} & {share(before[i])}\\% "
                f"& {after[i]:3d} & {share(after[i])}\\% \\\\")

latex = "\n".join([
    r"\begin{table}[h!]",
    r"    \centering",
    r"    \begin{tabular}{lrrrr}",
    r"        \toprule",
    r"        & \multicolumn{2}{c}{Before repair} & \multicolumn{2}{c}{After repair} \\",
    r"        \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
    r"        Defect & Meshes & Share & Meshes & Share \\",
    r"        \midrule",
    *rows,
    r"        \bottomrule",
    r"    \end{tabular}",
    r"    \caption{Mesh defects in a random sample of " + str(n) + r" meshes (seed " + str(SEED)
    + r"), after merging duplicate vertices (before repair) and after the full repair "
    r"pipeline (after repair). A mesh can have several defects, so the counts do not sum to "
    + str(n) + r".}",
    r"    \label{tab:diagnosis_after}",
    r"\end{table}",
])

print("\nLaTeX:\n")
print(latex)
with open("defects_table.tex", "w") as f:
    f.write(latex + "\n")