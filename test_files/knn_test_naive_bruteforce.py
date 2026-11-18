import time
import numpy as np
from scipy.spatial.distance import cdist

from pathlib import Path
import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from mesh_browser.knn import ShapeIndex

idx = ShapeIndex.from_file()
X, N, k = idx.features, len(idx.names), 7   # k+1, the shape itself is in the database
N_QUERIES = 2000

rng = np.random.default_rng(0)
queries = X[rng.choice(N, size=min(N_QUERIES, N), replace=False)]
Q = len(queries)


# --- the three ways to answer one query ---------------------------------------------
def tree_query(q):
    return idx.tree.query(q[None, :], k=k)

def numpy_brute(q):
    d = np.linalg.norm(X - q, axis=1)
    nearest = np.argpartition(d, k)[:k]
    return nearest[np.argsort(d[nearest])]

def naive_brute(q):
    d = [np.sqrt(((x - q) ** 2).sum()) for x in X]
    return sorted(range(N), key=d.__getitem__)[:k]


def avg_ms(method):
    for q in queries[:20]:          # warm-up (caches, lazy initialization)
        method(q)
    t = time.perf_counter()
    for q in queries:
        method(q)
    return (time.perf_counter() - t) / Q * 1e3


# --- 1) how much work does the tree skip? --------------------------------------------
idx.tree.reset_n_calls()
idx.tree.query(queries, k=k)
calls = idx.tree.get_n_calls()
print(f"{Q} queries, database of {N} shapes, {X.shape[1]} dimensions")
print(f"distance evals per query: {calls / Q:.0f} of {N} ({calls / (Q * N):.0%} of brute force)\n")

# --- 2) one query at a time (how the retrieval system is actually used) --------------
t_tree  = avg_ms(tree_query)
t_numpy = avg_ms(numpy_brute)
t_naive = avg_ms(naive_brute)

print("Average time per query, one query at a time:")
print(f"  k-d tree            {t_tree:7.3f} ms")
print(f"  numpy brute force   {t_numpy:7.3f} ms   (tree speedup {t_numpy / t_tree:5.1f}x)")
print(f"  naive python loop   {t_naive:7.3f} ms   (tree speedup {t_naive / t_tree:5.1f}x)")

# --- 3) all queries in one batched call ----------------------------------------------
t = time.perf_counter(); idx.tree.query(queries, k=k); t_tree_b = time.perf_counter() - t
t = time.perf_counter()
D = cdist(queries, X); np.argpartition(D, k, axis=1)[:, :k]
t_bf_b = time.perf_counter() - t

print("\nAverage time per query, all queries batched:")
print(f"  k-d tree            {t_tree_b / Q * 1e3:7.3f} ms")
print(f"  cdist brute force   {t_bf_b / Q * 1e3:7.3f} ms   (tree speedup {t_bf_b / t_tree_b:5.1f}x)")

# --- 4) sanity check: the tree and brute force find the same neighbors ----------------
dist_tree, _ = idx.tree.query(queries, k=k)
dist_bf = np.sort(D, axis=1)[:, :k]
print(f"\ntree results identical to brute force: {np.allclose(dist_tree, dist_bf)}")