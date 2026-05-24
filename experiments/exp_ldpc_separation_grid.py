"""
exp_ldpc_separation_grid.py — replicate the Class-C separation result
across multiple (graph, k_v, k_e) configurations to show it's
structural, not a one-off cycle-10 artifact.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import networkx as nx
from sheaf_lib import build_sheaf, wilson_ci
from exp_ldpc_separation import (
    build_binary_ldpc_on_same_tanner, vertex_block_swap,
    sheaf_check, ldpc_check_xor, ldpc_check_bitplane,
)
from scipy.linalg import null_space

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def run_config(G_builder, k_v, k_e, name, n_trials=1000):
    G = G_builder()
    H_sheaf = build_sheaf(G, k_v=k_v, k_e=k_e, seed=42)
    H_ldpc = build_binary_ldpc_on_same_tanner(H_sheaf)
    ker = null_space(H_sheaf)
    if ker.shape[1] == 0:
        return None
    coeffs = np.random.default_rng(42).standard_normal(ker.shape[1])
    x_clean = (ker @ coeffs).astype(np.float64)
    if np.linalg.norm(H_sheaf @ x_clean) > 1e-9:
        return None  # bad codeword
    rng = np.random.default_rng(hash(name) % 100000)
    n_state = len(x_clean)
    n_vertices = n_state // k_v
    sheaf_det, ldpc_xor_det, ldpc_plane_det, n_actual = 0, 0, 0, 0
    for trial in range(n_trials):
        vertex_id = rng.integers(0, n_vertices)
        start = vertex_id * k_v
        blk = x_clean[start:start + k_v]
        if len(np.unique(blk)) < 2:
            continue
        x_obs = vertex_block_swap(x_clean, vertex_id, k_v, rng)
        if np.allclose(x_obs, x_clean, atol=0):
            continue
        n_actual += 1
        if sheaf_check(H_sheaf, x_obs, x_clean):
            sheaf_det += 1
        if ldpc_check_xor(H_ldpc, x_obs, x_clean):
            ldpc_xor_det += 1
        if ldpc_check_bitplane(H_ldpc, x_obs, x_clean):
            ldpc_plane_det += 1
    return {
        "name":    name,
        "k_v":     k_v,
        "k_e":     k_e,
        "n_vertices": n_vertices,
        "n_trials":   n_actual,
        "sheaf_rate":      sheaf_det / max(n_actual, 1),
        "ldpc_xor_rate":   ldpc_xor_det / max(n_actual, 1),
        "ldpc_plane_rate": ldpc_plane_det / max(n_actual, 1),
        "sheaf_ci":      list(wilson_ci(sheaf_det, n_actual)[1:]) if n_actual else [0,0],
        "ldpc_xor_ci":   list(wilson_ci(ldpc_xor_det, n_actual)[1:]) if n_actual else [0,0],
        "ldpc_plane_ci": list(wilson_ci(ldpc_plane_det, n_actual)[1:]) if n_actual else [0,0],
    }


def main():
    print("=== Exp 15: LDPC separation across configurations ===\n")
    configs = [
        (lambda: nx.cycle_graph(6),   4, 2, "cycle-6 kv=4"),
        (lambda: nx.cycle_graph(10),  4, 2, "cycle-10 kv=4"),
        (lambda: nx.cycle_graph(20),  4, 2, "cycle-20 kv=4"),
        (lambda: nx.cycle_graph(10),  3, 2, "cycle-10 kv=3"),
        (lambda: nx.cycle_graph(10),  5, 2, "cycle-10 kv=5"),
        (lambda: nx.cycle_graph(10),  4, 3, "cycle-10 kv=4 ke=3"),
        (lambda: nx.complete_graph(5), 4, 2, "complete-5 kv=4"),
        (lambda: nx.path_graph(10),   4, 2, "path-10 kv=4"),
    ]
    rows = []
    for builder, kv, ke, name in configs:
        t0 = time.time()
        r = run_config(builder, kv, ke, name, n_trials=1000)
        if r is None:
            print(f"  {name}: SKIPPED (empty kernel or bad codeword)")
            continue
        rows.append(r)
        print(f"  {name}: n={r['n_trials']}  "
              f"sheaf={r['sheaf_rate']:.4f}  "
              f"ldpc-xor={r['ldpc_xor_rate']:.4f}  "
              f"ldpc-32plane={r['ldpc_plane_rate']:.4f}  "
              f"({time.time()-t0:.1f}s)")
    out = {
        "description": ("LDPC separation (vertex-block-swap fault) across "
                        "multiple (graph, k_v, k_e) configurations"),
        "rows": rows,
    }
    path = os.path.join(OUT_DIR, "exp15_ldpc_separation_grid.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
