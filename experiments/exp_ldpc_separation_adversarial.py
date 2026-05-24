"""
exp_ldpc_separation_adversarial.py — strongest binary LDPC competitor.

The 'all-1 columns' LDPC of exp14/exp15 misses Class C trivially because
swapping within a vertex block leaves the row sum invariant. A clever
designer might assign DIFFERENT binary patterns to different columns
within a vertex block, breaking the swap-invariance.

We test this: instead of H_ldpc = (H_sheaf != 0), we pick H_ldpc with
the same SUPPORT (i.e. non-zeros are in the same rows for each column)
but with random binary entries in each cell. This represents the
strongest binary LDPC code on the same Tanner graph.

PREDICTION (from the theorem we'll prove): even this adversarial LDPC
must miss a constant fraction of vertex-permutation faults because the
swap of two scalars within a vertex block, when both have equal XOR
parity, is invisible regardless of how the binary check is structured.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import networkx as nx
from sheaf_lib import build_sheaf, wilson_ci
from exp_ldpc_separation import (
    vertex_block_swap, sheaf_check,
    ldpc_check_xor, ldpc_check_bitplane,
    scalar_parity, to_bits,
)
from scipy.linalg import null_space

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def build_adversarial_binary_ldpc(H_sheaf, seed=0):
    """Random binary fillings of the same support pattern.
    For each non-zero in H_sheaf, place an independent Bernoulli(1/2)
    in H_ldpc. This is the most aggressive LDPC competitor with one
    binary check per row."""
    rng = np.random.default_rng(seed)
    mask = (np.abs(H_sheaf) > 1e-12).astype(np.uint8)
    H_ldpc = mask * rng.integers(0, 2, size=H_sheaf.shape, dtype=np.uint8)
    # Ensure no zero rows or columns (degenerate code)
    for r in range(H_ldpc.shape[0]):
        if H_ldpc[r].sum() == 0 and mask[r].sum() > 0:
            nz = np.where(mask[r] > 0)[0]
            H_ldpc[r, nz[0]] = 1
    return H_ldpc


def main():
    print("=== Exp 16: Adversarial binary LDPC vs sheaf, Class C ===\n")
    G = nx.cycle_graph(10)
    k_v, k_e = 4, 2
    H_sheaf = build_sheaf(G, k_v=k_v, k_e=k_e, seed=42)
    ker = null_space(H_sheaf)
    coeffs = np.random.default_rng(42).standard_normal(ker.shape[1])
    x_clean = (ker @ coeffs).astype(np.float64)
    n_state = len(x_clean)
    n_vertices = n_state // k_v

    n_trials_per_ldpc = 2000
    n_ldpc_seeds = 20   # Try 20 different random binary LDPCs

    aggregates = []
    for ldpc_seed in range(n_ldpc_seeds):
        H_ldpc = build_adversarial_binary_ldpc(H_sheaf, seed=ldpc_seed)
        sheaf_det = ldpc_xor_det = ldpc_plane_det = 0
        n_actual = 0
        rng = np.random.default_rng(1000 + ldpc_seed)
        for trial in range(n_trials_per_ldpc):
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
        aggregates.append({
            "ldpc_seed":         ldpc_seed,
            "n":                 n_actual,
            "sheaf_rate":        sheaf_det / max(n_actual, 1),
            "ldpc_xor_rate":     ldpc_xor_det / max(n_actual, 1),
            "ldpc_plane_rate":   ldpc_plane_det / max(n_actual, 1),
            "ldpc_density":      int(H_ldpc.sum()),
        })

    # Pooled statistics across all LDPC seeds
    total_n          = sum(a["n"] for a in aggregates)
    sheaf_total      = sum(int(a["sheaf_rate"]      * a["n"]) for a in aggregates)
    ldpc_xor_total   = sum(int(a["ldpc_xor_rate"]   * a["n"]) for a in aggregates)
    ldpc_plane_total = sum(int(a["ldpc_plane_rate"] * a["n"]) for a in aggregates)
    print(f"Pooled across {n_ldpc_seeds} adversarial LDPC fillings, "
          f"{total_n} fault trials:")
    print(f"  sheaf:        {sheaf_total/total_n:.4f}  "
          f"CI=[{wilson_ci(sheaf_total, total_n)[1]:.4f},"
          f"{wilson_ci(sheaf_total, total_n)[2]:.4f}]")
    print(f"  ldpc-xor:     {ldpc_xor_total/total_n:.4f}  "
          f"CI=[{wilson_ci(ldpc_xor_total, total_n)[1]:.4f},"
          f"{wilson_ci(ldpc_xor_total, total_n)[2]:.4f}]")
    print(f"  ldpc-32plane: {ldpc_plane_total/total_n:.4f}  "
          f"CI=[{wilson_ci(ldpc_plane_total, total_n)[1]:.4f},"
          f"{wilson_ci(ldpc_plane_total, total_n)[2]:.4f}]")
    print()
    print(f"Per-LDPC-seed (best-case for LDPC): ")
    best_xor   = max(a["ldpc_xor_rate"]   for a in aggregates)
    best_plane = max(a["ldpc_plane_rate"] for a in aggregates)
    worst_xor   = min(a["ldpc_xor_rate"]   for a in aggregates)
    worst_plane = min(a["ldpc_plane_rate"] for a in aggregates)
    print(f"  ldpc-xor:     min={worst_xor:.4f}  max={best_xor:.4f}  "
          f"(min sheaf={min(a['sheaf_rate'] for a in aggregates):.4f})")
    print(f"  ldpc-32plane: min={worst_plane:.4f}  max={best_plane:.4f}")

    out = {
        "description": ("Adversarial binary LDPC: random binary fillings of "
                        "the sheaf's support pattern, 20 independent seeds, "
                        "vertex-permutation fault class. Tests whether "
                        "ANY binary LDPC design on the Tanner graph can "
                        "match the sheaf."),
        "graph": "cycle-10",
        "k_v":   k_v,
        "k_e":   k_e,
        "n_ldpc_seeds":        n_ldpc_seeds,
        "n_trials_per_ldpc":   n_trials_per_ldpc,
        "per_seed":            aggregates,
        "pooled": {
            "total_n":          total_n,
            "sheaf_rate":       sheaf_total / total_n,
            "ldpc_xor_rate":    ldpc_xor_total / total_n,
            "ldpc_plane_rate":  ldpc_plane_total / total_n,
            "sheaf_ci":      list(wilson_ci(sheaf_total, total_n)[1:]),
            "ldpc_xor_ci":   list(wilson_ci(ldpc_xor_total, total_n)[1:]),
            "ldpc_plane_ci": list(wilson_ci(ldpc_plane_total, total_n)[1:]),
        },
        "best_case_ldpc": {
            "ldpc_xor_max":   best_xor,
            "ldpc_plane_max": best_plane,
        },
    }
    path = os.path.join(OUT_DIR, "exp16_adversarial_ldpc.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
