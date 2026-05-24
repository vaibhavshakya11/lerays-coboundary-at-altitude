"""
sheaf_lib.py: canonical implementation of the cellular sheaf framework.

All experiments import from this module to ensure consistency.

Functions:
    build_sheaf(G, k_v, k_e, seed): construct parity-check matrix H for sheaf on graph G
    omp_decode(H, x_obs, max_sparsity): orthogonal matching pursuit decoder
    measure_distance(H, max_weight): exhaustive minimum-distance search
    wilson_ci(k, n, alpha): Wilson score confidence interval for binomial proportion
    secded_detect(delta_int): SECDED Hamming code detection model
    tmr_recover(replicas, fault_indicators): triple-modular-redundancy majority vote
    swift_r_recover(replicas, fault_indicators): SWIFT-R per-instruction vote
    abft_matmul_detect(A, B, C): algorithm-based fault tolerance for matrix multiply
"""

import numpy as np
import networkx as nx
from scipy.linalg import null_space
from itertools import combinations
from scipy.stats import norm


# =====================================================================
# Sheaf construction
# =====================================================================

def build_sheaf(G, k_v, k_e, seed=0):
    """Construct the coboundary matrix H for a cellular sheaf on G.
    Restriction maps are i.i.d. Gaussian with given seed.

    Args:
        G: networkx Graph
        k_v: dimension of each vertex stalk
        k_e: dimension of each edge stalk
        seed: RNG seed for restriction maps

    Returns:
        H: numpy array of shape (|E| * k_e, |V| * k_v)
    """
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    n_v = G.number_of_nodes()
    H = np.zeros((len(edges) * k_e, n_v * k_v))
    for e_idx, (u, v) in enumerate(edges):
        F_u = rng.standard_normal((k_e, k_v))
        F_v = rng.standard_normal((k_e, k_v))
        rs = e_idx * k_e
        H[rs:rs + k_e, u * k_v:(u + 1) * k_v] = F_u
        H[rs:rs + k_e, v * k_v:(v + 1) * k_v] = -F_v
    return H


def build_sheaf_with_program_invariants(invariants, n_iters, k_v):
    """Construct H from a list of linear program invariants.

    Each invariant is a tuple (target_var_idx, coefficient_vector, source_iter_offset)
    representing target = sum(coeff * source_var). Builds parity-check matrix
    encoding these invariants across n_iters loop iterations.
    """
    n_constraints = n_iters * len(invariants)
    n_vars = (n_iters + 1) * k_v
    H = np.zeros((n_constraints, n_vars))
    row = 0
    for k in range(n_iters):
        for (target_idx, coeffs, _) in invariants:
            H[row, (k + 1) * k_v + target_idx] = 1.0
            for src_idx, c in enumerate(coeffs):
                if c != 0:
                    H[row, k * k_v + src_idx] -= c
            row += 1
    return H


# =====================================================================
# Decoder
# =====================================================================

def omp_decode(H, x_obs, max_sparsity=5, residual_tol=1e-8):
    """Orthogonal Matching Pursuit decoder for sparse fault recovery.

    Given observation x_obs that may contain a sparse additive fault,
    return the estimated clean codeword.

    Returns:
        x_clean: recovered codeword
        support: indices identified as fault positions
        n_iters: number of OMP iterations actually run
    """
    syndrome = H @ x_obs
    s = syndrome.copy()
    r = syndrome.copy()
    support = []
    n = H.shape[1]
    col_norms = np.linalg.norm(H, axis=0)
    col_norms_safe = np.where(col_norms > 1e-12, col_norms, 1.0)

    for it in range(max_sparsity):
        correlations = np.abs(H.T @ r) / col_norms_safe
        for i in support:
            correlations[i] = 0
        i_best = int(np.argmax(correlations))
        if correlations[i_best] < 1e-10:
            break
        support.append(i_best)
        cols = H[:, support]
        mags, _, _, _ = np.linalg.lstsq(cols, s, rcond=None)
        r = s - cols @ mags
        if np.linalg.norm(r) < residual_tol:
            break

    fault = np.zeros(n)
    if support:
        cols = H[:, support]
        mags, _, _, _ = np.linalg.lstsq(cols, s, rcond=None)
        for idx, pos in enumerate(support):
            fault[pos] = mags[idx]
    return x_obs - fault, support, len(support)


def measure_distance(H, max_weight=4):
    """Exhaustive minimum-distance search up to max_weight."""
    n = H.shape[1]
    ker = null_space(H)
    if ker.shape[1] == 0:
        return None
    for w in range(1, max_weight + 1):
        for support in combinations(range(n), w):
            cols = H[:, list(support)]
            ker_cols = null_space(cols)
            for j in range(ker_cols.shape[1]):
                c = ker_cols[:, j]
                if np.all(np.abs(c) > 1e-9):
                    return w
    return max_weight + 1


def find_min_weight_codeword_at_vertex(H, v_id, k_v):
    """For Theorem 1 verification: find the codeword supported entirely
    at vertex v_id, if one exists in the null space."""
    v_cols = list(range(v_id * k_v, (v_id + 1) * k_v))
    H_v = H[:, v_cols]
    relevant = np.nonzero(np.any(np.abs(H_v) > 1e-12, axis=1))[0]
    H_restricted = H_v[relevant, :]
    ker = null_space(H_restricted)
    if ker.shape[1] == 0:
        return None, None
    u = ker[:, 0]
    codeword = np.zeros(H.shape[1])
    codeword[v_cols] = u
    weight = int(np.sum(np.abs(codeword) > 1e-9))
    return codeword, weight


# =====================================================================
# Statistical tools
# =====================================================================

def wilson_ci(k, n, alpha=0.05):
    """Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation, especially near 0 or 1.

    Returns:
        (point_estimate, lower, upper)
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (p_hat, max(0.0, center - half), min(1.0, center + half))


def mcnemar_test(b, c):
    """McNemar's exact test for paired binary outcomes.
    b: count where method A correct and B incorrect
    c: count where method A incorrect and B correct

    Returns:
        p_value: two-sided p-value
    """
    if b + c == 0:
        return 1.0
    from scipy.stats import binom
    k = min(b, c)
    n = b + c
    # Two-sided p value
    p = 2 * binom.cdf(k, n, 0.5)
    return min(p, 1.0)


# =====================================================================
# Baseline fault-tolerance models
# =====================================================================

def hamming_weight(x):
    """Population count of a 64-bit integer."""
    return bin(int(x) & 0xFFFFFFFFFFFFFFFF).count('1')


def secded_detect(delta_int):
    """SECDED Hamming code detection model.
    Detects 1- and 2-bit errors; misses 3+ bit errors.
    """
    if delta_int == 0:
        return False
    return hamming_weight(delta_int) <= 2


def tmr_recover(replicas):
    """Triple-modular-redundancy majority vote.
    Given 3 replicas of a value, return the majority.
    Returns the value and a flag indicating whether voting succeeded
    (False means all three differ).
    """
    assert len(replicas) == 3
    if replicas[0] == replicas[1] or replicas[0] == replicas[2]:
        return replicas[0], True
    if replicas[1] == replicas[2]:
        return replicas[1], True
    return replicas[0], False


def swift_r_step(true_value, n_replicas=3, fault_prob_per_replica=0.0,
                 common_mode_prob=0.0, rng=None):
    """One SWIFT-R per-instruction voting step.
    Each of n_replicas computes the value; with probability fault_prob_per_replica
    each is independently perturbed. With probability common_mode_prob, ALL
    replicas are perturbed identically.

    Returns the voted result.
    """
    if rng is None:
        rng = np.random.default_rng()
    replicas = [true_value] * n_replicas
    # Common-mode fault
    if rng.random() < common_mode_prob:
        delta = rng.standard_normal() * 0.5
        replicas = [r + delta for r in replicas]
    # Independent faults
    for i in range(n_replicas):
        if rng.random() < fault_prob_per_replica:
            replicas[i] = replicas[i] + rng.standard_normal() * 0.5
    # Median vote (robust to one fault)
    return float(np.median(replicas))


def abft_matmul_detect(A, B, C):
    """Algorithm-based fault tolerance for matrix multiply.
    Returns (row_check_passed, col_check_passed).
    Huang-Abraham row/column checksums.
    """
    n = C.shape[0]
    row_residual = np.linalg.norm(np.ones(n) @ C - (np.ones(n) @ A) @ B)
    col_residual = np.linalg.norm(C @ np.ones(C.shape[1]) -
                                   A @ (B @ np.ones(B.shape[1])))
    return (row_residual < 1e-9, col_residual < 1e-9)


# =====================================================================
# Fault injection models
# =====================================================================

def inject_seu(value_int, rng):
    """Single-event upset: flip one random bit."""
    bit = rng.integers(0, 32)
    return value_int ^ (1 << bit)


def inject_mbu(value_int, n_bits, rng):
    """Multi-bit upset: flip n_bits adjacent bits."""
    start = rng.integers(0, max(1, 32 - n_bits))
    mask = ((1 << n_bits) - 1) << start
    return value_int ^ mask


def inject_sefi(value_int, rng):
    """Single-event functional interrupt: wholesale value substitution."""
    return int(rng.integers(1, 2**31))


def inject_sign_flip_float(value):
    """Flip the sign of a floating-point value."""
    return -value


def inject_stuck_at_zero():
    return 0.0


if __name__ == "__main__":
    # Quick smoke test
    G = nx.cycle_graph(10)
    H = build_sheaf(G, k_v=4, k_e=2, seed=0)
    print(f"H shape: {H.shape}")
    print(f"Kernel dimension: {null_space(H).shape[1]}")

    # Make a codeword and inject a fault, recover
    ker = null_space(H)
    rng = np.random.default_rng(0)
    codeword = ker @ rng.standard_normal(ker.shape[1])
    observed = codeword.copy()
    observed[5] += 1.5
    recovered, support, n = omp_decode(H, observed, max_sparsity=3)
    err = np.linalg.norm(recovered - codeword)
    print(f"OMP recovered with error {err:.2e}, support {support}")

    print(f"Wilson CI (50/100): {wilson_ci(50, 100)}")
    print(f"Wilson CI (99/100): {wilson_ci(99, 100)}")
