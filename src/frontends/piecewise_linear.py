"""
piecewise_linear.py: piecewise-linear frontend (Ring 3).

Handles operations whose behaviour is linear *within regions* of state
space but changes at boundaries.  Examples:

  * Saturating arithmetic:   y = clip(x, lo, hi)
  * ReLU activation:         y = max(0, x)
  * Clamped controllers:     u = clip(K*e, -u_max, u_max)
  * Deadband / hysteresis:   nonlinearities common in actuators

Strategy: each piecewise-linear function partitions the state space into
*regions*; within each region the transition is linear.  The frontend
identifies which region the current state lies in (using a region oracle
that the user supplies) and uses the corresponding restriction map.

For protection, the parity-check matrix is built dynamically: when a fault
arrives, we re-evaluate which region the corrupted state is in, but the
INVARIANT being checked is "the next-state is consistent with the linear
map for the region the *clean* state belongs to." The region oracle is
itself protected by a small auxiliary syndrome (boundary-distance check).

This is a stratified sheaf in the formal sense; for the engineering
purpose here, we approximate it by maintaining a separate H_r for each
region and switching at runtime.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Callable, Dict, Optional
from .frontend_interface import Frontend, SheafSpec


class PiecewiseLinearFrontend(Frontend):
    name = "piecewise_linear"

    def extract(self,
                n_vars:    int,
                n_iters:   int,
                regions:   List[Tuple[Callable, np.ndarray, np.ndarray, str]],
                trajectory: Optional[np.ndarray] = None) -> SheafSpec:
        """Build a sheaf for a piecewise-linear program.

        Args:
            n_vars:    number of state variables.
            n_iters:   iterations to unroll.
            regions:   list of (oracle, A, b, description) tuples.  Each
                       region's oracle is a callable region(x) -> bool that
                       returns True iff x lies in this region. A and b
                       describe the per-region linear map x_{k+1} = A x_k + b.
                       Regions must form a partition (oracles disjoint and
                       jointly exhaustive over the operating range).
            trajectory: optional shape (n_iters+1, n_vars) ground-truth
                       clean trajectory.  If supplied, used to determine
                       which region applies at each step and to compute
                       empirical clean syndrome.  If None, region 0 is
                       assumed for every step (useful for testing).

        Returns:
            SheafSpec. The H matrix has one block per iteration; the block
            at iteration k uses the linear map of whichever region the
            ground-truth state at iter k lies in.
        """
        n_reg = len(regions)
        if trajectory is None:
            trajectory = np.zeros((n_iters + 1, n_vars))

        # Determine region assignment per iteration
        region_idx_per_iter: List[int] = []
        for k in range(n_iters):
            x_k = trajectory[k]
            assigned = None
            for r, (oracle, _, _, _) in enumerate(regions):
                if oracle(x_k):
                    assigned = r
                    break
            if assigned is None:
                # Fall back to region 0 if no oracle matched
                assigned = 0
            region_idx_per_iter.append(assigned)

        k_v = n_vars
        k_e = n_vars

        G = nx.Graph()
        G.add_nodes_from(range(n_iters + 1))
        G.add_edges_from([(k, k + 1) for k in range(n_iters)])

        n_constraints = n_iters * k_e
        n_cols        = (n_iters + 1) * k_v
        H = np.zeros((n_constraints, n_cols))
        b = np.zeros(n_constraints)
        descs: List[str] = []

        # For each iter, write x_{k+1} = A_r x_k + b_r as I x_{k+1} - A_r x_k = b_r
        for k in range(n_iters):
            r = region_idx_per_iter[k]
            _, A_r, b_r, desc_r = regions[r]
            r0 = k * n_vars
            c_k  = k * n_vars
            c_k1 = (k + 1) * n_vars
            H[r0:r0 + n_vars, c_k1:c_k1 + n_vars]  = np.eye(n_vars)
            H[r0:r0 + n_vars, c_k:c_k + n_vars]   -= A_r
            b[r0:r0 + n_vars] = b_r
            for v_i in range(n_vars):
                descs.append(f"iter{k} [reg {r}: {desc_r}] var{v_i}")

        # Compute clean residual on supplied trajectory.
        x_clean = trajectory.reshape(-1)
        resid   = float(np.linalg.norm(H @ x_clean - b))

        return SheafSpec(
            graph = G,
            k_v   = k_v,
            k_e   = k_e,
            H     = H,
            residual_norm_clean = resid,
            invariant_descriptions = descs,
            metadata = {
                "frontend":        "piecewise_linear",
                "n_regions":       n_reg,
                "n_vars":          n_vars,
                "n_iters":         n_iters,
                "region_idx_per_iter": region_idx_per_iter,
                "b_offset":        b,
                "x_clean":         x_clean,
                "regions_meta":    [(d,) for (_, _, _, d) in regions],
            },
        )


# =====================================================================
# Helpers: build regions for common piecewise-linear functions
# =====================================================================

def relu_regions(n_vars: int, weight: np.ndarray, bias: np.ndarray):
    """Build the 2^n regions for an n-input ReLU layer y_i = max(0, W_i x).

    For small n only (n <= 4); the region count explodes combinatorially.
    """
    if n_vars > 4:
        raise ValueError("relu_regions enumerates 2^n regions; use n_vars <= 4.")
    regions = []
    for mask_int in range(1 << n_vars):
        mask = np.array([(mask_int >> i) & 1 for i in range(n_vars)], dtype=bool)
        # Region: ReLU active for variables where mask[i]=True
        A = np.where(mask[:, None],
                     weight,
                     np.zeros_like(weight))
        b = np.where(mask, bias, np.zeros_like(bias))
        # Oracle: pre-activation sign pattern matches mask
        def make_oracle(m):
            def oracle(x):
                pre = weight @ x + bias
                return np.all((pre > 0) == m)
            return oracle
        regions.append((make_oracle(mask), A, b,
                        f"relu mask {bin(mask_int)[2:]:>0{n_vars}}"))
    return regions


def saturating_regions(n_vars: int, A_linear: np.ndarray, b_linear: np.ndarray,
                        lo: float, hi: float, var_idx: int = 0):
    """Three regions for a saturating linear update on one variable:
       below saturation, in linear range, above saturation."""
    A_below = A_linear.copy(); A_below[var_idx, :] = 0.0
    b_below = b_linear.copy(); b_below[var_idx]    = lo
    A_above = A_linear.copy(); A_above[var_idx, :] = 0.0
    b_above = b_linear.copy(); b_above[var_idx]    = hi
    def lin_oracle(x):
        pred = A_linear[var_idx, :] @ x + b_linear[var_idx]
        return lo <= pred <= hi
    def below_oracle(x):
        pred = A_linear[var_idx, :] @ x + b_linear[var_idx]
        return pred < lo
    def above_oracle(x):
        pred = A_linear[var_idx, :] @ x + b_linear[var_idx]
        return pred > hi
    return [
        (lin_oracle,   A_linear, b_linear, "linear range"),
        (below_oracle, A_below,  b_below,  f"saturated at {lo}"),
        (above_oracle, A_above,  b_above,  f"saturated at {hi}"),
    ]


if __name__ == "__main__":
    # smoke test: a clamped PID controller
    # x_{k+1} = clip(0.9 * x_k + 0.1 * u_k, -10, 10)
    # We treat u_k as a constant input baked into b.
    n_vars = 1
    A_linear = np.array([[0.9]])
    u = 5.0
    b_linear = np.array([0.1 * u])

    regions = saturating_regions(n_vars=1, A_linear=A_linear, b_linear=b_linear,
                                  lo=-10.0, hi=10.0)
    # Generate clean trajectory
    x = 0.0
    traj = [x]
    for _ in range(20):
        x = max(-10.0, min(10.0, 0.9 * x + 0.5))
        traj.append(x)
    traj = np.array(traj).reshape(-1, 1)

    fe = PiecewiseLinearFrontend()
    spec = fe.extract(n_vars=1, n_iters=20, regions=regions, trajectory=traj)
    print(f"H shape:            {spec.H.shape}")
    print(f"# regions:          {spec.metadata['n_regions']}")
    print(f"region per iter:    {spec.metadata['region_idx_per_iter']}")
    print(f"clean syndrome norm:{spec.residual_norm_clean:.3e}")
