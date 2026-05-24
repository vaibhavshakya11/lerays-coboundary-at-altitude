"""
statistical.py: statistical frontend (Ring 5: learned-from-data invariants).

For systems where closed-form invariants are not available (legacy code,
table-driven controllers, learned controllers, simulation black-boxes),
this frontend infers invariants from observed clean execution traces.

Strategy: collect N clean-state samples X = [x_1, ..., x_N] of shape
(N, d). Fit a Gaussian model x ~ N(mu, Sigma). Any clean sample lies
within a Mahalanobis ellipsoid around mu with high probability. We
encode this as a set of LINEAR invariants on the centred state:

    v_j^T (x - mu) = 0  for each j

where {v_j} are the directions of low variance in Sigma (the null /
small-eigenvalue space, the directions in which the clean data is
constant or nearly constant).

A clean sample produces |v_j^T (x - mu)| ~ sqrt(lambda_j) which is
small.  A faulted sample produces large violations on directions
orthogonal to the data manifold.

This is the right frontend when:
  * the system is too complex to extract algebraic invariants from
  * we have many clean execution traces (logs, replays)
  * we want approximate but useful protection

The "clean residual" is non-zero by construction (it equals the
typical projection scale on each direction); the framework reports
this so the user can set the detection threshold accordingly.
"""

import numpy as np
import networkx as nx
from typing import List, Optional
from .frontend_interface import Frontend, SheafSpec


class StatisticalFrontend(Frontend):
    name = "statistical"

    def extract(self,
                clean_samples: np.ndarray,
                n_invariants:   Optional[int] = None,
                variance_threshold: float = 1e-3) -> SheafSpec:
        """Fit a Gaussian to clean samples and extract low-variance directions.

        Args:
            clean_samples:  shape (N, d) array of N clean-state vectors.
            n_invariants:   how many low-variance directions to use as
                            invariants. If None, use all directions with
                            normalised variance below variance_threshold.
            variance_threshold: cutoff (relative to max eigenvalue) below
                            which a direction is considered "essentially
                            zero variance" and used as an invariant.
        """
        if clean_samples.ndim != 2:
            raise ValueError("clean_samples must be 2D (N, d)")
        N, d = clean_samples.shape

        mu = clean_samples.mean(axis=0)
        Xc = clean_samples - mu
        # Eigendecompose covariance (use SVD for numerical stability)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # S contains singular values; eigenvalues of covariance are S^2/(N-1)
        eigs = (S ** 2) / max(1, N - 1)
        max_eig = eigs.max()

        # Pick low-variance directions: sort ascending, take those below
        # variance_threshold * max_eig (or take n_invariants smallest).
        order = np.argsort(eigs)            # smallest first
        if n_invariants is None:
            mask = eigs[order] < variance_threshold * max_eig
            chosen = order[mask]
        else:
            chosen = order[:n_invariants]
        if len(chosen) == 0:
            raise ValueError(
                "No low-variance directions found; consider raising "
                "variance_threshold or supplying samples that lie on a "
                "lower-dimensional manifold."
            )

        # Each chosen v_j gives one invariant: v_j^T (x - mu) = 0
        n_inv = len(chosen)
        H = np.zeros((n_inv, d))
        b = np.zeros(n_inv)
        descs: List[str] = []
        for r, j in enumerate(chosen):
            v = Vt[j]                         # right-singular vector
            H[r, :] = v
            b[r]    = float(v @ mu)
            descs.append(
                f"low-var direction {j}: eigenvalue {eigs[j]:.3e} "
                f"({100*eigs[j]/max_eig:.2f}% of max)"
            )

        # Graph: trivial single-vertex graph (the state is one "block").
        G = nx.Graph()
        G.add_node(0)

        # Empirical clean residual: averaged over the supplied samples.
        residuals = []
        for x_s in clean_samples:
            residuals.append(float(np.linalg.norm(H @ x_s - b)))
        clean_resid_mean = float(np.mean(residuals))
        clean_resid_p99  = float(np.percentile(residuals, 99))

        return SheafSpec(
            graph = G,
            k_v   = d,
            k_e   = n_inv,
            H     = H,
            residual_norm_clean = clean_resid_mean,
            invariant_descriptions = descs,
            metadata = {
                "frontend":             "statistical",
                "d_state":              d,
                "n_samples":            N,
                "n_invariants":         n_inv,
                "mu":                   mu,
                "eigenvalues":          eigs,
                "b_offset":             b,
                "clean_resid_mean":     clean_resid_mean,
                "clean_resid_p99":      clean_resid_p99,
                "detection_threshold":  3 * clean_resid_p99,  # 3-sigma rule of thumb
                "chosen_directions":    chosen.tolist(),
            },
        )


if __name__ == "__main__":
    # smoke test: state lives on a 2D plane in R^5 with small noise
    rng = np.random.default_rng(0)
    # Generate clean samples on the plane spanned by e_0, e_1 + small noise on e_2..e_4
    basis = rng.standard_normal((5, 2))
    coeffs = rng.standard_normal((1000, 2))
    clean = coeffs @ basis.T + rng.standard_normal((1000, 5)) * 0.01
    fe = StatisticalFrontend()
    spec = fe.extract(clean_samples=clean, variance_threshold=0.01)
    print(f"H shape:              {spec.H.shape}")
    print(f"# invariants:         {spec.metadata['n_invariants']}")
    print(f"eigenvalues (sorted): {sorted(spec.metadata['eigenvalues'])[:5]}")
    print(f"mean clean residual:  {spec.metadata['clean_resid_mean']:.3e}")
    print(f"p99 clean residual:   {spec.metadata['clean_resid_p99']:.3e}")
    print(f"detection threshold:  {spec.metadata['detection_threshold']:.3e}")
    # Inject a fault and check the syndrome jumps
    x_clean = clean[0]
    x_fault = x_clean.copy()
    x_fault[2] += 5.0    # large perturbation on a low-var direction
    print(f"\nclean syndrome:   {np.linalg.norm(spec.H @ x_clean - spec.metadata['b_offset']):.3e}")
    print(f"faulted syndrome: {np.linalg.norm(spec.H @ x_fault - spec.metadata['b_offset']):.3e}")
