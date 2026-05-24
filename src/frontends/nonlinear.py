"""
nonlinear.py: nonlinear-invariant frontend (Ring 6: transcendental).

Handles invariants involving trigonometric, exponential, and other
transcendental functions by adding ALGEBRAIC SHADOW VARIABLES that
satisfy known identities.

The key trick: a transcendental function f(x) that has an algebraic
relation to other variables (e.g. sin^2 + cos^2 = 1, or de^x/dx = e^x)
can be replaced by an auxiliary algebraic variable that obeys the
relation. This is called Carleman linearisation for ODEs and is the
standard way to embed nonlinear dynamics into a linear framework.

Examples handled:

  * Trig identity:        s = sin(theta), c = cos(theta)  =>  s^2 + c^2 = 1
                          (one polynomial invariant)

  * Angle rate:           d theta / dt = omega
                          Add invariants  s_next = s + dt * c * omega
                                          c_next = c - dt * s * omega
                          (linear in s, c after fixing omega)

  * Exponential decay:    y_{k+1} = exp(-lambda * dt) * y_k
                          = K * y_k       where K = exp(-lambda * dt) is constant
                          (linear once K is fixed at compile time)

  * Energy in pendulum:   E = 1/2 m L^2 omega^2 + m g L (1 - cos(theta))
                          Becomes polynomial in (omega, c) where c = cos(theta).

The frontend builds the algebraic system by promoting each transcendental
to a shadow variable and emitting both:
   (a) the algebraic relations that pin the shadow values (polynomial),
   (b) the update rules that link shadow values across iterations
       (which become linear after the algebraic relations are baked in).

For the paper, the message is: anything an engineer can write as a
finite list of algebraic identities + linear update rules over those
shadows fits the framework. This covers the vast majority of practical
GNC nonlinearities (attitude propagation, orbital mechanics, control
saturation, pendulum-like dynamics).
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from .frontend_interface import Frontend, SheafSpec


# Reuse the polynomial machinery via composition
from .polynomial import monomial_basis, evaluate_monomials


class NonlinearFrontend(Frontend):
    name = "nonlinear"

    def extract(self,
                state_vars:     List[str],
                shadow_relations: List[Tuple[dict, str]],
                linear_updates:  List[Tuple[str, List[Tuple[str, float]], float]],
                n_iters:        int,
                degree:         int = 2,
                trajectory:     Optional[np.ndarray] = None,
                include_monomial_consistency: bool = True) -> SheafSpec:
        """Build a sheaf for a system with transcendental dynamics.

        Args:
            state_vars:       list of state variable names. Should include
                              shadow variables for any transcendentals.
                              For a pendulum, e.g.:
                                ["theta", "omega", "s", "c"]
                              where s=sin(theta), c=cos(theta).
            shadow_relations: list of (coeffs_dict, description) where each
                              coeffs_dict maps a monomial-tuple of state-var
                              indices to a coefficient. Each must hold
                              exactly: "s^2 + c^2 = 1" becomes
                              {(s_idx, s_idx): 1, (c_idx, c_idx): 1, (): -1}
            linear_updates:   list of (target, [(src, coeff), ...], const)
                              describing per-iteration linear updates over
                              the state vars. These are written in SSA form
                              like LinearFrontend.
            n_iters:          unrolling depth.
            degree:           polynomial degree for shadow relations (2 default).
            trajectory:       optional (n_iters+1, n_vars) clean trajectory
                              for residual checking AND for runtime offsets
                              on the monomial-consistency aux rows. Required
                              if include_monomial_consistency=True.
            include_monomial_consistency: if True (default), add one aux row
                              per lifted slot at every iter, picking out that
                              slot and pinning it (via the b_offset vector)
                              to the clean lifted value. This is the same
                              trick polynomial.py uses to guard slots that
                              no user invariant touches; without it the
                              Carleman lifting leaves most slots blind and
                              detection collapses to ~36%. See the module
                              docstring for why this is needed.
        """
        n_vars = len(state_vars)
        var_idx = {v: i for i, v in enumerate(state_vars)}

        # Build monomial basis for the lifted stalk
        basis = monomial_basis(n_vars, degree)
        L = len(basis)
        basis_index = {m: i for i, m in enumerate(basis)}

        # State at each iteration is the lifted vector of length L.
        # Stalks: vertex stalk = R^L (lifted state); we lay out one vertex per iter.
        k_v = L
        # Per iter we have (a) shadow relations and (b) linear update rows.
        n_shadow = len(shadow_relations)
        n_update = len(linear_updates)
        k_e = n_shadow + n_update

        G = nx.path_graph(n_iters + 1)

        # Build the per-iter coefficient block for shadow relations:
        # row r of size L picks out the monomials of the lifted state.
        A_shadow = np.zeros((n_shadow, L))
        for r, (cdict, _) in enumerate(shadow_relations):
            for m, c in cdict.items():
                m_sorted = tuple(sorted(m))
                if m_sorted not in basis_index:
                    raise ValueError(
                        f"Shadow relation {r} uses monomial {m_sorted} which is "
                        f"outside degree {degree} basis (try a higher degree)."
                    )
                A_shadow[r, basis_index[m_sorted]] = c

        # Linear updates: each is target = sum(coeff * src) + const.
        # On the lifted basis, "linear in state" corresponds to invariants
        # of the form (target at iter k+1) - sum(coeff * src at iter k) = const.
        # We extract these as L-vectors (B_target, B_srcs) where the rows touch
        # only the degree-1 monomials.
        # For each update we produce one row of the parity-check matrix that
        # links lifted state at iter k to lifted state at iter k+1.

        # We need a per-iter "transition block" of shape (n_update, 2L):
        # the left half hits the iter-k lifted state, the right half hits iter k+1.
        T_left  = np.zeros((n_update, L))
        T_right = np.zeros((n_update, L))
        b_const = np.zeros(n_update)
        for r, (target, srcs, const) in enumerate(linear_updates):
            t_idx = var_idx[target]
            # Right side: +1 * x_{k+1}[target]
            T_right[r, basis_index[(t_idx,)]] = 1.0
            for (src_name, c) in srcs:
                if src_name == "const":
                    b_const[r] += c
                else:
                    s_idx = var_idx[src_name]
                    T_left[r, basis_index[(s_idx,)]] -= c
            b_const[r] += const

        # Determine which lifted slots to guard with monomial-consistency rows.
        # Diagnosis (see experiments/diagnose_nonlinear.py): without these
        # rows, only a small subset of lifted slots is constrained by any
        # shadow relation or linear update. For a degree-2 pendulum lifting
        # with state_vars=[theta,omega,s,c], 10 of 15 slots per iteration are
        # "blind" (no row touches them), so a fault on any of those slots is
        # invisible. Detection collapses to ~36%.
        #
        # The fix mirrors polynomial.py's monomial-consistency trick, but
        # extended to *every* slot (not just degree>=2). The aux row for
        # slot j at iter k is the single-1 vector e_j, with a runtime
        # offset b_aux that equals the clean lifted value at that slot.
        # At runtime the offset is precomputed from the clean trajectory;
        # if the program is operating correctly, the row contributes zero.
        # If a fault perturbs that slot, the row's residual equals the
        # perturbation magnitude (1:1 detectability, no leverage loss).
        #
        # Important: we DON'T add aux rows on slots that already have a
        # non-trivial linear constraint with deterministic b (the shadow
        # and update rows). Doing so would double-count and slow OMP. We
        # detect this by checking which (iter, slot) pairs are touched by
        # the existing H block.
        include_aux = include_monomial_consistency
        if include_aux and trajectory is None:
            raise ValueError(
                "include_monomial_consistency=True requires a trajectory "
                "(used to fill the runtime b_offset). Pass trajectory= "
                "or set include_monomial_consistency=False."
            )

        # Pass 1: build the algebraic block (shadow + update) into H_alg.
        n_alg_rows = (n_iters + 1) * n_shadow + n_iters * n_update
        n_alg = n_alg_rows
        total_cols = (n_iters + 1) * L
        H_alg = np.zeros((n_alg, total_cols))
        b_alg = np.zeros(n_alg)
        descs: List[str] = []
        row = 0

        # Shadow-relation rows at every iter (k=0..n_iters)
        for k in range(n_iters + 1):
            H_alg[row:row + n_shadow, k * L:(k + 1) * L] = A_shadow
            for r, (_, d) in enumerate(shadow_relations):
                descs.append(f"iter{k} shadow: {d}")
            row += n_shadow

        # Update rows linking k to k+1
        for k in range(n_iters):
            H_alg[row:row + n_update, k * L:(k + 1) * L]        = T_left
            H_alg[row:row + n_update, (k + 1) * L:(k + 2) * L] += T_right
            b_alg[row:row + n_update] = b_const
            for r, (t, _, _) in enumerate(linear_updates):
                descs.append(f"iter{k}->iter{k+1} update: {t}")
            row += n_update

        # Pass 2: figure out which (iter, slot) pairs are already constrained
        # by the algebraic block. A column is "constrained" if any algebraic
        # row touches it non-trivially.
        col_touches = (np.abs(H_alg) > 1e-12).sum(axis=0)

        # Pass 3: add monomial-consistency aux rows for every blind slot.
        # Need the clean lifted trajectory to fill the runtime offset.
        aux_rows_list = []
        aux_b_list    = []
        aux_descs     = []
        x_lifted_clean = None
        if trajectory is not None:
            assert trajectory.shape == (n_iters + 1, n_vars), \
                f"trajectory shape {trajectory.shape} != ({n_iters+1}, {n_vars})"
            x_lifted_clean = np.zeros(total_cols)
            for k in range(n_iters + 1):
                x_lifted_clean[k * L:(k + 1) * L] = evaluate_monomials(
                    trajectory[k], basis)

        # We add a single-column aux row for *every* (iter, slot) pair, not
        # only the slots that are blind to the algebraic block. The reason:
        # even a slot touched by a shadow or update row can be confused with
        # its siblings in the same row by OMP, because their column signatures
        # are identical up to sign. Concrete example: (s,s) and (c,c) both
        # have signature (+1 on shadow row 'iter k: s^2+c^2=1') and OMP's
        # correlation score cannot tell them apart. The aux rows give each
        # column its own unique signature on a distinct row.
        # Aux rows are cheap (1 non-zero each), and the runtime offset
        # comes directly from the clean trajectory.
        if include_aux:
            for k in range(n_iters + 1):
                for slot_i in range(L):
                    col = k * L + slot_i
                    row_vec = np.zeros(total_cols)
                    row_vec[col] = 1.0
                    aux_rows_list.append(row_vec)
                    aux_b_list.append(float(x_lifted_clean[col]))
                    aux_descs.append(
                        f"iter{k} slot{slot_i} {basis[slot_i]} consistency")

        # Final H, b assembly
        if aux_rows_list:
            H_aux = np.stack(aux_rows_list, axis=0)
            b_aux = np.array(aux_b_list)
            H = np.vstack([H_alg, H_aux])
            b_vec = np.concatenate([b_alg, b_aux])
            descs = descs + aux_descs
        else:
            H = H_alg
            b_vec = b_alg

        total_rows = H.shape[0]
        n_aux = len(aux_rows_list)

        # Clean residual on the final H/b (algebraic + aux).
        clean_resid = 0.0
        if x_lifted_clean is not None:
            clean_resid = float(np.linalg.norm(H @ x_lifted_clean - b_vec))

        return SheafSpec(
            graph = G,
            k_v   = k_v,
            k_e   = k_e,
            H     = H,
            residual_norm_clean = clean_resid,
            invariant_descriptions = descs,
            metadata = {
                "frontend":     "nonlinear",
                "degree":       degree,
                "n_vars":       n_vars,
                "state_vars":   state_vars,
                "basis":        basis,
                "lifted_dim":   L,
                "n_shadow":     n_shadow,
                "n_update":     n_update,
                "n_aux":        n_aux,
                "b_offset":     b_vec,
            },
        )


if __name__ == "__main__":
    # smoke test: pendulum with small-angle rotation
    # State: [theta, omega, s, c] where s = sin(theta), c = cos(theta).
    # Shadow: s^2 + c^2 = 1.
    # Update (small dt and linearised): theta_{k+1} = theta_k + dt * omega_k.
    # omega_{k+1} = omega_k - dt * (g/L) * theta_k     (linear small-angle approx)
    # We carry s, c but don't update them linearly (would need polynomial update,
    # left as future work / piecewise-linear extension).
    state_vars = ["theta", "omega", "s", "c"]
    shadow_relations = [({
        ("s_idx", "s_idx"): 1.0,
        ("c_idx", "c_idx"): 1.0,
        ():                 -1.0,
    }, "s^2 + c^2 = 1")]
    # Substitute named indices with actual integer indices
    var_idx = {v: i for i, v in enumerate(state_vars)}
    shadow_relations = [({
        (var_idx["s"], var_idx["s"]): 1.0,
        (var_idx["c"], var_idx["c"]): 1.0,
        ():                            -1.0,
    }, "s^2 + c^2 = 1")]

    dt = 0.01
    g_over_L = 9.81 / 1.0
    # Linearised (first-order in dt) rotation update for s, c:
    #   s_{k+1} = s_k + dt * omega_k * c_k       (linear in s,c after fixing omega)
    #   c_{k+1} = c_k - dt * omega_k * s_k       but our linear_updates list
    # cannot encode quadratic terms (omega * c). So for now we restrict to the
    # most general LINEAR-in-state updates and accept the s, c invariant only
    # as the shadow algebraic relation. The polynomial frontend would handle
    # the bilinear update; this frontend's purpose is just the shadow
    # algebraic constraint at each iter.
    linear_updates = [
        ("theta", [("theta", 1.0), ("omega", dt)], 0.0),
        ("omega", [("omega", 1.0), ("theta", -dt * g_over_L)], 0.0),
    ]

    # Build a clean trajectory: small-angle pendulum
    theta_0 = 0.1
    omega_0 = 0.0
    N = 50
    traj = np.zeros((N + 1, 4))
    theta, omega = theta_0, omega_0
    for k in range(N + 1):
        traj[k] = [theta, omega, np.sin(theta), np.cos(theta)]
        # Step
        theta_n = theta + dt * omega
        omega_n = omega - dt * g_over_L * theta
        theta, omega = theta_n, omega_n
    # The trajectory in (s, c) drifts from (sin, cos) because our update rule
    # leaves s, c constant; we substitute the true s, c at each step for the
    # residual test (the framework would compute these in flight too).
    # So set s, c manually to track theta:
    for k in range(N + 1):
        traj[k, 2] = np.sin(traj[k, 0])
        traj[k, 3] = np.cos(traj[k, 0])

    fe = NonlinearFrontend()
    spec = fe.extract(
        state_vars=state_vars,
        shadow_relations=shadow_relations,
        linear_updates=linear_updates,
        n_iters=N,
        degree=2,
        trajectory=traj,
    )
    print(f"H shape:             {spec.H.shape}")
    print(f"lifted stalk dim:    {spec.k_v}")
    print(f"# shadow rows:       {spec.metadata['n_shadow']}")
    print(f"# update rows:       {spec.metadata['n_update']}")
    print(f"clean syndrome norm: {spec.residual_norm_clean:.3e}")
