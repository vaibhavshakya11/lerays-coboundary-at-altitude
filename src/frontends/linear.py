"""
linear.py: linear-invariant frontend (Ring 1).

Extracts invariants of the form x_{k+1} = A x_k + b from straight-line code
that uses only linear operations.  This is the baseline frontend that
covers Kalman prediction, PID update, FFT butterflies, matrix-multiply
ABFT, and most of classical GNC.

Programs are described as a list of basic-block transitions in
single-static-assignment (SSA) form:

    [(target_var_name, [(src_var_name, coefficient), ...], const_term), ...]

For example a 2-variable accumulator + position update:

    energy = energy + input        ->  ("energy", [("energy", 1), ("input", 1)], 0)
    pos    = pos + vel             ->  ("pos",    [("pos", 1),    ("vel", 1)],   0)
    vel    = vel - pos             ->  ("vel",    [("vel", 1),    ("pos", -1)],  0)
        -- SSA matters: this last read sees the UPDATED pos.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict
from .frontend_interface import Frontend, SheafSpec


class LinearFrontend(Frontend):
    name = "linear"

    def extract(self,
                body:           List[Tuple[str, List[Tuple[str, float]], float]],
                variables:      List[str],
                n_iters:        int,
                inputs:         Optional[np.ndarray] = None) -> SheafSpec:
        """Build a sheaf from a linear program.

        Args:
            body:      ordered list of assignments (target, [(src, coeff), ...], const).
                       SSA semantics: reads see prior writes within same iter.
            variables: list of variable names (defines stalk ordering).
                       "input" is a reserved name for the per-iteration scalar input.
            n_iters:   number of loop iterations to unroll.
            inputs:    optional length-n_iters array of input scalars (for offset).

        Returns:
            SheafSpec encoding x_{k+1} = A x_k for each iteration.
        """
        n_vars = len(variables)
        if inputs is None:
            inputs = np.zeros(n_iters)
        var_idx = {v: i for i, v in enumerate(variables)}

        # SSA-aware symbolic evaluation of one iteration.
        # sigma[v] = dict mapping each source var (incl. "const", "input") to its
        # coefficient in the expression for the current value of v.
        sigma: Dict[str, Dict[str, float]] = {
            v: {v: 1.0, "const": 0.0, "input": 0.0} for v in variables
        }

        def eval_linear(srcs, const):
            """Substitute current symbolic state into an expression."""
            out: Dict[str, float] = {"const": const, "input": 0.0}
            for v in variables:
                out[v] = 0.0
            for (src_name, coeff) in srcs:
                if src_name == "input":
                    out["input"] += coeff
                elif src_name == "const":
                    out["const"] += coeff
                else:
                    sub = sigma[src_name]
                    for k, c in sub.items():
                        out[k] = out.get(k, 0.0) + coeff * c
            return out

        for (target, srcs, const) in body:
            sigma[target] = eval_linear(srcs, const)

        # Build graph: vertex per (iter, var), edge from each iter to next.
        # Vertex stalk = R^{k_v} where k_v = n_vars (one slot per variable).
        # Edge stalk  = R^{k_e} where k_e = n_vars (one invariant per variable).
        k_v = n_vars
        k_e = n_vars

        G = nx.Graph()
        for k in range(n_iters + 1):
            G.add_node(k)
        for k in range(n_iters):
            G.add_edge(k, k + 1)

        n_constraints = n_iters * k_e
        n_cols        = (n_iters + 1) * k_v
        H = np.zeros((n_constraints, n_cols))
        b = np.zeros(n_constraints)
        descs: List[str] = []

        # One invariant per (iteration k, variable v): the assigned value of v
        # at iteration k+1 must equal sigma[v] applied to iteration-k state.
        for k in range(n_iters):
            for v_i, v in enumerate(variables):
                row = k * n_vars + v_i
                # +1 coefficient on x_{k+1}[v]
                H[row, (k + 1) * n_vars + v_i] = 1.0
                # subtract sigma[v] components from iteration k
                for src, c in sigma[v].items():
                    if src == "const" or src == "input":
                        continue
                    H[row, k * n_vars + var_idx[src]] -= c
                # offset (input contribution + const)
                b[row] = sigma[v]["const"] + sigma[v]["input"] * inputs[k]
                descs.append(f"{v}[{k+1}] = sigma({v}) applied to state[{k}]")

        # Empirical clean-state syndrome: simulate one trajectory and check.
        rng = np.random.default_rng(0)
        x_clean = np.zeros(n_cols)
        x_clean[:n_vars] = rng.standard_normal(n_vars)
        for k in range(n_iters):
            x_next = np.zeros(n_vars)
            for v_i, v in enumerate(variables):
                val = 0.0
                for src, c in sigma[v].items():
                    if src == "const":
                        val += c
                    elif src == "input":
                        val += c * inputs[k]
                    else:
                        val += c * x_clean[k * n_vars + var_idx[src]]
                x_next[v_i] = val
            x_clean[(k + 1) * n_vars:(k + 2) * n_vars] = x_next
        resid = float(np.linalg.norm(H @ x_clean - b))

        return SheafSpec(
            graph     = G,
            k_v       = k_v,
            k_e       = k_e,
            H         = H,
            residual_norm_clean = resid,
            invariant_descriptions = descs,
            metadata  = {
                "frontend":  "linear",
                "n_iters":   n_iters,
                "n_vars":    n_vars,
                "variables": variables,
                "b_offset":  b,
                "x_clean":   x_clean,
            },
        )


if __name__ == "__main__":
    # smoke test: the example from the paper
    fe = LinearFrontend()
    body = [
        ("energy", [("energy", 1), ("input", 1)], 0),
        ("pos",    [("pos", 1),    ("vel", 1)],    0),
        ("vel",    [("vel", 1),    ("pos", -1)],   0),
    ]
    spec = fe.extract(
        body=body, variables=["energy", "pos", "vel"], n_iters=15,
        inputs=np.ones(15) * 0.5,
    )
    print(f"H shape:            {spec.H.shape}")
    print(f"clean syndrome norm: {spec.residual_norm_clean:.3e}")
    print(f"# invariants:        {len(spec.invariant_descriptions)}")
