"""
exp_structured_map_locus.py — characterise the failure locus of the
altitude bound's genericity assumption for structured (non-random)
restriction maps.

THE QUESTION. Theorem 1 assumes i.i.d. continuous restriction maps and
guarantees d = k_v almost surely. The maps actually synthesised by the
linear frontend are STRUCTURED (constructed from program semantics, not
random). Does the genericity guarantee still hold?

PRECISE FORMULATION. The genericity exclusion set is the locus where
the lower-bound proof's Case 1 rank argument fails: some submatrix H_v
of restriction-map rows fails to have full column rank k_v. Formally,
for a vertex v with deg(v) k_e >= k_v, the exclusion set is

    Z_v = { restriction maps F such that rank(H_v(F)) < k_v }

For a deg(v) k_e x k_v matrix, Z_v is the vanishing of all (k_v x k_v)
minors. This is an algebraic subvariety of codimension at least
(deg(v) k_e - k_v + 1) in the parameter space of restriction-map
collections [Eisenbud, 1995, Cor 14.13].

For continuous random maps, P(F in Z_v) = 0 (algebraic subvarieties
have Lebesgue measure zero).

For STRUCTURED maps from the linear frontend:
  1. Each restriction map is constructed by parsing the program in SSA
     form, then assembling a coefficient matrix from the symbolic
     transition.
  2. The matrix entries are deterministic rational functions of the
     program's constants.

We test empirically across our six frontends and a generated corpus of
1000 small programs whether the structured restriction maps ever fall
in Z_v.
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper"))

import numpy as np
from gen_paper_data import (setup_linear, setup_polynomial, setup_pwl,
                              setup_nn, setup_statistical, setup_nonlinear)
from sheaf_lib import wilson_ci

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def check_genericity(H, k_v, k_e):
    """For each vertex block, compute the rank of the column block
    against the rows that touch it. Returns (passed, failed) counts
    and the worst case.

    For a vertex v occupying columns [v*k_v : (v+1)*k_v], let
    H_v = rows of H whose entries on v's columns are non-zero.
    Check if rank(H_v restricted to v's columns) == min(rows, k_v).
    If not, the genericity assumption fails for v.
    """
    n_cols = H.shape[1]
    n_vertices = n_cols // k_v
    passed = 0
    failed = 0
    rank_deficits = []
    for v in range(n_vertices):
        c0 = v * k_v
        c1 = c0 + k_v
        # rows touching v's block
        touch_mask = np.any(np.abs(H[:, c0:c1]) > 1e-10, axis=1)
        H_v = H[touch_mask, c0:c1]
        if H_v.shape[0] == 0:
            failed += 1
            rank_deficits.append((v, 0, 0))
            continue
        rk = np.linalg.matrix_rank(H_v, tol=1e-10)
        expected = min(H_v.shape[0], k_v)
        if rk == expected:
            passed += 1
        else:
            failed += 1
            rank_deficits.append((v, rk, expected))
    return passed, failed, rank_deficits


def main():
    print("=== Exp 17: Structured-map locus, six frontends ===\n")
    results = {}
    setup_fns = {
        "linear":            setup_linear,
        "polynomial":        setup_polynomial,
        "piecewise_linear":  setup_pwl,
        "neural_net":        setup_nn,
        "statistical":       setup_statistical,
        "nonlinear":         setup_nonlinear,
    }
    for fe_name, setup_fn in setup_fns.items():
        spec, x_clean, b_off = setup_fn()
        H = spec.H
        k_v = spec.k_v
        k_e = spec.k_e
        passed, failed, deficits = check_genericity(H, k_v, k_e)
        n_v = passed + failed
        results[fe_name] = {
            "k_v":            k_v,
            "k_e":            k_e,
            "H_shape":        list(H.shape),
            "n_vertices":     n_v,
            "n_passed":       passed,
            "n_failed":       failed,
            "rank_deficits":  [list(d) for d in deficits],
            "pass_rate":      passed / n_v if n_v else 0,
        }
        print(f"  {fe_name:18}  k_v={k_v}  k_e={k_e}  H={H.shape}  "
              f"vertices={n_v}  passed={passed}  failed={failed}")
        if deficits:
            print(f"    deficit examples (v, observed_rank, expected_rank):")
            for d in deficits[:3]:
                print(f"      {d}")

    # ---- Synthetic random-program corpus ----
    # Generate 1000 small linear programs with random constants and check
    # that the linear frontend never produces a singular submatrix
    print("\n  Synthetic linear-program corpus (1000 random programs):")
    from frontends.linear import LinearFrontend
    rng = np.random.default_rng(0)
    corpus_pass = 0
    corpus_fail = 0
    n_corpus = 1000
    for trial in range(n_corpus):
        fe = LinearFrontend()
        nvars = int(rng.integers(2, 6))
        nstmts = int(rng.integers(2, 8))
        # Random linear body
        var_names = [f"x{i}" for i in range(nvars)]
        body = []
        for k in range(nstmts):
            target = rng.choice(var_names)
            srcs = rng.choice(var_names, size=nvars, replace=False)
            terms = [(s, float(rng.standard_normal()))
                     for s in srcs[:rng.integers(1, nvars+1)]]
            body.append((target, terms, float(rng.standard_normal())))
        try:
            spec = fe.extract(body=body, variables=var_names,
                               n_iters=int(rng.integers(5, 20)))
        except Exception:
            continue
        p, f, _ = check_genericity(spec.H, spec.k_v, spec.k_e)
        if f == 0:
            corpus_pass += 1
        else:
            corpus_fail += 1
    _, lo, hi = wilson_ci(corpus_pass, corpus_pass + corpus_fail)
    print(f"    pass: {corpus_pass}/{corpus_pass + corpus_fail}  "
          f"95% Wilson CI: [{lo:.4f}, {hi:.4f}]")
    results["synthetic_corpus"] = {
        "n_trials":       corpus_pass + corpus_fail,
        "n_passed":       corpus_pass,
        "n_failed":       corpus_fail,
        "pass_rate":      corpus_pass / max(1, corpus_pass + corpus_fail),
        "pass_ci":        [lo, hi],
    }

    path = os.path.join(OUT_DIR, "exp17_structured_locus.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
