"""
gen_rigor_data.py
=================
Reviewer-flagged rigor improvements for v5.1.

Three additions:
  (1) exp01b_statistical_deep: re-run statistical frontend with a state
      dimension that yields enough events for a tight Wilson CI
      (target: half-width < 2pp on the headline detection claim).
  (2) exp05b_cm_sensitivity:  common-mode sensitivity sweep at multiple
      total fault rates, to address "what if heavy-ion common-mode is
      actually 20%, not 5%". This produces a 2D surface (rate x cm fraction).
  (3) exp10_storm_sweep:      sweep storm_fraction from 0.1 to 0.9 to
      show that detection holds under unusually bursty fault regimes.

These supplement (don't replace) exp01..exp08 which remain valid.

Run:  python3 paper/gen_rigor_data.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper"))

import numpy as np
from typing import Dict, List

from fault_model import FaultStream, FaultClass, FaultEvent
from sheaf_lib import (omp_decode, wilson_ci, tmr_recover)
from frontends.statistical import StatisticalFrontend
from frontends.linear       import LinearFrontend

# Reuse evaluation harness and setup functions from gen_paper_data
from gen_paper_data import (evaluate_protection, setup_linear, setup_polynomial,
                              setup_pwl, setup_nn, setup_nonlinear)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# Exp 01b: Statistical frontend, deep n
#
# Reviewer note: n=39 is pilot-sized. We:
#   (a) raise state dim from 10 to 40 (closer to deployment scale)
#   (b) run 60 fault streams (vs 10) over 4h windows each
# Target: n_events >= 1000 so 95% Wilson half-width < 1.5pp at p~=0.97.
# =====================================================================

def setup_statistical_deep(state_dim=40, n_manifold_samples=5000, manifold_dim=8):
    """Higher-dimensional state living on an 8-dim manifold in R^40.
    The PCA frontend should recover the 32 low-variance directions as
    invariants and use them to detect faults."""
    fe = StatisticalFrontend()
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((state_dim, manifold_dim))
    coeffs = rng.standard_normal((n_manifold_samples, manifold_dim))
    samples = coeffs @ basis.T + rng.standard_normal((n_manifold_samples, state_dim)) * 0.01
    spec = fe.extract(clean_samples=samples, variance_threshold=0.01)
    return spec, samples[0], spec.metadata["b_offset"]


def exp01b_statistical_deep():
    print("\n=== Exp 01b: Statistical frontend, deep n ===")
    out = {"description": "Statistical frontend at production state dim, "
                          "60 streams x 4h windows, n_events > 1000 target"}
    TOTAL_SECONDS = 3600 * 4
    N_STREAMS = 60
    STORM_FRACTION = 0.5

    spec, x_clean, b_offset = setup_statistical_deep(state_dim=40, manifold_dim=8)
    n_vars = spec.H.shape[1]
    print(f"  H shape: {spec.H.shape}, n_invariants: {spec.H.shape[0]}")
    print(f"  state dim: {n_vars}, clean residual: "
          f"{float(np.linalg.norm(spec.H @ x_clean - b_offset)):.3e}")
    agg = {"n":0, "det":0, "rec":0, "fp":0, "fp_trials":0,
           "by_class": {fc.value: {"n":0,"det":0,"rec":0} for fc in FaultClass},
           "by_bitwidth": {},
           "by_common_mode": {"common_mode":{"n":0,"det":0,"rec":0},
                              "independent":{"n":0,"det":0,"rec":0}}}
    t0 = time.time()
    for seed in range(N_STREAMS):
        stream = FaultStream(n_vars=n_vars, total_seconds=TOTAL_SECONDS,
                             seed=seed + 20000, storm_fraction=STORM_FRACTION)
        events = stream.generate()[:500]
        if not events:
            continue
        res = evaluate_protection(spec, x_clean, events, b_offset=b_offset,
                                  n_fp_trials=100)
        for k in ("n_events","n_detected","n_recovered","n_fp"):
            agg[{"n_events":"n","n_detected":"det","n_recovered":"rec",
                 "n_fp":"fp"}[k]] += res[k]
        agg["fp_trials"] += res["n_fp_trials"]
        for cls in agg["by_class"]:
            for kk in ("n","det","rec"):
                agg["by_class"][cls][kk] += res["by_class"][cls][kk]
        for bw, v in res["by_bitwidth"].items():
            if bw not in agg["by_bitwidth"]:
                agg["by_bitwidth"][bw] = {"n":0,"det":0,"rec":0}
            for kk in ("n","det","rec"):
                agg["by_bitwidth"][bw][kk] += v[kk]
        for cm in agg["by_common_mode"]:
            for kk in ("n","det","rec"):
                agg["by_common_mode"][cm][kk] += res["by_common_mode"][cm][kk]
    n = agg["n"]
    if n == 0:
        print("  ERROR: no events generated, aborting")
        return out
    _, dl, dh = wilson_ci(agg["det"], n)
    _, rl, rh = wilson_ci(agg["rec"], n)
    _, fl, fh = wilson_ci(agg["fp"], agg["fp_trials"])
    agg["detection_rate"] = agg["det"]/n
    agg["recovery_rate"]  = agg["rec"]/n
    agg["fpr"]            = agg["fp"]/agg["fp_trials"] if agg["fp_trials"] else 0
    agg["detection_ci"]   = [dl, dh]
    agg["recovery_ci"]    = [rl, rh]
    agg["fpr_ci"]         = [fl, fh]
    for cls, dd in agg["by_class"].items():
        if dd["n"] > 0:
            dd["detection_rate"] = dd["det"]/dd["n"]
            dd["recovery_rate"]  = dd["rec"]/dd["n"]
            _, lo, hi = wilson_ci(dd["det"], dd["n"])
            dd["detection_ci"]   = [lo, hi]
    agg["spec_summary"] = {
        "k_v":            spec.k_v,
        "k_e":            spec.k_e,
        "H_shape":        list(spec.H.shape),
        "n_invariants":   spec.H.shape[0],
        "state_dim":      n_vars,
        "manifold_dim":   8,
        "clean_residual": float(np.linalg.norm(spec.H @ x_clean - b_offset)),
    }
    half_width = (dh - dl) / 2
    print(f"  n_events={n}, det={agg['detection_rate']:.4f} "
          f"CI=[{dl:.4f},{dh:.4f}] half-width={half_width*100:.2f}pp  "
          f"({time.time()-t0:.1f}s)")
    out["results"] = agg
    return out


# =====================================================================
# Exp 05b: Common-mode sensitivity at multiple total fault rates
#
# Reviewer note: 5% common-mode is HPC, not spacecraft. What if it's 20%?
# We sweep cm_fraction at three total per-op fault rates: 5e-4, 1e-3, 2e-3
# (a 4x range bracketing the Quinn et al estimate).
# =====================================================================

def exp05b_cm_sensitivity():
    print("\n=== Exp 05b: Common-mode sensitivity at varied fault rates ===")
    out = {"description": "Failure rate vs common-mode fraction at three "
                          "per-op fault rates spanning a 4x range, to "
                          "address sensitivity to the Quinn et al. estimate"}
    spec, x_clean, b_offset = setup_linear()
    H = spec.H
    n_state = H.shape[1]
    n_iters = 30
    n_vars  = 4
    n_trials = 3000
    fault_rates = [5e-4, 1e-3, 2e-3]
    cm_fractions = [0.0, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]
    grids = {}
    for total_fault_prob in fault_rates:
        rows = []
        for cm in cm_fractions:
            rng = np.random.default_rng(int(cm * 10000) + int(total_fault_prob * 1e6) + 17)
            sheaf_fail, tmr_fail, secded_fail, none_fail = 0, 0, 0, 0
            for trial in range(n_trials):
                x_obs_sheaf = x_clean.copy()
                tmr_total_fail = False
                secded_total_fail = False
                corrupted = False
                for it in range(n_iters):
                    for v in range(n_vars):
                        if rng.random() < total_fault_prob:
                            corrupted = True
                            is_cm = rng.random() < cm
                            magnitude = rng.standard_normal() * 1.0
                            col = it * n_vars + v
                            if col < n_state:
                                x_obs_sheaf[col] += magnitude
                            n_bits = int(rng.choice(
                                [1,2,3,4,5,6,7,8],
                                p=[0.58,0.16,0.10,0.07,0.05,0.02,0.01,0.01]))
                            if n_bits > 2:
                                secded_total_fail = True
                            if is_cm:
                                tmr_total_fail = True
                s = np.linalg.norm(H @ x_obs_sheaf - b_offset)
                sheaf_caught = s > 1e-6
                if not sheaf_caught and corrupted:
                    sheaf_fail += 1
                elif sheaf_caught:
                    xr, supp, _ = omp_decode(H, x_obs_sheaf - x_clean, max_sparsity=10)
                    if np.linalg.norm(xr) / max(1.0, np.linalg.norm(x_clean)) > 1e-3:
                        sheaf_fail += 1
                if tmr_total_fail:
                    tmr_fail += 1
                if secded_total_fail:
                    secded_fail += 1
                if corrupted:
                    none_fail += 1
            rows.append({
                "common_mode_fraction": cm,
                "n_trials": n_trials,
                "sheaf_failure_rate": sheaf_fail / n_trials,
                "tmr_failure_rate":   tmr_fail / n_trials,
                "secded_failure_rate": secded_fail / n_trials,
                "none_failure_rate":  none_fail / n_trials,
                "sheaf_ci":  list(wilson_ci(sheaf_fail, n_trials)[1:]),
                "tmr_ci":    list(wilson_ci(tmr_fail, n_trials)[1:]),
                "secded_ci": list(wilson_ci(secded_fail, n_trials)[1:]),
                "none_ci":   list(wilson_ci(none_fail, n_trials)[1:]),
            })
        grids[str(total_fault_prob)] = rows
        print(f"  rate={total_fault_prob:.4f}  cm=0: tmr={rows[0]['tmr_failure_rate']:.4f}  "
              f"cm=0.20: tmr={rows[4]['tmr_failure_rate']:.4f}  "
              f"cm=1.0: tmr={rows[-1]['tmr_failure_rate']:.4f}  "
              f"sheaf flat at {rows[0]['sheaf_failure_rate']:.4f}")
    out["grids"] = grids
    out["fault_rates"] = fault_rates
    return out


# =====================================================================
# Exp 10: Storm-fraction sensitivity
#
# Sweep storm_fraction from 0.1 to 0.9; show that the framework's
# detection holds under unusually bursty fault regimes (storm fraction
# near 1.0 means almost continuous heavy-ion storm).
# =====================================================================

def exp10_storm_sweep():
    print("\n=== Exp 10: Storm-fraction sensitivity ===")
    out = {"description": "Detection vs storm fraction for the linear frontend, "
                          "to verify framework holds under bursty regimes"}
    storm_fracs = [0.1, 0.25, 0.5, 0.75, 0.9]
    spec, x_clean, b_offset = setup_linear()
    n_vars = spec.H.shape[1]
    TOTAL_SECONDS = 3600 * 4
    N_STREAMS = 10
    rows = []
    for sf in storm_fracs:
        agg = {"n": 0, "det": 0, "rec": 0}
        for seed in range(N_STREAMS):
            stream = FaultStream(n_vars=n_vars, total_seconds=TOTAL_SECONDS,
                                 seed=seed + 5000, storm_fraction=sf)
            events = stream.generate()[:500]
            if not events:
                continue
            res = evaluate_protection(spec, x_clean, events, b_offset=b_offset,
                                      n_fp_trials=20)
            agg["n"]   += res["n_events"]
            agg["det"] += res["n_detected"]
            agg["rec"] += res["n_recovered"]
        n = max(agg["n"], 1)
        det = agg["det"]/n
        rec = agg["rec"]/n
        _, dl, dh = wilson_ci(agg["det"], n) if n > 0 else (0,0,0)
        _, rl, rh = wilson_ci(agg["rec"], n) if n > 0 else (0,0,0)
        rows.append({
            "storm_fraction": sf,
            "n_events":       n,
            "detection_rate": det,
            "recovery_rate":  rec,
            "detection_ci":   [dl, dh],
            "recovery_ci":    [rl, rh],
        })
        print(f"  storm={sf:.2f}  n={n}  det={det:.4f}  CI=[{dl:.4f},{dh:.4f}]")
    out["rows"] = rows
    return out


if __name__ == "__main__":
    t_all = time.time()
    experiments = {
        "exp01b_statistical_deep": exp01b_statistical_deep,
        "exp05b_cm_sensitivity":   exp05b_cm_sensitivity,
        "exp10_storm_sweep":       exp10_storm_sweep,
    }
    for name, fn in experiments.items():
        try:
            out = fn()
            path = os.path.join(OUT_DIR, f"{name}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"  -> {path}")
        except Exception as e:
            import traceback
            print(f"  EXCEPTION in {name}: {e}")
            traceback.print_exc()
    print(f"\nTotal time: {time.time() - t_all:.1f}s")
