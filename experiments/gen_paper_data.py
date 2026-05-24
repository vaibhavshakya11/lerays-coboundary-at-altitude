"""
gen_paper_data.py
=================
Master data-generation script for the v5 paper.

Runs a comprehensive set of experiments and saves results to
paper/data/*.json. Each experiment is keyed for one figure or table.

Run once:   python3 paper/gen_paper_data.py
Then individual figure scripts read from paper/data/.

ALL fault counts and seeds documented; numbers are intentionally large
enough for tight Wilson 95% CIs (target half-width <= 2 pp on headline
claims, <= 5 pp on by-class breakdowns).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
from typing import Dict, List

from fault_model import FaultStream, FaultClass, FaultEvent
from sheaf_lib import (omp_decode, wilson_ci, build_sheaf, measure_distance,
                       secded_detect, tmr_recover, swift_r_step,
                       inject_seu, inject_mbu, inject_sefi, hamming_weight)
from frontends.linear            import LinearFrontend
from frontends.polynomial        import (PolynomialFrontend, evaluate_monomials,
                                          quaternion_norm_invariant, energy_invariant)
from frontends.piecewise_linear  import PiecewiseLinearFrontend, saturating_regions
from frontends.neural_net        import NeuralNetFrontend, SmallMLP
from frontends.statistical       import StatisticalFrontend
from frontends.nonlinear         import NonlinearFrontend
import networkx as nx

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# Frontend setup functions (copied from experiments/exp_matrix.py)
# =====================================================================

def setup_linear():
    fe = LinearFrontend()
    body = [
        ("x", [("x", 1.0), ("v", 1.0)],      0.0),
        ("v", [("v", 0.95)],                 0.0),
        ("e", [("e", 1.0), ("v", 1.0)],      0.0),
        ("p", [("p", 1.0)],                  0.0),
    ]
    spec = fe.extract(body=body, variables=["x", "v", "e", "p"], n_iters=30)
    return spec, spec.metadata["x_clean"], spec.metadata["b_offset"]


def setup_polynomial():
    fe = PolynomialFrontend()
    inv1 = quaternion_norm_invariant(0, 1, 2, 3)
    inv2 = energy_invariant(v_idx=4, h_idx=5, mass=1.0, g=9.81, e_total=20.0)
    rng = np.random.default_rng(0)
    state_seqs = []
    for k in range(11):
        q = rng.standard_normal(4); q /= np.linalg.norm(q)
        h = 1.0
        v = float(np.sqrt(2 * (20.0 - 9.81 * h)))
        state_seqs.append(np.array([q[0], q[1], q[2], q[3], v, h]))

    spec = fe.extract(n_vars=6, n_iters=10, invariants=[inv1, inv2],
                      degree=2, include_monomial_consistency=True)
    basis = spec.metadata["basis"]
    L = spec.metadata["lifted_dim"]
    x_clean = np.zeros(11 * L)
    for k, s in enumerate(state_seqs):
        x_clean[k * L:(k + 1) * L] = evaluate_monomials(s, basis)
    n_inv = spec.metadata["n_invariants_per_iter"]
    n_aux = spec.metadata["n_aux_per_iter"]
    rows_per_iter = n_inv + n_aux
    n_blocks = 11
    b_full = np.zeros(spec.H.shape[0])
    for k in range(n_blocks):
        this_lifted = x_clean[k * L:(k + 1) * L]
        for r in range(n_aux):
            b_full[k * rows_per_iter + n_inv + r] = this_lifted[r]
    return spec, x_clean, b_full


def setup_pwl():
    fe = PiecewiseLinearFrontend()
    A = np.array([[0.9]])
    b = np.array([0.5])
    regions = saturating_regions(n_vars=1, A_linear=A, b_linear=b, lo=-10.0, hi=10.0)
    x = 0.0
    traj = [x]
    for _ in range(30):
        x = max(-10.0, min(10.0, 0.9 * x + 0.5))
        traj.append(x)
    traj = np.array(traj).reshape(-1, 1)
    spec = fe.extract(n_vars=1, n_iters=30, regions=regions, trajectory=traj)
    return spec, spec.metadata["x_clean"], spec.metadata["b_offset"]


def setup_nn():
    fe = NeuralNetFrontend()
    mlp = SmallMLP(layer_sizes=[4, 8, 8, 2], seed=0)
    rng = np.random.default_rng(1)
    inp = rng.standard_normal(4)
    spec = fe.extract(mlp=mlp, input_sample=inp, manifold_samples=None)
    return spec, spec.metadata["x_clean"], spec.metadata["b_offset"]


def setup_statistical():
    fe = StatisticalFrontend()
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((10, 3))
    coeffs = rng.standard_normal((2000, 3))
    samples = coeffs @ basis.T + rng.standard_normal((2000, 10)) * 0.01
    spec = fe.extract(clean_samples=samples, variance_threshold=0.01)
    return spec, samples[0], spec.metadata["b_offset"]


def setup_nonlinear():
    fe = NonlinearFrontend()
    state_vars = ["theta", "omega", "s", "c"]
    var_idx = {v: i for i, v in enumerate(state_vars)}
    shadow_relations = [({
        (var_idx["s"], var_idx["s"]): 1.0,
        (var_idx["c"], var_idx["c"]): 1.0,
        ():                            -1.0,
    }, "s^2 + c^2 = 1")]
    dt = 0.01
    g_over_L = 9.81
    linear_updates = [
        ("theta", [("theta", 1.0), ("omega", dt)], 0.0),
        ("omega", [("omega", 1.0), ("theta", -dt * g_over_L)], 0.0),
    ]
    N = 30
    theta, omega = 0.1, 0.0
    traj = np.zeros((N + 1, 4))
    for k in range(N + 1):
        traj[k] = [theta, omega, np.sin(theta), np.cos(theta)]
        theta_n = theta + dt * omega
        omega_n = omega - dt * g_over_L * theta
        theta, omega = theta_n, omega_n
    for k in range(N + 1):
        traj[k, 2] = np.sin(traj[k, 0])
        traj[k, 3] = np.cos(traj[k, 0])
    spec = fe.extract(state_vars=state_vars,
                      shadow_relations=shadow_relations,
                      linear_updates=linear_updates,
                      n_iters=N, degree=2, trajectory=traj)
    basis = spec.metadata["basis"]
    L = spec.metadata["lifted_dim"]
    x_clean = np.zeros((N + 1) * L)
    for k in range(N + 1):
        x_clean[k * L:(k + 1) * L] = evaluate_monomials(traj[k], basis)
    return spec, x_clean, spec.metadata["b_offset"]


SETUP_FUNCTIONS = {
    "linear":            setup_linear,
    "polynomial":        setup_polynomial,
    "piecewise_linear":  setup_pwl,
    "neural_net":        setup_nn,
    "statistical":       setup_statistical,
    "nonlinear":         setup_nonlinear,
}


# =====================================================================
# Common evaluation harness
# =====================================================================

def evaluate_protection(spec, x_clean, fault_events, b_offset=None,
                        detect_threshold_factor=10.0,
                        max_sparsity=8, recovery_tol=1e-3,
                        n_fp_trials=1000):
    H = spec.H
    if b_offset is None:
        b_offset = spec.metadata.get("b_offset", np.zeros(H.shape[0]))
    clean_resid = float(np.linalg.norm(H @ x_clean - b_offset))
    threshold = max(detect_threshold_factor * max(clean_resid, 1e-12), 1e-6)

    results = {
        "n_events":           0,
        "n_detected":         0,
        "n_recovered":        0,
        "n_fp":               0,
        "by_class":           {fc.value: {"n": 0, "det": 0, "rec": 0}
                               for fc in FaultClass},
        "by_bitwidth":        {},
        "by_common_mode":     {True: {"n": 0, "det": 0, "rec": 0},
                               False: {"n": 0, "det": 0, "rec": 0}},
        "threshold":          threshold,
        "clean_residual":     clean_resid,
    }

    rng = np.random.default_rng(7)
    for _ in range(n_fp_trials):
        x_noisy = x_clean + rng.standard_normal(x_clean.shape) * 1e-10
        s = np.linalg.norm(H @ x_noisy - b_offset)
        if s > threshold:
            results["n_fp"] += 1
    results["n_fp_trials"] = n_fp_trials

    n_cols = H.shape[1]
    for ev in fault_events:
        var_idx = ev.var_idx % n_cols
        x_fault = x_clean.copy()
        x_fault[var_idx] += ev.magnitude

        s_norm = float(np.linalg.norm(H @ x_fault - b_offset))
        detected = s_norm > threshold

        recovered = False
        if detected:
            try:
                x_rec, supp, _ = omp_decode(H, x_fault - x_clean, max_sparsity=max_sparsity)
                rel_err = np.linalg.norm(x_rec) / max(1.0, np.linalg.norm(ev.magnitude))
                recovered = rel_err < recovery_tol
            except Exception:
                recovered = False

        results["n_events"] += 1
        if detected: results["n_detected"] += 1
        if recovered: results["n_recovered"] += 1

        cls = ev.fault_class.value
        results["by_class"][cls]["n"] += 1
        if detected:  results["by_class"][cls]["det"] += 1
        if recovered: results["by_class"][cls]["rec"] += 1

        bw_key = str(ev.bit_width)
        if bw_key not in results["by_bitwidth"]:
            results["by_bitwidth"][bw_key] = {"n": 0, "det": 0, "rec": 0}
        results["by_bitwidth"][bw_key]["n"] += 1
        if detected:  results["by_bitwidth"][bw_key]["det"] += 1
        if recovered: results["by_bitwidth"][bw_key]["rec"] += 1

        cm = ev.common_mode
        results["by_common_mode"][cm]["n"] += 1
        if detected:  results["by_common_mode"][cm]["det"] += 1
        if recovered: results["by_common_mode"][cm]["rec"] += 1

    results["by_common_mode"] = {
        ("common_mode" if k else "independent"): v
        for k, v in results["by_common_mode"].items()
    }
    return results


# =====================================================================
# Experiment 01: Frontend matrix (DEEP version, more fault streams)
# =====================================================================

def exp01_frontend_matrix():
    print("\n=== Exp 01: Frontend matrix (deep) ===")
    out: Dict = {"description": "Six frontends, fault-stream evaluation, "
                                 "10 streams per frontend, 4h window each"}
    TOTAL_SECONDS = 3600 * 4
    N_STREAMS = 10
    STORM_FRACTION = 0.5

    all_results = {}
    for fe_name, setup_fn in SETUP_FUNCTIONS.items():
        t0 = time.time()
        spec, x_clean, b_offset = setup_fn()
        n_vars = spec.H.shape[1]
        agg = {"n":0, "det":0, "rec":0, "fp":0, "fp_trials":0,
               "by_class": {fc.value: {"n":0,"det":0,"rec":0} for fc in FaultClass},
               "by_bitwidth": {},
               "by_common_mode": {"common_mode":{"n":0,"det":0,"rec":0},
                                  "independent":{"n":0,"det":0,"rec":0}}}
        for seed in range(N_STREAMS):
            stream = FaultStream(n_vars=n_vars, total_seconds=TOTAL_SECONDS,
                                 seed=seed+2000, storm_fraction=STORM_FRACTION)
            events = stream.generate()[:300]
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
        _, dl, dh = wilson_ci(agg["det"], n) if n else (0,0,0)
        _, rl, rh = wilson_ci(agg["rec"], n) if n else (0,0,0)
        _, fl, fh = wilson_ci(agg["fp"], agg["fp_trials"]) if agg["fp_trials"] else (0,0,0)
        agg["detection_rate"] = agg["det"]/n if n else 0
        agg["recovery_rate"]  = agg["rec"]/n if n else 0
        agg["fpr"]            = agg["fp"]/agg["fp_trials"] if agg["fp_trials"] else 0
        agg["detection_ci"]   = [dl, dh]
        agg["recovery_ci"]    = [rl, rh]
        agg["fpr_ci"]         = [fl, fh]
        for cls, d in agg["by_class"].items():
            if d["n"] > 0:
                d["detection_rate"] = d["det"]/d["n"]
                d["recovery_rate"]  = d["rec"]/d["n"]
                _, lo, hi = wilson_ci(d["det"], d["n"])
                d["detection_ci"]   = [lo, hi]
        agg["spec_summary"] = {
            "k_v": spec.k_v, "k_e": spec.k_e,
            "H_shape": list(spec.H.shape),
            "n_invariants": len(spec.invariant_descriptions),
            "clean_residual": float(np.linalg.norm(spec.H @ x_clean - b_offset)),
        }
        all_results[fe_name] = agg
        print(f"  {fe_name:20s} n={n:>6d}  det={agg['detection_rate']:.4f}  rec={agg['recovery_rate']:.4f}  "
              f"fpr={agg['fpr']:.4f}  ({time.time()-t0:.1f}s)")
    out["results"] = all_results
    return out


# =====================================================================
# Experiment 02: Decoder scaling on cycle-graph sheaves
# =====================================================================

def exp02_decoder_scaling():
    print("\n=== Exp 02: Decoder scaling ===")
    out = {"description": "Single-fault OMP latency and success vs cycle-graph size"}
    # null_space() dominates wall-time at large n (79s for n=2000).
    # We cap at n=1000 for the curve and use fewer sheaf seeds for the
    # larger sizes since scaling is monotone and the noise is small.
    sizes  = [10, 20, 50, 100, 200, 500, 1000]
    seeds  = [3,  3,  3,  3,   3,   2,   2]
    trials_per_size = 50
    k_v, k_e = 4, 2
    rows = []
    for idx, n in enumerate(sizes):
        latencies, successes = [], 0
        t0 = time.time()
        from scipy.linalg import null_space
        for seed in range(seeds[idx]):
            G = nx.cycle_graph(n)
            H = build_sheaf(G, k_v=k_v, k_e=k_e, seed=seed)
            ker = null_space(H)
            if ker.shape[1] == 0: continue
            rng = np.random.default_rng(seed * 100)
            for trial in range(trials_per_size):
                c = ker @ rng.standard_normal(ker.shape[1])
                col = rng.integers(0, H.shape[1])
                mag = rng.standard_normal() * 2.0
                x_obs = c.copy(); x_obs[col] += mag
                t_decode = time.time()
                x_rec, supp, _ = omp_decode(H, x_obs - c, max_sparsity=3)
                t_decode = (time.time() - t_decode) * 1000.0
                latencies.append(t_decode)
                if supp == [col]: successes += 1
        n_trials_total = len(latencies)
        if n_trials_total == 0: continue
        lat = np.array(latencies)
        rows.append({
            "n": n,
            "median_ms": float(np.median(lat)),
            "p25_ms":    float(np.percentile(lat, 25)),
            "p75_ms":    float(np.percentile(lat, 75)),
            "p95_ms":    float(np.percentile(lat, 95)),
            "n_trials":  n_trials_total,
            "n_success": successes,
            "success_rate": successes / n_trials_total,
            "success_ci": list(wilson_ci(successes, n_trials_total)[1:]),
        })
        print(f"  n={n:5d}  median={rows[-1]['median_ms']:7.3f}ms  "
              f"p95={rows[-1]['p95_ms']:7.3f}ms  success={rows[-1]['success_rate']:.4f}  "
              f"({time.time()-t0:.1f}s)")
    out["rows"] = rows
    return out


# =====================================================================
# Experiment 03: Multi-fault recovery on cycle-100 sheaf
# =====================================================================

def exp03_multifault():
    print("\n=== Exp 03: Multi-fault recovery ===")
    out = {"description": "OMP recovery rate vs simultaneous fault count, cycle-100 sheaf"}
    n_vertices = 100
    k_v, k_e = 4, 2
    n_sheaf_seeds = 3
    trials_per_k = 100
    ks = list(range(1, 21))
    rows = []
    # Precompute kernels once per sheaf seed
    from scipy.linalg import null_space
    sheaves = []
    for sheaf_seed in range(n_sheaf_seeds):
        G = nx.cycle_graph(n_vertices)
        H = build_sheaf(G, k_v=k_v, k_e=k_e, seed=sheaf_seed)
        ker = null_space(H)
        sheaves.append((H, ker))
    for k_faults in ks:
        per_seed = []
        for sheaf_seed, (H, ker) in enumerate(sheaves):
            if ker.shape[1] == 0:
                per_seed.append(0.0); continue
            rng = np.random.default_rng(sheaf_seed * 1000 + k_faults)
            succ = 0
            for trial in range(trials_per_k):
                c = ker @ rng.standard_normal(ker.shape[1])
                cols = rng.choice(H.shape[1], size=k_faults, replace=False)
                mags = rng.standard_normal(k_faults) * 2.0
                x_obs = c.copy(); x_obs[cols] += mags
                x_rec, supp, _ = omp_decode(H, x_obs - c, max_sparsity=k_faults*2+1)
                if set(supp) == set(cols.tolist()): succ += 1
            per_seed.append(succ / trials_per_k)
        mean_rate = float(np.mean(per_seed))
        n_total = n_sheaf_seeds * trials_per_k
        n_succ  = int(round(mean_rate * n_total))
        _, lo, hi = wilson_ci(n_succ, n_total)
        rows.append({
            "k": k_faults,
            "mean_rate": mean_rate,
            "per_seed_rates": per_seed,
            "n_trials_total": n_total,
            "ci_lo": lo, "ci_hi": hi,
        })
        print(f"  k={k_faults:2d}  rate={mean_rate:.4f}  CI=[{lo:.3f},{hi:.3f}]")
    out["rows"] = rows
    return out


# =====================================================================
# Experiment 04: Altitude bound empirical verification
# =====================================================================

def exp04_altitude_bound():
    print("\n=== Exp 04: Altitude bound verification ===")
    out = {"description": "Measured minimum distance vs theorem prediction"}
    configs = [
        # (G_builder, k_v, k_e, name)
        (lambda: nx.path_graph(5),   3, 2, "path-5 kv=3"),
        (lambda: nx.path_graph(10),  3, 2, "path-10 kv=3"),
        (lambda: nx.balanced_tree(2, 3), 3, 2, "tree-2-3 kv=3"),
        (lambda: nx.balanced_tree(3, 2), 3, 2, "tree-3-2 kv=3"),
        (lambda: nx.star_graph(5),   3, 2, "star-5 kv=3"),
        (lambda: nx.cycle_graph(4),  3, 2, "cycle-4 kv=3"),
        (lambda: nx.cycle_graph(6),  3, 2, "cycle-6 kv=3"),
        (lambda: nx.path_graph(5),   4, 2, "path-5 kv=4"),
        (lambda: nx.path_graph(5),   5, 2, "path-5 kv=5"),
        (lambda: nx.cycle_graph(6),  5, 2, "cycle-6 kv=5"),
    ]
    rows = []
    for builder, k_v, k_e, name in configs:
        G = builder()
        if G.number_of_nodes() > 0:
            min_deg = min(dict(G.degree()).values())
        else:
            min_deg = 0
        threshold = k_v / k_e
        hyp_met = min_deg * k_e < k_v
        # CORRECTED prediction (v5): when hypothesis met, d = min_deg*ke + 1
        # almost surely. The v3/v4 paper claimed d = kv, which is the correct
        # UPPER bound (a kv-weight codeword exists at v0) but is not tight:
        # weight-w codewords supported at v0 also exist for any w in
        # [min_deg*ke + 1, kv], because the (ke * min_deg)-by-w restriction
        # of Hv0 has nontrivial kernel for any w > rank(Hv0) = ke*min_deg.
        predicted_v5 = (min_deg * k_e + 1) if hyp_met else None
        predicted_v3 = k_v if hyp_met else None
        measured = []
        # Use 15 seeds for small acyclic graphs (fast), 10 for cyclics.
        n_seeds = 15 if "cycle" not in name else 10
        for sheaf_seed in range(n_seeds):
            H = build_sheaf(G, k_v=k_v, k_e=k_e, seed=sheaf_seed)
            d = measure_distance(H, max_weight=k_v + 2)
            measured.append(d if d is not None else (k_v + 3))
        rows.append({
            "name": name,
            "n_vertices": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "k_v": k_v, "k_e": k_e,
            "min_deg": min_deg,
            "hypothesis_met": bool(hyp_met),
            "predicted_v5_corrected": predicted_v5,
            "predicted_v3_paper": predicted_v3,
            "predicted": predicted_v5,  # backward-compat
            "measured_mean": float(np.mean(measured)),
            "measured_min": int(min(measured)),
            "measured_max": int(max(measured)),
            "measured_all": measured,
            "n_seeds": len(measured),
        })
        print(f"  {name:22s}  pred(v5)={predicted_v5} pred(v3)={predicted_v3}  "
              f"measured mean={rows[-1]['measured_mean']:.2f} "
              f"[min={rows[-1]['measured_min']}, max={rows[-1]['measured_max']}]")
    out["rows"] = rows
    return out


# =====================================================================
# Experiment 05: Common-mode sensitivity
# =====================================================================

def exp05_common_mode():
    print("\n=== Exp 05: Common-mode sensitivity ===")
    out = {"description": "End-to-end failure rate as common-mode fraction varies, "
                          "at fixed total per-operation fault rate of 0.001"}
    # Workload: 4-state linear program, 30 iters
    # Per-operation fault rate 0.001 (lower so most trials don't have catastrophic
    # fault counts). At ~120 ops per trial that's mean ~0.12 faults/trial.
    spec, x_clean, b_offset = setup_linear()
    H = spec.H
    n_state = H.shape[1]
    n_iters = 30
    n_vars  = 4
    n_trials = 5000
    cm_fractions = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5, 0.75, 1.0]
    rows = []
    total_fault_prob = 0.001    # per-op
    for cm in cm_fractions:
        rng = np.random.default_rng(int(cm * 10000) + 17)
        sheaf_fail, tmr_fail, swift_fail, secded_fail, none_fail = 0,0,0,0,0
        for trial in range(n_trials):
            x_obs_sheaf = x_clean.copy()
            tmr_total_fail = False
            swift_total_fail = False
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
                        n_bits = int(rng.choice([1,2,3,4,5,6,7,8],
                                                 p=[0.58,0.16,0.10,0.07,0.05,0.02,0.01,0.01]))
                        # SECDED detects 1 and 2 bit errors
                        if n_bits > 2:
                            secded_total_fail = True
                        # TMR: if common-mode, all replicas wrong, vote returns wrong value
                        # If independent, only one replica wrong, vote ignores it
                        if is_cm:
                            tmr_total_fail = True
                            swift_total_fail = True
                        # else: TMR vote recovers automatically (no flag)
            # Sheaf check (semantic, syndrome-based, detects regardless of cm)
            s = np.linalg.norm(H @ x_obs_sheaf - b_offset)
            sheaf_threshold = 1e-6
            sheaf_caught = s > sheaf_threshold
            if not sheaf_caught and corrupted:
                sheaf_fail += 1
            elif sheaf_caught:
                xr, supp, _ = omp_decode(H, x_obs_sheaf - x_clean, max_sparsity=10)
                if np.linalg.norm(xr) / max(1.0, np.linalg.norm(x_clean)) > 1e-3:
                    sheaf_fail += 1
            if tmr_total_fail:
                tmr_fail += 1
            if swift_total_fail:
                swift_fail += 1
            if secded_total_fail:
                secded_fail += 1
            if corrupted:
                none_fail += 1
        rows.append({
            "common_mode_fraction": cm,
            "n_trials": n_trials,
            "sheaf_failure_rate": sheaf_fail / n_trials,
            "tmr_failure_rate":   tmr_fail / n_trials,
            "swift_failure_rate": swift_fail / n_trials,
            "secded_failure_rate": secded_fail / n_trials,
            "none_failure_rate":  none_fail / n_trials,
            "sheaf_ci":  list(wilson_ci(sheaf_fail, n_trials)[1:]),
            "tmr_ci":    list(wilson_ci(tmr_fail, n_trials)[1:]),
            "swift_ci":  list(wilson_ci(swift_fail, n_trials)[1:]),
            "secded_ci": list(wilson_ci(secded_fail, n_trials)[1:]),
            "none_ci":   list(wilson_ci(none_fail, n_trials)[1:]),
        })
        print(f"  cm={cm:.3f}  sheaf={rows[-1]['sheaf_failure_rate']:.4f}  "
              f"tmr={rows[-1]['tmr_failure_rate']:.4f}  "
              f"secded={rows[-1]['secded_failure_rate']:.4f}  "
              f"none={rows[-1]['none_failure_rate']:.4f}")
    out["rows"] = rows
    return out


# =====================================================================
# Experiment 06: Bit-width sensitivity (sheaf only)
# =====================================================================

def exp06_bitwidth():
    print("\n=== Exp 06: Bit-width sensitivity ===")
    out = {"description": "Sheaf detection vs bit-width of fault, per frontend"}
    rows = []
    for fe_name, setup_fn in SETUP_FUNCTIONS.items():
        spec, x_clean, b_offset = setup_fn()
        H = spec.H
        clean_resid = float(np.linalg.norm(H @ x_clean - b_offset))
        threshold = max(10.0 * max(clean_resid, 1e-12), 1e-6)
        per_bw = {bw: {"n": 0, "det": 0} for bw in range(1, 9)}
        rng = np.random.default_rng(42)
        n_trials_per_bw = 500
        for bw in range(1, 9):
            for trial in range(n_trials_per_bw):
                col = rng.integers(0, H.shape[1])
                # Magnitude proportional to bit-width (per fault_model.py)
                mag = rng.standard_normal() * (2.0 ** bw)
                xf = x_clean.copy(); xf[col] += mag
                s = np.linalg.norm(H @ xf - b_offset)
                if s > threshold:
                    per_bw[bw]["det"] += 1
                per_bw[bw]["n"] += 1
        for bw in range(1, 9):
            d = per_bw[bw]
            d["detection_rate"] = d["det"] / d["n"]
            _, lo, hi = wilson_ci(d["det"], d["n"])
            d["ci"] = [lo, hi]
        rows.append({"frontend": fe_name, "by_bitwidth": per_bw})
        print(f"  {fe_name:20s}  bw=1: {per_bw[1]['detection_rate']:.3f}  "
              f"bw=4: {per_bw[4]['detection_rate']:.3f}  "
              f"bw=8: {per_bw[8]['detection_rate']:.3f}")
    out["rows"] = rows
    return out


# =====================================================================
# Experiment 07: Aux-row ablation (this session's main contribution!)
# =====================================================================

def exp07_aux_row_ablation():
    print("\n=== Exp 07: Aux-row ablation (monomial-consistency) ===")
    out = {"description": "Detection rate WITH vs WITHOUT monomial-consistency aux rows"}
    rows = []
    # Run polynomial and nonlinear with include_monomial_consistency=True/False
    for fe_name in ["polynomial", "nonlinear"]:
        for include_aux in [False, True]:
            if fe_name == "polynomial":
                fe = PolynomialFrontend()
                inv1 = quaternion_norm_invariant(0, 1, 2, 3)
                inv2 = energy_invariant(v_idx=4, h_idx=5, mass=1.0, g=9.81, e_total=20.0)
                rng = np.random.default_rng(0)
                state_seqs = []
                for k in range(11):
                    q = rng.standard_normal(4); q /= np.linalg.norm(q)
                    state_seqs.append(np.array([q[0], q[1], q[2], q[3],
                                                float(np.sqrt(2*(20-9.81))), 1.0]))
                spec = fe.extract(n_vars=6, n_iters=10, invariants=[inv1, inv2],
                                  degree=2, include_monomial_consistency=include_aux)
                basis = spec.metadata["basis"]
                L = spec.metadata["lifted_dim"]
                x_clean = np.zeros(11 * L)
                for k, s in enumerate(state_seqs):
                    x_clean[k * L:(k + 1) * L] = evaluate_monomials(s, basis)
                n_inv = spec.metadata["n_invariants_per_iter"]
                n_aux = spec.metadata["n_aux_per_iter"]
                rows_per_iter = n_inv + n_aux
                b_full = np.zeros(spec.H.shape[0])
                if n_aux > 0:
                    for k in range(11):
                        this_lifted = x_clean[k * L:(k + 1) * L]
                        for r in range(n_aux):
                            b_full[k * rows_per_iter + n_inv + r] = this_lifted[r]
                b_offset = b_full
            else:
                fe = NonlinearFrontend()
                state_vars = ["theta", "omega", "s", "c"]
                var_idx = {v: i for i, v in enumerate(state_vars)}
                shadow_relations = [({
                    (var_idx["s"], var_idx["s"]): 1.0,
                    (var_idx["c"], var_idx["c"]): 1.0,
                    ():                            -1.0,
                }, "s^2 + c^2 = 1")]
                dt = 0.01; g_over_L = 9.81; N = 30
                linear_updates = [
                    ("theta", [("theta", 1.0), ("omega", dt)], 0.0),
                    ("omega", [("omega", 1.0), ("theta", -dt * g_over_L)], 0.0),
                ]
                theta, omega = 0.1, 0.0
                traj = np.zeros((N+1, 4))
                for k in range(N+1):
                    traj[k] = [theta, omega, np.sin(theta), np.cos(theta)]
                    theta_n = theta + dt * omega
                    omega_n = omega - dt * g_over_L * theta
                    theta, omega = theta_n, omega_n
                for k in range(N+1):
                    traj[k, 2] = np.sin(traj[k, 0])
                    traj[k, 3] = np.cos(traj[k, 0])
                spec = fe.extract(state_vars=state_vars,
                                  shadow_relations=shadow_relations,
                                  linear_updates=linear_updates,
                                  n_iters=N, degree=2, trajectory=traj,
                                  include_monomial_consistency=include_aux)
                basis = spec.metadata["basis"]
                L = spec.metadata["lifted_dim"]
                x_clean = np.zeros((N+1)*L)
                for k in range(N+1):
                    x_clean[k*L:(k+1)*L] = evaluate_monomials(traj[k], basis)
                b_offset = spec.metadata["b_offset"]
            # eval
            n_total, n_det, n_rec = 0, 0, 0
            for seed in range(10):
                stream = FaultStream(n_vars=spec.H.shape[1], total_seconds=3600*4,
                                     seed=seed+3000, storm_fraction=0.5)
                events = stream.generate()[:300]
                res = evaluate_protection(spec, x_clean, events, b_offset=b_offset,
                                           n_fp_trials=50)
                n_total += res["n_events"]
                n_det   += res["n_detected"]
                n_rec   += res["n_recovered"]
            _, dl, dh = wilson_ci(n_det, n_total)
            _, rl, rh = wilson_ci(n_rec, n_total)
            rows.append({
                "frontend": fe_name,
                "include_aux": include_aux,
                "n": n_total,
                "detection_rate": n_det / n_total,
                "recovery_rate":  n_rec / n_total,
                "detection_ci": [dl, dh],
                "recovery_ci":  [rl, rh],
                "H_shape": list(spec.H.shape),
            })
            print(f"  {fe_name:12s} aux={'YES' if include_aux else 'NO ':3s}  "
                  f"det={n_det/n_total:.4f}  rec={n_rec/n_total:.4f}  "
                  f"H={spec.H.shape}")
    out["rows"] = rows
    return out


# =====================================================================
# Experiment 08: Europa Clipper mission profile (modeled)
# =====================================================================

def exp08_mission():
    print("\n=== Exp 08: Mission scale ===")
    out = {"description": "24-month Europa Clipper profile with adaptive sheaf protection"}
    # 24-month timeline, 6 flybys at months 3, 7, 11, 14, 17, 20
    flyby_months = [3, 7, 11, 14, 17, 20]
    flyby_peak = 30.0   # x baseline particle flux
    flyby_width = 0.3   # months
    months = np.linspace(0, 24, 720)   # ~1-day resolution
    flux = np.ones_like(months)
    for fm in flyby_months:
        flux += (flyby_peak - 1.0) * np.exp(-((months - fm)/flyby_width)**2)
    # Strategies:
    #  - TMR always: cost 3.0/month
    #  - Max sheaf always: cost 1.5/month (high-coverage k_v=8 say)
    #  - Minimal always: cost 1.0/month (low-coverage k_v=3)
    #  - Adaptive: 1.0 during low flux, switch to 1.5 above threshold flux
    #
    # Coverage:
    #  - TMR: 90% during high flux (common-mode), 99% else
    #  - Max sheaf: 99% always
    #  - Minimal: 92% always
    #  - Adaptive: 92% low + 99% high
    threshold_flux = 5.0
    dt = months[1] - months[0]
    tmr_cost = 3.0 * np.ones_like(months)
    max_cost = 1.5 * np.ones_like(months)
    min_cost = 1.0 * np.ones_like(months)
    adapt_cost = np.where(flux > threshold_flux, 1.5, 1.0)
    tmr_cum = np.cumsum(tmr_cost) * dt
    max_cum = np.cumsum(max_cost) * dt
    min_cum = np.cumsum(min_cost) * dt
    adapt_cum = np.cumsum(adapt_cost) * dt
    # Coverage (faults that pass through undetected)
    cm_fraction = 0.05
    failure_rate_per_storm = 0.05
    tmr_cov = np.where(flux > threshold_flux,
                       1.0 - cm_fraction * failure_rate_per_storm * (flux/5),
                       0.99)
    max_cov = np.full_like(months, 0.99)
    min_cov = np.full_like(months, 0.92)
    adapt_cov = np.where(flux > threshold_flux, 0.99, 0.92)
    # Mean coverage weighted by flux exposure
    weights = flux
    out["months"]   = months.tolist()
    out["flux"]     = flux.tolist()
    out["tmr_cum_cost"]   = tmr_cum.tolist()
    out["max_cum_cost"]   = max_cum.tolist()
    out["min_cum_cost"]   = min_cum.tolist()
    out["adapt_cum_cost"] = adapt_cum.tolist()
    out["totals"] = {
        "tmr":   float(tmr_cum[-1]),
        "max":   float(max_cum[-1]),
        "min":   float(min_cum[-1]),
        "adapt": float(adapt_cum[-1]),
    }
    out["mean_coverage"] = {
        "tmr":   float(np.average(tmr_cov, weights=weights)),
        "max":   float(np.average(max_cov, weights=weights)),
        "min":   float(np.average(min_cov, weights=weights)),
        "adapt": float(np.average(adapt_cov, weights=weights)),
    }
    print(f"  Totals: TMR={out['totals']['tmr']:.1f}, "
          f"Max={out['totals']['max']:.1f}, "
          f"Adapt={out['totals']['adapt']:.1f}, "
          f"Min={out['totals']['min']:.1f}")
    print(f"  Coverages: TMR={out['mean_coverage']['tmr']:.4f}, "
          f"Adapt={out['mean_coverage']['adapt']:.4f}")
    out["savings_vs_tmr_pct"] = (out['totals']['tmr'] - out['totals']['adapt']) / out['totals']['tmr'] * 100
    print(f"  Adaptive saves {out['savings_vs_tmr_pct']:.1f}% vs TMR")
    return out


# =====================================================================
# Experiment 09: Blind-slot diagnostic (intuitive figure for the
# monomial-consistency section)
# =====================================================================

def exp09_blind_slots():
    print("\n=== Exp 09: Blind-slot count vs frontend ===")
    out = {"description": "Number of lifted slots unconstrained by algebraic invariants"}
    rows = []
    # Build polynomial WITHOUT aux rows and count
    fe = PolynomialFrontend()
    inv1 = quaternion_norm_invariant(0, 1, 2, 3)
    inv2 = energy_invariant(v_idx=4, h_idx=5, mass=1.0, g=9.81, e_total=20.0)
    spec_no_aux = fe.extract(n_vars=6, n_iters=10, invariants=[inv1, inv2],
                              degree=2, include_monomial_consistency=False)
    spec_with_aux = fe.extract(n_vars=6, n_iters=10, invariants=[inv1, inv2],
                                degree=2, include_monomial_consistency=True)
    touches_no  = (np.abs(spec_no_aux.H) > 1e-12).sum(axis=0)
    touches_yes = (np.abs(spec_with_aux.H) > 1e-12).sum(axis=0)
    rows.append({
        "frontend": "polynomial",
        "lifted_dim_per_iter": spec_no_aux.metadata["lifted_dim"],
        "n_cols": int(spec_no_aux.H.shape[1]),
        "blind_cols_no_aux":  int((touches_no == 0).sum()),
        "blind_cols_with_aux": int((touches_yes == 0).sum()),
        "H_shape_no_aux":  list(spec_no_aux.H.shape),
        "H_shape_with_aux": list(spec_with_aux.H.shape),
    })
    # Same for nonlinear
    fe = NonlinearFrontend()
    state_vars = ["theta", "omega", "s", "c"]
    var_idx = {v: i for i, v in enumerate(state_vars)}
    shadow_relations = [({
        (var_idx["s"], var_idx["s"]): 1.0,
        (var_idx["c"], var_idx["c"]): 1.0,
        ():                            -1.0,
    }, "s^2 + c^2 = 1")]
    dt = 0.01; g_over_L = 9.81; N = 30
    linear_updates = [
        ("theta", [("theta", 1.0), ("omega", dt)], 0.0),
        ("omega", [("omega", 1.0), ("theta", -dt * g_over_L)], 0.0),
    ]
    theta, omega = 0.1, 0.0
    traj = np.zeros((N+1, 4))
    for k in range(N+1):
        traj[k] = [theta, omega, np.sin(theta), np.cos(theta)]
        theta_n = theta + dt * omega
        omega_n = omega - dt * g_over_L * theta
        theta, omega = theta_n, omega_n
    for k in range(N+1):
        traj[k, 2] = np.sin(traj[k, 0])
        traj[k, 3] = np.cos(traj[k, 0])
    spec_no_aux = fe.extract(state_vars=state_vars,
                              shadow_relations=shadow_relations,
                              linear_updates=linear_updates,
                              n_iters=N, degree=2, trajectory=traj,
                              include_monomial_consistency=False)
    spec_with_aux = fe.extract(state_vars=state_vars,
                                shadow_relations=shadow_relations,
                                linear_updates=linear_updates,
                                n_iters=N, degree=2, trajectory=traj,
                                include_monomial_consistency=True)
    touches_no  = (np.abs(spec_no_aux.H) > 1e-12).sum(axis=0)
    touches_yes = (np.abs(spec_with_aux.H) > 1e-12).sum(axis=0)
    rows.append({
        "frontend": "nonlinear",
        "lifted_dim_per_iter": spec_no_aux.metadata["lifted_dim"],
        "n_cols": int(spec_no_aux.H.shape[1]),
        "blind_cols_no_aux":  int((touches_no == 0).sum()),
        "blind_cols_with_aux": int((touches_yes == 0).sum()),
        "H_shape_no_aux":  list(spec_no_aux.H.shape),
        "H_shape_with_aux": list(spec_with_aux.H.shape),
    })
    for r in rows:
        pct_no  = 100.0 * r["blind_cols_no_aux"] / r["n_cols"]
        pct_yes = 100.0 * r["blind_cols_with_aux"] / r["n_cols"]
        print(f"  {r['frontend']:12s}  blind no-aux: {r['blind_cols_no_aux']:>4d}/{r['n_cols']} "
              f"({pct_no:.1f}%)   with-aux: {r['blind_cols_with_aux']}/{r['n_cols']} ({pct_yes:.1f}%)")
    out["rows"] = rows
    return out


# =====================================================================
# Main driver
# =====================================================================

if __name__ == "__main__":
    t_all = time.time()
    experiments = {
        "exp01_frontend_matrix":  exp01_frontend_matrix,
        "exp02_decoder_scaling":  exp02_decoder_scaling,
        "exp03_multifault":       exp03_multifault,
        "exp04_altitude_bound":   exp04_altitude_bound,
        "exp05_common_mode":      exp05_common_mode,
        "exp06_bitwidth":         exp06_bitwidth,
        "exp07_aux_row_ablation": exp07_aux_row_ablation,
        "exp08_mission":          exp08_mission,
        "exp09_blind_slots":      exp09_blind_slots,
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
