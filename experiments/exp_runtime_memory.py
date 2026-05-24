"""
exp_runtime_memory.py
======================
Measures wall-clock decoder runtime and memory footprint of the framework's
core data structures on this host.

DISCLAIMER (must be stated in the paper): all measurements here are on a
generic x86-64 Linux container with NumPy/SciPy and DOUBLE-PRECISION
floating point. They are intended to illustrate the *shape* of overhead --
how it scales with state dimension and what fraction of memory goes into
which data structure -- NOT as a substitute for cycle-accurate measurements
on a radiation-hardened flight processor such as the BAE RAD750. A PowerPC
RAD750 running at 200 MHz would produce different absolute numbers; the
trends (linear-in-nnz syndrome cost, quadratic-in-fault-count OMP cost,
fixed memory footprint of the sparse H) would persist.

We measure per frontend:
  - Memory footprint of the parity-check matrix H (CSR sparse and dense)
  - Memory footprint of the b_offset vector and clean-state cache
  - Wall-clock time for one syndrome computation H x - b
  - Wall-clock time for one OMP decode at sparsity k = 1
  - Wall-clock time for one OMP decode at sparsity k = 5
"""
import os, sys, json, time, platform
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
from scipy.sparse import csr_matrix
from sheaf_lib import omp_decode

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def measure_one_sheaf(name, H, b_offset, x_clean, n_iters=200):
    """Profile memory + time for one sheaf spec."""
    # Memory footprint
    n_rows, n_cols = H.shape
    nnz = int(np.count_nonzero(H))
    # Dense: rows * cols * 8 bytes (double)
    mem_dense_bytes = H.nbytes
    # CSR: nnz * (8 data + 4 indices) + (rows+1) * 4 indptr
    mem_csr_bytes = nnz * 12 + (n_rows + 1) * 4
    # b_offset: n_rows * 8 bytes
    mem_b_bytes = b_offset.nbytes if b_offset is not None else 0
    # x_clean cache: n_cols * 8 bytes
    mem_x_bytes = x_clean.nbytes
    # Total runtime overhead vs TMR: TMR is 3x the state.
    # Framework footprint: x_clean (1x state) + H sparse + b
    # TMR footprint: 3 * x.nbytes (three replicas)
    mem_tmr_bytes = 3 * x_clean.nbytes
    mem_framework_bytes = mem_csr_bytes + mem_b_bytes + mem_x_bytes
    overhead_vs_tmr_ratio = mem_framework_bytes / max(mem_tmr_bytes, 1)

    # Timing: build a fault and time syndrome + decode
    H_csr = csr_matrix(H)

    # Syndrome time: time H @ x - b across n_iters trials
    rng = np.random.default_rng(0)
    x = x_clean.copy()
    # Add a small perturbation so we exercise the actual path
    x[0] += 1e-3
    t0 = time.perf_counter()
    for _ in range(n_iters):
        syn = H_csr @ x - b_offset
    syn_t_per_call_us = (time.perf_counter() - t0) / n_iters * 1e6

    # Single-fault OMP time
    omp_t_k1_us = []
    for trial in range(20):
        fault_idx = int(rng.integers(0, n_cols))
        x = x_clean.copy()
        x[fault_idx] += 1.0 + rng.standard_normal()
        # omp_decode takes the OBSERVED state x and computes the syndrome itself
        t0 = time.perf_counter()
        _ = omp_decode(H, x - x_clean, max_sparsity=2)
        omp_t_k1_us.append((time.perf_counter() - t0) * 1e6)

    # 5-fault OMP time
    omp_t_k5_us = []
    for trial in range(20):
        x = x_clean.copy()
        for _ in range(5):
            fi = int(rng.integers(0, n_cols))
            x[fi] += 1.0 + rng.standard_normal()
        t0 = time.perf_counter()
        _ = omp_decode(H, x - x_clean, max_sparsity=10)
        omp_t_k5_us.append((time.perf_counter() - t0) * 1e6)

    return {
        "frontend":              name,
        "n_rows":                int(n_rows),
        "n_cols":                int(n_cols),
        "nnz":                   nnz,
        "density":               nnz / (n_rows * n_cols),
        "mem_H_dense_bytes":     int(mem_dense_bytes),
        "mem_H_csr_bytes":       int(mem_csr_bytes),
        "mem_b_offset_bytes":    int(mem_b_bytes),
        "mem_x_clean_bytes":     int(mem_x_bytes),
        "mem_framework_bytes":   int(mem_framework_bytes),
        "mem_tmr_bytes":         int(mem_tmr_bytes),
        "mem_ratio_framework_to_tmr": overhead_vs_tmr_ratio,
        "t_syndrome_us":         float(syn_t_per_call_us),
        "t_omp_k1_median_us":    float(np.median(omp_t_k1_us)),
        "t_omp_k1_p95_us":       float(np.percentile(omp_t_k1_us, 95)),
        "t_omp_k5_median_us":    float(np.median(omp_t_k5_us)),
        "t_omp_k5_p95_us":       float(np.percentile(omp_t_k5_us, 95)),
    }


def main():
    print("\n=== Exp 19: Runtime and memory measurements ===\n")
    print(f"  Platform: {platform.platform()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  NumPy: {np.__version__}\n")

    # Import the same setup functions used by gen_paper_data
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_paper_data import (setup_linear, setup_polynomial, setup_pwl,
                                  setup_nn, setup_statistical, setup_nonlinear)

    setups = {
        "linear":            setup_linear,
        "polynomial":        setup_polynomial,
        "piecewise_linear":  setup_pwl,
        "neural_net":        setup_nn,
        "statistical":       setup_statistical,
        "nonlinear":         setup_nonlinear,
    }

    results = []
    for name, fn in setups.items():
        spec, x_clean, b_offset = fn()
        r = measure_one_sheaf(name, spec.H, b_offset, x_clean)
        results.append(r)
        print(f"  {name:18}  "
              f"H {r['n_rows']:>4}x{r['n_cols']:<4}  "
              f"nnz={r['nnz']:>6}  "
              f"mem_csr={r['mem_H_csr_bytes']/1024:>6.1f}KB  "
              f"syn={r['t_syndrome_us']:>6.1f}us  "
              f"OMP_k1={r['t_omp_k1_median_us']:>7.1f}us  "
              f"OMP_k5={r['t_omp_k5_median_us']:>7.1f}us")

    out = {
        "description": ("Runtime and memory footprint of the framework on "
                        "the six implemented frontends. Measurements taken "
                        "on an x86-64 Linux host using double precision; "
                        "absolute timings on a radiation-hardened flight "
                        "processor will differ."),
        "platform":   platform.platform(),
        "processor":  platform.processor(),
        "numpy_version": np.__version__,
        "results":    results,
    }
    path = os.path.join(OUT_DIR, "exp19_runtime_memory.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
