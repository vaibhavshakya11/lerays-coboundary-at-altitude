"""
exp_composed_frontends.py
=========================
One realistic spacecraft task protected by either ONE of the two relevant
frontends or by BOTH composed.

Task: spacecraft attitude controller, state = (q_w, q_x, q_y, q_z, omega_x,
omega_y, omega_z). The quaternion is unit-norm; the body rates satisfy a
linear Kalman update omega_{k+1} = A omega_k.

Two natural invariant classes:
  LINEAR:     7 linear constraints from the Kalman state-space model
              (omega_{k+1} = A omega_k, viewed as 3 equations per step
              between consecutive states, plus the trivial q_{k+1} = q_k
              when no torque is applied).
  POLYNOMIAL: 1 polynomial invariant per iteration, ||q||^2 = 1
              (lifted to monomial basis via the polynomial frontend).

Three protection regimes:
  (A) LINEAR-ONLY:       only the linear frontend's sheaf is built.
  (B) POLYNOMIAL-ONLY:   only the quaternion-norm sheaf is built.
  (C) COMPOSED:          both sheafs stacked into a single H matrix.

Faults are injected into four scenarios:
  scenario_quat:   one bit-flip in a quaternion component
  scenario_rate:   one bit-flip in a body-rate component
  scenario_swap:   exchange q_x and q_y (preserves both linear & polynomial
                   constraints individually but violates physical correctness)
  scenario_burst:  multi-bit burst across both subspaces

We expect:
  - LINEAR-ONLY catches rate faults but is blind to quaternion-only faults
    (the Kalman model has no constraint touching q if it doesn't get torqued).
  - POLYNOMIAL-ONLY catches faults that break norm but is blind to anything
    that preserves norm and to all rate faults.
  - COMPOSED catches everything either alone catches; in particular,
    catches the swap scenario only when both constraints are present
    (the swap may preserve linear constraints AND quaternion norm).
"""
import os, sys, json, time, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import networkx as nx
from sheaf_lib import wilson_ci, omp_decode
from frontends.linear import LinearFrontend
from frontends.polynomial import PolynomialFrontend

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def to_bits(value):
    packed = struct.pack('>f', np.float32(value))
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))


def from_bits(bits):
    packed = np.packbits(bits.astype(np.uint8)).tobytes()
    return float(struct.unpack('>f', packed)[0])


def bit_flip(x, idx, bit):
    bits = to_bits(float(x[idx]))
    bits[bit] ^= 1
    out = x.copy()
    out[idx] = from_bits(bits)
    return out


def build_linear_sheaf(n_iters=8):
    """Linear sheaf: rate Kalman update + identity for quaternion (no torque).
    State per iter = (qw, qx, qy, qz, wx, wy, wz)."""
    fe = LinearFrontend()
    # State-space: q stays constant when no torque, omega slightly damped
    # body = [(target, [(src, coef), ...], const)]
    body = [
        ("qw", [("qw", 1.0)], 0.0),
        ("qx", [("qx", 1.0)], 0.0),
        ("qy", [("qy", 1.0)], 0.0),
        ("qz", [("qz", 1.0)], 0.0),
        ("wx", [("wx", 0.98)], 0.0),
        ("wy", [("wy", 0.98)], 0.0),
        ("wz", [("wz", 0.98)], 0.0),
    ]
    var_names = ["qw", "qx", "qy", "qz", "wx", "wy", "wz"]
    spec = fe.extract(body=body, variables=var_names, n_iters=n_iters)
    return spec


def build_polynomial_sheaf(n_iters=8, fixed_q=None):
    """Polynomial sheaf: ||q||^2 = qw^2 + qx^2 + qy^2 + qz^2 = 1 each iter.
    State per iter = (qw, qx, qy, qz) -- only the quaternion components.

    Uses a deterministic fixed unit quaternion across all iterations so the
    b_offset is consistent and we can produce a true clean codeword."""
    if fixed_q is None:
        rng = np.random.default_rng(43)
        fixed_q = rng.standard_normal(4); fixed_q = fixed_q / np.linalg.norm(fixed_q)
    fe = PolynomialFrontend()
    inv = ({
        (0,0): 1.0, (1,1): 1.0, (2,2): 1.0, (3,3): 1.0,
        (): -1.0,
    }, "||q||^2 = 1")

    # IMPORTANT: state_sampler must return the SAME state for all calls
    # so the b_offset (which pins each slot to its iter-k clean value)
    # is consistent with what we'll use as the clean codeword later.
    def sampler():
        return fixed_q.copy()

    spec = fe.extract(n_vars=4, n_iters=n_iters,
                       invariants=[inv], degree=2,
                       state_sampler=sampler, test_n=n_iters+1,
                       include_monomial_consistency=True)
    return spec, fixed_q


def make_clean_state_polynomial(spec_pol, fixed_q, n_iters=8):
    """Clean lifted codeword matching the polynomial sheaf's baked-in b_offset.
    We tile the same lifted vector across all iterations."""
    basis = spec_pol.metadata["basis"]
    v = np.empty(len(basis))
    for i, m in enumerate(basis):
        prod = 1.0
        for j in m: prod *= fixed_q[j]
        v[i] = prod
    n_blocks = n_iters + 1
    state = np.tile(v, n_blocks)
    return state


def make_clean_state_linear(spec_lin, n_iters=8):
    """Clean codeword for the linear sheaf, sampled in null(H - b)."""
    rng = np.random.default_rng(42)
    n_state = spec_lin.H.shape[1]
    n_vars = spec_lin.k_v  # = 7
    n_vertices = n_state // n_vars
    q = rng.standard_normal(4); q = q / np.linalg.norm(q)
    w0 = rng.standard_normal(3) * 0.1
    state = np.zeros(n_state)
    for k in range(n_vertices):
        state[k*n_vars:k*n_vars+4] = q
        state[k*n_vars+4:k*n_vars+7] = w0 * (0.98 ** k)
    return state, n_vertices



def build_composed_sheaf(spec_lin, spec_poly, n_iters=8):
    """Stack the two H matrices side-by-side with a shared state vector.
    We use the 'extended' state layout: per iteration we carry both the
    7-var linear state AND the monomial basis on the quaternion 4-tuple.

    The composed H is a block matrix:
        H_composed = [ H_lin   0     ]
                     [ 0       H_pol ]
    The composed state x = (x_lin || x_poly) lives in the direct-sum stalk.
    A fault on x_lin is caught by the linear block; a fault on x_poly by
    the polynomial block; a coordinated fault touching both must satisfy
    both blocks to escape."""
    from scipy.linalg import block_diag
    H_c = block_diag(spec_lin.H, spec_poly.H)
    # Total state dim
    n_lin = spec_lin.H.shape[1]
    n_pol = spec_poly.H.shape[1]
    return H_c, n_lin, n_pol


def syndrome_norm(H, x, b):
    """Magnitude of the syndrome H x - b. Detection iff > threshold."""
    return float(np.linalg.norm(H @ x - b))


def run_scenario(scenario, H_lin, x_lin_clean, b_lin,
                  H_pol, x_pol_clean, b_pol,
                  H_comp, x_comp_clean, b_comp,
                  n_vert_lin, n_vert_pol,
                  n_trials=500, seed=0):
    """Inject faults; record detection by each scheme."""
    rng = np.random.default_rng(seed)
    n_lin = H_lin.shape[1]
    n_pol = H_pol.shape[1]
    k_v_lin = n_lin // n_vert_lin
    k_v_pol = n_pol // n_vert_pol
    det_lin = det_pol = det_comp = 0
    n_actual = 0

    for trial in range(n_trials):
        x_lin = x_lin_clean.copy()
        x_pol = x_pol_clean.copy()

        if scenario == "quat_bitflip":
            # Single bit-flip in one quaternion component
            k = int(rng.integers(0, n_vert_lin))
            comp = int(rng.integers(0, 4))  # qw..qz
            bit = int(rng.integers(9, 32))  # avoid sign/high-exp to avoid NaN
            x_lin = bit_flip(x_lin, k*k_v_lin + comp, bit)
            # Same scalar value should also be in the polynomial sheaf
            kp = min(k, n_vert_pol - 1)
            x_pol = bit_flip(x_pol, kp * k_v_pol + 1 + comp, bit)

        elif scenario == "rate_bitflip":
            # Single bit-flip in one body-rate component
            k = int(rng.integers(0, n_vert_lin))
            comp = int(rng.integers(0, 3))  # wx..wz
            bit = int(rng.integers(9, 32))
            x_lin = bit_flip(x_lin, k*k_v_lin + 4 + comp, bit)

        elif scenario == "quat_swap":
            # Exchange qx and qy in one iteration
            k = int(rng.integers(0, n_vert_lin))
            i_x = k*k_v_lin + 1; i_y = k*k_v_lin + 2
            x_lin[i_x], x_lin[i_y] = x_lin[i_y], x_lin[i_x]
            kp = min(k, n_vert_pol - 1)
            p_x = kp * k_v_pol + 2; p_y = kp * k_v_pol + 3
            x_pol[p_x], x_pol[p_y] = x_pol[p_y], x_pol[p_x]

        elif scenario == "burst":
            # 3-bit burst affecting both q and omega in one iter
            k = int(rng.integers(0, n_vert_lin))
            for offset in [0, 4, 5]:
                bit = int(rng.integers(9, 32))
                x_lin = bit_flip(x_lin, k*k_v_lin + offset, bit)
            for comp in [0]:
                bit = int(rng.integers(9, 32))
                kp = min(k, n_vert_pol - 1)
                x_pol = bit_flip(x_pol, kp * k_v_pol + 1 + comp, bit)

        d_lin = syndrome_norm(H_lin, x_lin, b_lin)
        d_pol = syndrome_norm(H_pol, x_pol, b_pol)

        x_comp = np.concatenate([x_lin, x_pol])
        d_comp = syndrome_norm(H_comp, x_comp, b_comp)

        n_actual += 1
        threshold = 1e-6
        if d_lin > threshold:  det_lin  += 1
        if d_pol > threshold:  det_pol  += 1
        if d_comp > threshold: det_comp += 1

    return {
        "scenario": scenario,
        "n": n_actual,
        "linear_only_rate":     det_lin / max(n_actual, 1),
        "polynomial_only_rate": det_pol / max(n_actual, 1),
        "composed_rate":        det_comp / max(n_actual, 1),
        "linear_only_ci":     list(wilson_ci(det_lin,  n_actual)[1:]) if n_actual else [0,0],
        "polynomial_only_ci": list(wilson_ci(det_pol,  n_actual)[1:]) if n_actual else [0,0],
        "composed_ci":        list(wilson_ci(det_comp, n_actual)[1:]) if n_actual else [0,0],
    }


def main():
    print("\n=== Exp 18: Composed-frontend quaternion attitude task ===\n")
    n_iters = 8

    print("  Building linear sheaf...")
    spec_lin = build_linear_sheaf(n_iters)
    print(f"    H_lin shape: {spec_lin.H.shape}, k_v={spec_lin.k_v}")

    print("  Building polynomial sheaf...")
    spec_pol, fixed_q = build_polynomial_sheaf(n_iters)
    print(f"    H_pol shape: {spec_pol.H.shape}, k_v={spec_pol.k_v}")
    print(f"    Fixed quaternion: {fixed_q}")

    print("  Building composed sheaf...")
    H_comp, n_lin, n_pol = build_composed_sheaf(spec_lin, spec_pol, n_iters)
    print(f"    H_comp shape: {H_comp.shape}")

    # Clean states
    x_lin_clean, n_vert_lin = make_clean_state_linear(spec_lin, n_iters)
    b_lin = spec_lin.metadata.get('b_offset', np.zeros(spec_lin.H.shape[0]))
    print(f"    Linear residual: {np.linalg.norm(spec_lin.H @ x_lin_clean - b_lin):.3e}")
    print(f"    n_vertices_lin: {n_vert_lin}")

    x_pol_clean = make_clean_state_polynomial(spec_pol, fixed_q, n_iters)
    b_pol = spec_pol.metadata.get('b_offset', np.zeros(spec_pol.H.shape[0]))
    print(f"    Polynomial residual: {np.linalg.norm(spec_pol.H @ x_pol_clean - b_pol):.3e}")
    n_vert_pol = spec_pol.H.shape[1] // spec_pol.k_v
    print(f"    n_vertices_pol: {n_vert_pol}")

    # Composed clean state and composed b_offset (block-diagonal stack)
    x_comp_clean = np.concatenate([x_lin_clean, x_pol_clean])
    b_comp = np.concatenate([b_lin, b_pol])

    # Run all four scenarios
    scenarios = ["quat_bitflip", "rate_bitflip", "quat_swap", "burst"]
    results = []
    for sc in scenarios:
        t0 = time.time()
        r = run_scenario(sc,
                         spec_lin.H, x_lin_clean, b_lin,
                         spec_pol.H, x_pol_clean, b_pol,
                         H_comp, x_comp_clean, b_comp,
                         n_vert_lin, n_vert_pol, n_trials=1000,
                         seed=hash(sc) % (1<<20))
        elapsed = time.time() - t0
        print(f"  {sc:18}  n={r['n']}  "
              f"lin-only={r['linear_only_rate']:.4f}  "
              f"poly-only={r['polynomial_only_rate']:.4f}  "
              f"composed={r['composed_rate']:.4f}  "
              f"({elapsed:.1f}s)")
        results.append(r)

    out = {
        "description": ("Spacecraft attitude controller protected by linear, "
                        "polynomial, or composed sheaf; four fault scenarios."),
        "n_iters":       n_iters,
        "H_lin_shape":   list(spec_lin.H.shape),
        "H_pol_shape":   list(spec_pol.H.shape),
        "H_comp_shape":  list(H_comp.shape),
        "n_trials_per_scenario": 1000,
        "results":       results,
    }
    path = os.path.join(OUT_DIR, "exp18_composed_frontends.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
