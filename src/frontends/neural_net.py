"""
neural_net.py: neural-network frontend (Ring 4: statistical/learned).

Protects the inference path of a small feed-forward neural network.

State vector layout for an L-layer MLP with sizes [n_0, n_1, ..., n_L]:

    x = [a_0, z_1, a_1, z_2, a_2, ..., z_L, a_L]

where a_l = activation output of layer l (a_0 = input) and z_l = pre-activation
of layer l. We carry both because:

  * z_l = W_l a_{l-1} + b_l   is LINEAR in (a_{l-1}, z_l)  -- one row per neuron
  * a_l = relu(z_l)             is PIECEWISE LINEAR; conditioned on the sign
                                pattern of z_l, it becomes linear:
                                  a_l[j] =  z_l[j]   if z_l[j] > 0
                                  a_l[j] =  0        otherwise
                                The sign pattern is fixed by the clean run.

For the output layer (no activation) we still record both z_L and a_L = z_L
for uniformity.

We also support an OPTIONAL statistical guard: learned mean/principal-component
ellipsoid constraints on each layer's pre-activation, fit from clean training
samples.  These rows are inequalities approximated as equalities at the
sample mean projection; they're soft guards, not exact invariants, so they
contribute a small clean residual (reported separately in metadata).
"""

import numpy as np
import networkx as nx
from typing import List, Optional
from .frontend_interface import Frontend, SheafSpec


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


class SmallMLP:
    """Untrained MLP for protection demos.  All layers Linear+ReLU except last
    which is Linear (typical regression head)."""
    def __init__(self, layer_sizes: List[int], seed: int = 0):
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            W = rng.standard_normal((layer_sizes[i + 1], fan_in)) / np.sqrt(fan_in)
            b = rng.standard_normal(layer_sizes[i + 1]) * 0.1
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: np.ndarray):
        """Return (pre[0..L-1], act[0..L]) lists."""
        pre = []
        act = [x.copy()]
        for L, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ act[-1] + b
            pre.append(z)
            if L < len(self.weights) - 1:
                act.append(_relu(z))
            else:
                act.append(z)  # output layer: identity activation
        return pre, act


class NeuralNetFrontend(Frontend):
    name = "neural_net"

    def extract(self,
                mlp:           SmallMLP,
                input_sample:  np.ndarray,
                manifold_samples: Optional[np.ndarray] = None) -> SheafSpec:
        """Build a sheaf for one inference pass.

        Args:
            mlp:           SmallMLP instance.
            input_sample:  shape (n_0,) input we are protecting.
            manifold_samples: optional shape (N, n_0) clean samples for
                fitting statistical guards on pre-activations.
        """
        sizes = mlp.layer_sizes              # [n_0, n_1, ..., n_L]
        L     = len(sizes) - 1               # number of layers

        # State layout: [a_0, z_1, a_1, z_2, a_2, ..., z_L, a_L]
        # Number of blocks = 1 + 2L. Sizes = [n_0, n_1, n_1, n_2, n_2, ..., n_L, n_L].
        block_sizes = [sizes[0]]
        for l in range(1, L + 1):
            block_sizes.append(sizes[l])    # z_l
            block_sizes.append(sizes[l])    # a_l
        offsets = np.cumsum([0] + block_sizes)
        total_dim = offsets[-1]

        def slot_a(l):
            # index range for a_l in state vector
            if l == 0:
                return offsets[0], offsets[1]
            # a_l is the (2l+1)th block (0-indexed: a_0=block0, z_1=1, a_1=2, ..., a_l=2l)
            return offsets[2 * l], offsets[2 * l + 1]

        def slot_z(l):
            # z_l is at block (2l - 1)
            return offsets[2 * l - 1], offsets[2 * l]

        # Compute clean trajectory to determine ReLU sign patterns
        pre, act = mlp.forward(input_sample)
        sign_patterns = [pre[l - 1] > 0 for l in range(1, L + 1)]  # for layers 1..L
        # Build clean state vector
        x_clean = np.zeros(total_dim)
        a0_lo, a0_hi = slot_a(0); x_clean[a0_lo:a0_hi] = act[0]
        for l in range(1, L + 1):
            zlo, zhi = slot_z(l); x_clean[zlo:zhi] = pre[l - 1]
            alo, ahi = slot_a(l); x_clean[alo:ahi] = act[l]

        # Build invariants:
        #   (i) Linearity rows: for each layer l, one row per neuron j:
        #       z_l[j] - W_l[j,:] a_{l-1}[:] = b_l[j]
        #   (ii) ReLU rows: for each hidden layer l (l < L), one row per
        #        neuron j enforcing a_l[j] = z_l[j] if sign[l][j] else a_l[j] = 0
        #   (iii) Output identity rows: for last layer, a_L[j] = z_L[j].
        n_linearity = sum(sizes[l] for l in range(1, L + 1))      # one per neuron per layer
        n_relu      = sum(sizes[l] for l in range(1, L))           # hidden layers only
        n_identity  = sizes[L]                                     # output layer
        n_stat      = (L if manifold_samples is not None else 0)

        n_rows = n_linearity + n_relu + n_identity + n_stat
        H = np.zeros((n_rows, total_dim))
        b = np.zeros(n_rows)
        descs: List[str] = []
        row = 0

        # (i) linearity rows
        for l in range(1, L + 1):
            W = mlp.weights[l - 1]
            bias = mlp.biases[l - 1]
            alo_prev, ahi_prev = slot_a(l - 1)
            zlo, zhi = slot_z(l)
            for j in range(sizes[l]):
                H[row, zlo + j]               = 1.0
                H[row, alo_prev:ahi_prev]    -= W[j, :]
                b[row]                        = bias[j]
                descs.append(f"layer {l} neuron {j} linearity")
                row += 1

        # (ii) ReLU rows
        for l in range(1, L):
            zlo, zhi = slot_z(l)
            alo, ahi = slot_a(l)
            for j in range(sizes[l]):
                if sign_patterns[l - 1][j]:
                    # a_l[j] = z_l[j]   ==>   a_l[j] - z_l[j] = 0
                    H[row, alo + j] = 1.0
                    H[row, zlo + j] = -1.0
                else:
                    # a_l[j] = 0
                    H[row, alo + j] = 1.0
                descs.append(f"layer {l} neuron {j} relu (active={sign_patterns[l-1][j]})")
                row += 1

        # (iii) output identity rows
        zlo, zhi = slot_z(L)
        alo, ahi = slot_a(L)
        for j in range(sizes[L]):
            H[row, alo + j] = 1.0
            H[row, zlo + j] = -1.0
            descs.append(f"output layer neuron {j} identity")
            row += 1

        # (iv) statistical guard rows
        clean_resid_stat = 0.0
        if manifold_samples is not None:
            for l in range(1, L + 1):
                Zs = []
                for x_s in manifold_samples:
                    pre_s, _ = mlp.forward(x_s)
                    Zs.append(pre_s[l - 1])
                Z = np.stack(Zs)
                mu = Z.mean(axis=0)
                U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
                v_top = Vt[0]
                zlo, zhi = slot_z(l)
                H[row, zlo:zhi] = v_top
                b[row]          = float(v_top @ mu)
                clean_resid_stat += abs(float(v_top @ pre[l - 1]) - b[row]) ** 2
                descs.append(f"layer {l} pre-activation manifold (top PC)")
                row += 1
            clean_resid_stat = float(np.sqrt(clean_resid_stat))

        # Residuals
        resid_all = float(np.linalg.norm(H @ x_clean - b))
        resid_alg = float(np.linalg.norm(
            (H[:n_linearity + n_relu + n_identity] @ x_clean
             - b[:n_linearity + n_relu + n_identity])))

        G = nx.path_graph(L + 1)  # 0 -- 1 -- 2 -- ... -- L

        return SheafSpec(
            graph = G,
            k_v   = max(block_sizes),
            k_e   = int(np.ceil(n_rows / max(1, L))),
            H     = H,
            residual_norm_clean = resid_all,
            invariant_descriptions = descs,
            metadata = {
                "frontend":         "neural_net",
                "layer_sizes":      sizes,
                "offsets":          offsets.tolist(),
                "n_linearity":      n_linearity,
                "n_relu":           n_relu,
                "n_identity":       n_identity,
                "n_stat":           n_stat,
                "b_offset":         b,
                "x_clean":          x_clean,
                "sign_patterns":    sign_patterns,
                "alg_residual":     resid_alg,
                "stat_residual":    clean_resid_stat,
                "input_sample":     input_sample,
                "mlp":              mlp,
            },
        )


if __name__ == "__main__":
    mlp = SmallMLP(layer_sizes=[4, 8, 8, 2], seed=0)
    rng = np.random.default_rng(1)
    inp = rng.standard_normal(4)
    samples = rng.standard_normal((50, 4))
    fe = NeuralNetFrontend()
    # Without statistical guard
    spec = fe.extract(mlp=mlp, input_sample=inp, manifold_samples=None)
    print(f"algebraic-only:  H={spec.H.shape}, clean_resid={spec.residual_norm_clean:.3e}")
    # With statistical guard
    spec = fe.extract(mlp=mlp, input_sample=inp, manifold_samples=samples)
    print(f"with statistical guard:")
    print(f"  H={spec.H.shape}")
    print(f"  algebraic clean residual:  {spec.metadata['alg_residual']:.3e}")
    print(f"  statistical clean residual: {spec.metadata['stat_residual']:.3e}")
    print(f"  total clean residual:       {spec.residual_norm_clean:.3e}")
