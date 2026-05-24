"""
frontend_interface.py: the universal contract every frontend implements.

A "frontend" is anything that turns a system description (program, state
machine, neural network, sensor configuration, ODE) into the triple the
core machinery consumes:

      (graph, vertex stalk dim k_v, edge stalk dim k_e,
       restriction maps F_{v <- e} for each incidence)

If a frontend can produce that triple, the core's sheaf construction,
syndrome computation, OMP decoder, altitude bound, and runtime adaptation
all apply unchanged.  This is what makes the system a *framework* rather
than a tool.

Five frontends in this directory currently implement the interface:

  LinearFrontend         linear-algebra programs (Kalman, FFT, PID, matmul)
  PolynomialFrontend     quadratic invariants (energy, norms, covariance SPD)
  PiecewiseLinearFrontend saturating arithmetic, ReLU, clamped controllers
  NeuralNetFrontend      learned-manifold invariants on layer outputs
  StatisticalFrontend    Gaussian invariants from observed clean traces
  NonlinearFrontend      Taylor / Carleman lifting of trig and exponentials

Each frontend is a subclass of Frontend and overrides `extract`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np
import networkx as nx


# =====================================================================
# Core output type: what every frontend produces
# =====================================================================

@dataclass
class SheafSpec:
    """The triple every frontend produces.

    Attributes:
        graph:      networkx.Graph (vertices = protected state slots,
                    edges = invariant locations)
        k_v:        dimension of each vertex stalk
        k_e:        dimension of each edge stalk
        H:          parity-check matrix of shape (|E|*k_e, |V|*k_v).
                    Each row encodes one scalar invariant.
        residual_norm_clean:  empirical clean-state syndrome norm (should
                              be ~0 if invariants are exact, or small if
                              statistical/learned).
        invariant_descriptions: human-readable list of what each invariant
                                says, length |E|*k_e. For debug & reports.
        metadata:   freeform per-frontend info (program name, ML model
                    hash, sensor topology, etc).
    """
    graph: nx.Graph
    k_v:   int
    k_e:   int
    H:     np.ndarray
    residual_norm_clean: float
    invariant_descriptions: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# =====================================================================
# Base class
# =====================================================================

class Frontend(ABC):
    """Abstract frontend: extract a SheafSpec from a problem description."""

    name: str = "abstract"

    @abstractmethod
    def extract(self, *args, **kwargs) -> SheafSpec:
        """Produce a SheafSpec.

        Concrete frontends define their own argument signatures (a program
        source string, a sensor topology, an ML model, ...). The shared
        contract is the return type.
        """
        ...

    @staticmethod
    def validate(spec: SheafSpec, x_clean: np.ndarray,
                 tol: float = 1e-8) -> dict:
        """Sanity-check: clean state should produce small syndrome.

        Returns a dict with `passed`, `syndrome_norm`, and `relative_norm`.
        If passed is False, the frontend has extracted incorrect invariants
        (the framework will produce false positives in deployment).
        """
        syn = spec.H @ x_clean
        norm = float(np.linalg.norm(syn))
        rel  = norm / max(1.0, float(np.linalg.norm(x_clean)))
        return {
            "passed":        norm < tol,
            "syndrome_norm": norm,
            "relative_norm": rel,
        }
