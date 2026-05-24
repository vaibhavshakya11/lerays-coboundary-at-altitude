"""
fault_model.py: realistic radiation fault model for fault-injection experiments.

Replaces the prior Bernoulli-per-op model with a structured model derived from
published radiation-effects literature:

  * Two-state Markov-modulated Poisson process (MMPP) for temporal structure:
    quiet cruise periods alternating with solar-particle-event (SPE) bursts.
    Quiet/storm rates calibrated to the order-of-magnitude ratios reported in
    GOES SEM particle data (NOAA SWPC).

  * Spatial clustering of bit flips along ion tracks: a single "strike"
    corrupts a contiguous block of 1-8 bits in adjacent positions, with the
    multiplicity distribution calibrated to the 42% multi-bit fraction
    reported by Tiwari et al. for production GPUs at scale.

  * Common-mode rate (correlated faults across replicas) set to ~5% per the
    NSREC TMR literature (Quinn et al., 2019).

  * Three fault CLASSES, not just bit flips:
      - VALUE flip (bit-level upset, possibly multi-bit)
      - SEFI (single-event functional interrupt: wholesale value substitution)
      - LATCH (stuck-at fault until reset)

Source notes:
    GOES SEM data:    https://www.swpc.noaa.gov/products/goes-proton-flux
    NSREC proceedings: IEEE Trans. on Nuclear Science annual issues
    JEDEC JESD89A:    soft-error-rate measurement standard
    Tiwari 2015 HPCA: 42% multi-bit fault fraction at scale

This module exposes a single FaultStream class that yields realistic
fault events over a simulated mission time window.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FaultClass(Enum):
    """Three classes of radiation fault, each with different recovery behaviour."""
    VALUE = "value"     # bit flip(s), possibly multi-bit. Sheaf can recover.
    SEFI  = "sefi"      # functional interrupt, wholesale corruption.
                        # Sheaf detects (large syndrome) but cannot recover
                        # via OMP; requires reset.
    LATCH = "latch"     # stuck-at fault. Detected indefinitely until reset.


@dataclass
class FaultEvent:
    """One radiation event."""
    time_s:       float          # mission-time seconds when strike arrived
    var_idx:      int            # index of struck variable
    fault_class:  FaultClass
    bit_width:    int = 1        # number of bits flipped (1-8 for spatial cluster)
    magnitude:    float = 1.0    # additive perturbation magnitude
    common_mode:  bool = False   # True if this strike hits all replicas


class FaultStream:
    """Generates realistic radiation fault sequences over mission time.

    Calibration parameters are documented inline. The default values produce
    rates consistent with order-of-magnitude estimates for a deep-space
    mission (Europa-Clipper class), but the structure of the model is what
    matters: bursty time, clustered bits, three fault classes.

    Usage:
        stream = FaultStream(n_vars=400, total_seconds=3600, seed=42)
        events = stream.generate()
        # events is a list of FaultEvent ordered by time_s
    """

    # --- temporal MMPP parameters (mission-seconds) ---
    QUIET_RATE_PER_VAR_PER_SEC = 1e-7    # cruise / quiet sun
    STORM_RATE_PER_VAR_PER_SEC = 1e-4    # solar particle event
    MEAN_QUIET_DURATION_S      = 36000   # ~10 hours quiet
    MEAN_STORM_DURATION_S      = 1800    # ~30 min storm

    # --- spatial clustering ---
    # P(bit_width = k) for an ion strike, k = 1..8.
    # Calibrated so P(k>=2) ~= 0.42 per Tiwari et al. 2015 HPCA
    # (42% of GPU faults observed at scale involved multiple adjacent bits).
    SPATIAL_PMF = np.array([0.58, 0.16, 0.10, 0.07, 0.05, 0.02, 0.01, 0.01])

    # --- fault-class proportions ---
    P_VALUE = 0.92    # bit-level upsets dominate
    P_SEFI  = 0.06    # rare but unrecoverable
    P_LATCH = 0.02    # rarer still

    # --- common-mode fraction (replica-correlated) ---
    P_COMMON_MODE = 0.05    # per NSREC TMR literature

    def __init__(self,
                 n_vars: int,
                 total_seconds: float,
                 seed: int = 0,
                 quiet_rate: Optional[float] = None,
                 storm_rate: Optional[float] = None,
                 storm_fraction: Optional[float] = None):
        self.n_vars       = n_vars
        self.total_seconds = total_seconds
        self.rng          = np.random.default_rng(seed)
        if quiet_rate is not None:
            self.QUIET_RATE_PER_VAR_PER_SEC = quiet_rate
        if storm_rate is not None:
            self.STORM_RATE_PER_VAR_PER_SEC = storm_rate
        if storm_fraction is not None:
            # set mean storm duration so the long-run storm fraction matches
            self.MEAN_STORM_DURATION_S = (
                storm_fraction * self.MEAN_QUIET_DURATION_S
                / (1 - storm_fraction)
            )

    def _sample_state_durations(self) -> List[tuple]:
        """Return list of (state, start_s, end_s) tuples covering [0, total_seconds]."""
        durations = []
        t = 0.0
        # start in quiet state with probability matching stationary distribution
        p_storm = (self.MEAN_STORM_DURATION_S
                   / (self.MEAN_QUIET_DURATION_S + self.MEAN_STORM_DURATION_S))
        state = "storm" if self.rng.random() < p_storm else "quiet"
        while t < self.total_seconds:
            mean = (self.MEAN_STORM_DURATION_S if state == "storm"
                    else self.MEAN_QUIET_DURATION_S)
            dur = self.rng.exponential(mean)
            end = min(t + dur, self.total_seconds)
            durations.append((state, t, end))
            t = end
            state = "quiet" if state == "storm" else "storm"
        return durations

    def _sample_bit_width(self) -> int:
        """Spatial cluster size of one ion strike."""
        return int(self.rng.choice(np.arange(1, 9), p=self.SPATIAL_PMF))

    def _sample_fault_class(self) -> FaultClass:
        u = self.rng.random()
        if u < self.P_VALUE:
            return FaultClass.VALUE
        if u < self.P_VALUE + self.P_SEFI:
            return FaultClass.SEFI
        return FaultClass.LATCH

    def generate(self) -> List[FaultEvent]:
        """Generate the full fault sequence for this mission window."""
        events: List[FaultEvent] = []
        state_segments = self._sample_state_durations()

        for state, t_start, t_end in state_segments:
            rate = (self.STORM_RATE_PER_VAR_PER_SEC if state == "storm"
                    else self.QUIET_RATE_PER_VAR_PER_SEC)
            # Aggregate Poisson rate across all variables for this segment
            total_rate = rate * self.n_vars
            expected_n = total_rate * (t_end - t_start)
            n_events = self.rng.poisson(expected_n)
            for _ in range(n_events):
                ev_time   = self.rng.uniform(t_start, t_end)
                ev_var    = int(self.rng.integers(0, self.n_vars))
                ev_class  = self._sample_fault_class()
                ev_bits   = self._sample_bit_width() if ev_class == FaultClass.VALUE else 1
                ev_common = (self.rng.random() < self.P_COMMON_MODE)
                if ev_class == FaultClass.VALUE:
                    # Magnitude proportional to bit width (bigger flip = larger
                    # numerical perturbation, roughly)
                    ev_mag = self.rng.standard_normal() * (2.0 ** ev_bits)
                elif ev_class == FaultClass.SEFI:
                    ev_mag = self.rng.standard_normal() * 1e6  # wholesale corruption
                else:  # LATCH
                    ev_mag = self.rng.choice([-1.0, 1.0]) * 1e3
                events.append(FaultEvent(
                    time_s      = ev_time,
                    var_idx     = ev_var,
                    fault_class = ev_class,
                    bit_width   = ev_bits,
                    magnitude   = ev_mag,
                    common_mode = ev_common,
                ))

        events.sort(key=lambda e: e.time_s)
        return events

    def summary(self, events: List[FaultEvent]) -> dict:
        """Quick summary statistics about a generated fault sequence."""
        if not events:
            return {"n_events": 0}
        return {
            "n_events":        len(events),
            "n_value":         sum(1 for e in events if e.fault_class == FaultClass.VALUE),
            "n_sefi":          sum(1 for e in events if e.fault_class == FaultClass.SEFI),
            "n_latch":         sum(1 for e in events if e.fault_class == FaultClass.LATCH),
            "n_multi_bit":     sum(1 for e in events if e.bit_width >= 2),
            "n_common_mode":   sum(1 for e in events if e.common_mode),
            "mean_bit_width":  float(np.mean([e.bit_width for e in events])),
            "p_multi_bit":     sum(1 for e in events if e.bit_width >= 2) / len(events),
            "p_common_mode":   sum(1 for e in events if e.common_mode) / len(events),
            "events_per_sec":  len(events) / self.total_seconds,
        }


if __name__ == "__main__":
    # smoke test
    stream = FaultStream(n_vars=400, total_seconds=24 * 3600, seed=42)
    events = stream.generate()
    summary = stream.summary(events)
    print("Fault stream summary (24h mission window, 400 vars):")
    for k, v in summary.items():
        print(f"  {k:18}: {v}")
