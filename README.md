# Leray's Coboundary at Altitude

A cellular-sheaf compiler framework for software-defined radiation hardness on deep-space flight computers.

This is the full reproducibility package for the paper submitted to the Synthica × NSRI Global Research Challenge 2026 by Vaibhav Shakya (Jayshree Periwal International School, Jaipur).

**Paper**: [`Leray_Coboundary_at_Altitude_GRC.pdf`](Leray_Coboundary_at_Altitude_GRC.pdf) (26 pages)
**Deck**: [`Leray_Coboundary_at_Altitude_GRC.pptx`](Leray_Coboundary_at_Altitude_GRC.pptx) (16 slides)

---

## What's in here

The framework builds a cellular sheaf on a program's control-flow graph, with restriction maps synthesised automatically from six interchangeable frontends (linear, polynomial, piecewise-linear, neural-network, statistical, nonlinear). A single orthogonal-matching-pursuit decoder protects every frontend. Two theorems are proved in full and verified empirically across 258,645 fault trials.

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── Leray_Coboundary_at_Altitude_GRC.pdf    Final 26-page paper
├── Leray_Coboundary_at_Altitude_GRC.tex    LaTeX source
├── Leray_Coboundary_at_Altitude_GRC.pptx   16-slide presentation deck
├── build_deck.js                            Deck builder (pptxgenjs)
│
├── src/                                     Framework code
│   ├── sheaf_lib.py                         Cellular-sheaf core: H matrices, OMP decoder, Wilson CI
│   ├── fault_model.py                       Markov-modulated Poisson fault model
│   ├── fault_model_real.py                  Real-data calibration scaffold (GOES proton flux)
│   ├── plot_style.py                        Matplotlib style for figures
│   └── frontends/
│       ├── frontend_interface.py            SheafSpec dataclass
│       ├── linear.py                        Linear/state-space frontend
│       ├── polynomial.py                    Polynomial frontend (Macaulay lifting)
│       ├── piecewise_linear.py              Saturation/ReLU frontend
│       ├── neural_net.py                    MLP-inference frontend
│       ├── statistical.py                   PCA-derived ellipsoid frontend
│       └── nonlinear.py                     Carleman-linearisation frontend
│
├── experiments/                             15 reproducible experiments
│   ├── gen_paper_data.py                    Headline coverage matrix (exp01)
│   ├── gen_rigor_data.py                    Deep stats + bitwidth + multifault + altitude + storm (exp01b–10)
│   ├── exp_ldpc_separation.py               Sheaf vs binary LDPC, four fault classes (exp14)
│   ├── exp_ldpc_separation_grid.py          LDPC separation across (kv, ke, G) (exp15)
│   ├── exp_ldpc_separation_adversarial.py   20 random binary LDPC adversarial codes (exp16)
│   ├── exp_structured_map_locus.py          Rank-lemma verification pilot (exp17)
│   ├── exp_structured_map_locus_v3.py       Rank-lemma deep run, 26,153 vertices (exp17c)
│   ├── exp_composed_frontends.py            Quaternion attitude controller, 4 scenarios (exp18)
│   ├── exp_runtime_memory.py                Per-frontend latency + memory (exp19)
│   ├── verify_separation_proof.py           Symbolic Theorem 2 verification
│   └── gen_paper_figures.py                 Build all 16 paper figures from data
│
├── data/                                    17 experiment result JSONs
└── figures/                                 16 figure PDFs
```

---

## Installing dependencies

```bash
pip install -r requirements.txt
```

The framework targets Python 3.11+. Core dependencies: NumPy, SciPy, NetworkX, scikit-learn, matplotlib. The deck builder additionally needs Node.js with pptxgenjs, react, react-dom, sharp, and react-icons.

---

## Reproducing the paper

All experiments use deterministic random seeds; rerunning produces the exact numbers in the paper. Tested on Python 3.11 / NumPy 2.4 / Linux x86-64.

```bash
# 1. Run all experiments (writes data/*.json)
python experiments/gen_paper_data.py                    # exp01 frontend matrix
python experiments/gen_rigor_data.py                    # exp01b–10
python experiments/exp_ldpc_separation.py               # exp14
python experiments/exp_ldpc_separation_grid.py          # exp15
python experiments/exp_ldpc_separation_adversarial.py   # exp16
python experiments/exp_structured_map_locus.py          # exp17 pilot
python experiments/exp_structured_map_locus_v3.py       # exp17c deep
python experiments/exp_composed_frontends.py            # exp18
python experiments/exp_runtime_memory.py                # exp19

# 2. Render figures from data
python experiments/gen_paper_figures.py                 # writes figures/fig*.pdf

# 3. Verify the Sheaf-LDPC separation symbolically (optional sanity check)
python experiments/verify_separation_proof.py

# 4. Compile the paper (LaTeX required)
pdflatex Leray_Coboundary_at_Altitude_GRC.tex
pdflatex Leray_Coboundary_at_Altitude_GRC.tex          # second pass for refs

# 5. Rebuild the deck (Node.js + dependencies required)
npm install pptxgenjs react react-dom sharp react-icons
node build_deck.js
```

Full campaign runtime: approximately 25 minutes on a modern laptop. The adversarial-LDPC experiment (exp16) is the longest single run at roughly 6 minutes.

---

## Headline results

258,645 fault trials across 15 experiments. 10,300 clean trials, zero false positives.

| Result | Value | Detail |
|---|---|---|
| Algebraic frontends (5) | 100% detection | linear, polynomial, PWL, NN, nonlinear |
| Statistical frontend | 83.2% [81.5, 84.8] | structural noise-floor ceiling, n = 1876 |
| False-positive rate | 0 / 10,300 | Wilson upper bound 3.7 × 10⁻⁴ |
| Common-mode at ρ=1.0 | 0% sheaf / 11.3% TMR | 45,000 trials |
| Multi-fault recovery | 100% at k=1, 24% at k=20 | OMP guarantee k ≲ √n |
| Sheaf vs adversarial LDPC | 100.00% vs 97.9% (best of 20) | 40,000 vertex-permutation trials |
| Altitude bound | 8 / 10 configurations match exactly | uniqueness clause added for 2 failures |
| Rank lemma | 97.7% pass rate | 26,153 random-program vertices |
| Mission energy savings | 63% vs always-on TMR | modelled 24-month Europa Clipper |

### Honest negatives

- **Statistical frontend ceiling**: 17% miss rate. PCA invariants have a numerical floor around 10⁻²; faults inside that envelope are absorbed.
- **Multi-fault degradation**: at k=15 simultaneous faults recovery is 53%, at k=20 it falls to 24%. OMP's standard guarantee.
- **Adversarial LDPC margin**: best of 20 random binary LDPC codes catches 97.9% of vertex-permutation faults; the sheaf's lead is 2.1 percentage points, not 100-vs-0.
- **Rank lemma residual**: 2.3% of randomly generated linear programs fall in a measure-zero exclusion set. Real flight code with continuous sensor calibrations passes; pathological synthetic constants can produce the failure.
- **Altitude bound failures**: original theorem statement failed on path-5 graphs with kv ∈ {4, 5} because they have two leaves each (both satisfying the slack condition). Corrected statement adds a uniqueness clause.

---

## The two theorems

**Theorem 1 (Altitude bound).** Let G be a finite graph with no isolated vertices, kv > ke ≥ 1, and restriction maps drawn from any continuous distribution on ℝ^(ke×kv). Suppose there exists a unique vertex v₀ with deg(v₀) < kv/ke, while every other vertex satisfies deg(v) ≥ ⌈kv/ke⌉. Then d(ℱ) = kv almost surely.

**Theorem 2 (Sheaf–LDPC separation).** No binary LDPC code on the same Tanner graph as the sheaf can detect the class of vertex-permutation faults (a swap of two scalars within one stalk whose XOR-parity hashes coincide). The sheaf catches them with probability one.

Both proofs appear in full in `Leray_Coboundary_at_Altitude_GRC.tex`. The symbolic correctness of Theorem 2 part (b) is verified by `experiments/verify_separation_proof.py`.

---

## Beyond spacecraft

The framework's specific advantages — semantic invariants rather than bit-level parity, immunity to common-mode contamination, frontend extensibility — translate to three other domains:

- **Autonomous-vehicle control**: linear and polynomial frontends cover typical control loops; common-mode resilience handles correlated failures in adjacent transistors.
- **Medical device firmware**: insulin pumps and pacemakers. The statistical frontend's PCA approach matches patient-specific calibration curves.
- **Industrial process control**: nuclear-plant PLCs and refinery DCS. Sensor mass-balance, energy-balance, and thermodynamic consistency relations are exactly the algebraic invariants the polynomial frontend was designed to encode.

Each requires domain-specific empirical validation; the framework's mathematical guarantees are domain-agnostic.

---

## Limitations and what's next

- **Real RAD750 cycle counts**: runtime numbers in this repo are x86-64 Linux, NumPy, double precision. A radiation-hardened flight processor would produce different absolute timings; the scaling trends (linear-in-nnz syndrome, O(n^1.5) OMP, sparse-CSR memory) would persist.
- **GOES proton-flux calibration**: the fault model's temporal cadence is currently order-of-magnitude. A calibration driver consuming real proton-flux CSV time series and fitting Markov-modulated Poisson parameters is scaffolded in `src/fault_model_real.py`.
- **LLVM compiler pass**: the framework currently operates on Python program objects. A real LLVM IR pass that inserts sheaf checks into a flight-software binary, followed by cycle-accurate PowerPC simulation, is the most direct route to a hardware claim.
- **Cross-vertex separation**: Theorem 2 covers within-vertex permutations only. Extension to cross-vertex permutations rests on different column-structure arguments and is open.

---

## Citation

If you use this work, please cite:

```bibtex
@unpublished{shakya2026leray,
  author = {Shakya, Vaibhav},
  title  = {Leray's Coboundary at Altitude: A Cellular-Sheaf Compiler Framework
            for Software-Defined Radiation Hardness on Deep-Space Flight Computers},
  note   = {Synthica × NSRI Global Research Challenge submission},
  year   = {2026},
  url    = {https://github.com/vaibhavshakya11/lerays-coboundary-at-altitude}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Contact

Vaibhav Shakya · Jayshree Periwal International School, Jaipur
GitHub: [@vaibhavshakya11](https://github.com/vaibhavshakya11)
