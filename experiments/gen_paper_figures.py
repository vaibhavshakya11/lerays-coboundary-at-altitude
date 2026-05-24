"""
gen_paper_figures.py
====================
Produces all figures for the v5 paper from paper/data/*.json into
paper/figures/*.pdf using matplotlib.

Palette: navy primary (#1E3A5F), gold accent (#B8860B). NO teal.
Style:   serif body, sans for labels, no top/right spines, gridlines minimal.

Run:  python3 paper/gen_paper_figures.py
"""
import os, json, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Palette
NAVY    = "#1E3A5F"
GOLD    = "#B8860B"
BRICK   = "#A8331E"
SLATE   = "#445566"
PLUM    = "#6B2D5C"
STEEL   = "#4A6B8A"
FOREST  = "#2D5F3F"
GREY    = "#8A8A8A"
LIGHT   = "#D8D8D8"

# Per-frontend color palette
FRONTEND_COLORS = {
    "linear":           NAVY,
    "polynomial":       GOLD,
    "piecewise_linear": BRICK,
    "neural_net":       FOREST,
    "statistical":      SLATE,
    "nonlinear":        PLUM,
}

# Per-technique color palette (for baseline comparisons)
TECH_COLORS = {
    "sheaf":   NAVY,
    "tmr":     BRICK,
    "swift":   PLUM,
    "swift_r": PLUM,
    "secded":  STEEL,
    "abft":    GOLD,
    "none":    GREY,
}

PRETTY_FRONTEND = {
    "linear":           "Linear",
    "polynomial":       "Polynomial",
    "piecewise_linear": "Piecewise-linear",
    "neural_net":       "Neural network",
    "statistical":      "Statistical",
    "nonlinear":        "Nonlinear",
}


def apply_style():
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Palatino", "Liberation Serif", "DejaVu Serif"],
        "font.size":         10,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.7,
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#101010",
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "grid.color":        "#D8D8D8",
        "grid.linewidth":    0.5,
        "grid.alpha":        0.7,
        "legend.frameon":    True,
        "legend.framealpha": 0.95,
        "legend.edgecolor":  "#D8D8D8",
        "legend.fancybox":   False,
        "figure.dpi":        110,
        "savefig.dpi":       250,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
        "lines.linewidth":   1.5,
        "lines.markersize":  5,
        "patch.linewidth":   0.5,
        "pdf.fonttype":      42,    # embed as TrueType so editors can read
        "ps.fonttype":       42,
    })


def add_grid(ax, axis="y", alpha=0.5):
    ax.grid(True, axis=axis, alpha=alpha, linestyle="-", linewidth=0.5,
            zorder=0)
    ax.set_axisbelow(True)


def load(name):
    with open(os.path.join(DATA_DIR, name + ".json")) as f:
        return json.load(f)


# ============================================================
# Figure 1: Frontend matrix — detection / recovery / FPR heatmap
# ============================================================
def fig1_frontend_matrix():
    """Redesigned for v5.1: bar chart with explicit Wilson 95% CIs, using
    the deep statistical run (exp01b) where available."""
    d  = load("exp01_frontend_matrix")
    try:
        d_stat = load("exp01b_statistical_deep")
        # The deep run replaces the pilot statistical numbers
        d["results"]["statistical"] = d_stat["results"]
    except FileNotFoundError:
        pass
    order = ["linear", "polynomial", "piecewise_linear",
             "neural_net", "statistical", "nonlinear"]
    fig, ax = plt.subplots(figsize=(9.7, 3.6))
    x = np.arange(len(order))
    bar_w = 0.36
    det   = [d["results"][f]["detection_rate"] for f in order]
    rec   = [d["results"][f]["recovery_rate"]  for f in order]
    det_ci = [d["results"][f]["detection_ci"] for f in order]
    rec_ci = [d["results"][f]["recovery_ci"] for f in order]
    det_err = [
        [max(0, det[i] - det_ci[i][0]) for i in range(len(order))],
        [max(0, det_ci[i][1] - det[i]) for i in range(len(order))],
    ]
    rec_err = [
        [max(0, rec[i] - rec_ci[i][0]) for i in range(len(order))],
        [max(0, rec_ci[i][1] - rec[i]) for i in range(len(order))],
    ]
    bars1 = ax.bar(x - bar_w/2, det, bar_w, color=NAVY,
                    yerr=det_err, capsize=2.5, ecolor="#222222",
                    error_kw={"elinewidth": 0.7},
                    label="Detection rate")
    bars2 = ax.bar(x + bar_w/2, rec, bar_w, color=GOLD,
                    yerr=rec_err, capsize=2.5, ecolor="#222222",
                    error_kw={"elinewidth": 0.7},
                    label="Recovery rate")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_FRONTEND[f] for f in order],
                        rotation=10, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Rate (95% Wilson CI)")
    ax.set_title("Frontend coverage: six problem classes, one decoder",
                 loc="left", fontsize=11)
    # Annotate n alongside each frontend
    for i, fn in enumerate(order):
        n = d["results"][fn]["n"]
        ax.text(x[i], -0.13, f"n={n}", ha="center", va="top",
                fontsize=7.5, color="#404040")
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig01_frontend_matrix.pdf"))
    plt.close()


# ============================================================
# Figure 2: Per-fault-class detection per frontend
# ============================================================
def fig2_by_class():
    d = load("exp01_frontend_matrix")
    try:
        d_stat = load("exp01b_statistical_deep")
        d["results"]["statistical"] = d_stat["results"]
    except FileNotFoundError:
        pass
    order = ["linear", "polynomial", "piecewise_linear",
             "neural_net", "statistical", "nonlinear"]
    classes = ["value", "sefi", "latch"]
    class_labels = ["Bit flip (VALUE)", "Functional interrupt (SEFI)",
                    "Stuck-at (LATCH)"]
    class_colors = [NAVY, GOLD, BRICK]
    fig, ax = plt.subplots(figsize=(8.7, 3.6))
    x = np.arange(len(order))
    width = 0.27
    for i, c in enumerate(classes):
        rates = []
        ci_lo, ci_hi = [], []
        for fn in order:
            cd = d["results"][fn]["by_class"][c]
            if cd["n"] > 0:
                rates.append(cd["det"] / cd["n"])
                ci_lo.append(cd["detection_ci"][0])
                ci_hi.append(cd["detection_ci"][1])
            else:
                rates.append(0); ci_lo.append(0); ci_hi.append(0)
        bars = ax.bar(x + (i - 1) * width, rates, width,
                      color=class_colors[i], edgecolor="#333333", linewidth=0.5,
                      label=class_labels[i])
        # error bars
        errs = np.array([[max(0, r - lo), max(0, hi - r)] for r, lo, hi in
                         zip(rates, ci_lo, ci_hi)]).T
        ax.errorbar(x + (i - 1) * width, rates, yerr=errs,
                    fmt="none", ecolor="#101010", elinewidth=0.7, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_FRONTEND[f] for f in order], rotation=20,
                       ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Detection rate")
    ax.set_title("Detection rate per fault class, per frontend (95% Wilson CI)",
                 loc="left", fontsize=11)
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.savefig(os.path.join(FIG_DIR, "fig02_by_class.pdf"))
    plt.close()


# ============================================================
# Figure 3: Aux-row ablation
# ============================================================
def fig3_aux_ablation():
    d = load("exp07_aux_row_ablation")
    fes = ["polynomial", "nonlinear"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    width = 0.32
    x = np.arange(len(fes))
    # Build per-frontend data
    by_fe = {fe: {} for fe in fes}
    for r in d["rows"]:
        key = "yes" if r["include_aux"] else "no"
        by_fe[r["frontend"]][key] = (r["detection_rate"], r["recovery_rate"],
                                      r["detection_ci"], r["recovery_ci"])
    det_no  = [by_fe[fe]["no"][0]  for fe in fes]
    det_yes = [by_fe[fe]["yes"][0] for fe in fes]
    rec_no  = [by_fe[fe]["no"][1]  for fe in fes]
    rec_yes = [by_fe[fe]["yes"][1] for fe in fes]
    bars1 = ax.bar(x - 1.5*width/2, det_no,  width/2, color=LIGHT,
                    edgecolor=NAVY, linewidth=0.8, label="det, no aux")
    bars2 = ax.bar(x - 0.5*width/2, det_yes, width/2, color=NAVY,
                    edgecolor="#101010", linewidth=0.5, label="det, with aux")
    bars3 = ax.bar(x + 0.5*width/2, rec_no,  width/2, color="#F4E4B0",
                    edgecolor=GOLD, linewidth=0.8, label="rec, no aux")
    bars4 = ax.bar(x + 1.5*width/2, rec_yes, width/2, color=GOLD,
                    edgecolor="#101010", linewidth=0.5, label="rec, with aux")
    # Annotate values
    for bars, vals in zip([bars1, bars2, bars3, bars4],
                          [det_no, det_yes, rec_no, rec_yes]):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_FRONTEND[fe] for fe in fes])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Aux-row ablation: monomial-consistency rows on every slot",
                 loc="left", fontsize=11)
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.savefig(os.path.join(FIG_DIR, "fig03_aux_ablation.pdf"))
    plt.close()


# ============================================================
# Figure 4: Blind slots before/after aux rows
# ============================================================
def fig4_blind_slots():
    d = load("exp09_blind_slots")
    fes = [r["frontend"] for r in d["rows"]]
    blind_no  = [r["blind_cols_no_aux"]  for r in d["rows"]]
    blind_yes = [r["blind_cols_with_aux"] for r in d["rows"]]
    n_cols    = [r["n_cols"] for r in d["rows"]]
    pct_no    = [100 * b / n for b, n in zip(blind_no, n_cols)]
    pct_yes   = [100 * b / n for b, n in zip(blind_yes, n_cols)]
    fig, ax = plt.subplots(figsize=(7.3, 2.8))
    x = np.arange(len(fes))
    width = 0.34
    b1 = ax.bar(x - width/2, pct_no, width, color=BRICK, edgecolor="#101010",
                linewidth=0.5, label="No aux rows")
    b2 = ax.bar(x + width/2, pct_yes, width, color=FOREST, edgecolor="#101010",
                linewidth=0.5, label="Aux rows on every slot")
    for bar, pct in zip(b1, pct_no):
        ax.text(bar.get_x() + bar.get_width()/2, pct + 1.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for bar, pct in zip(b2, pct_yes):
        ax.text(bar.get_x() + bar.get_width()/2, max(pct, 0) + 1.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_FRONTEND[fe] for fe in fes])
    ax.set_ylabel("Lifted slots with no constraint (%)")
    ax.set_ylim(0, 90)
    ax.set_title("Carleman/Macaulay lifting leaves most slots blind by default",
                 loc="left", fontsize=11)
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.savefig(os.path.join(FIG_DIR, "fig04_blind_slots.pdf"))
    plt.close()


# ============================================================
# Figure 5: Decoder scaling
# ============================================================
def fig5_decoder_scaling():
    d = load("exp02_decoder_scaling")
    rows = d["rows"]
    ns       = [r["n"] for r in rows]
    medians  = [r["median_ms"] for r in rows]
    p25      = [r["p25_ms"] for r in rows]
    p75      = [r["p75_ms"] for r in rows]
    p95      = [r["p95_ms"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax1.fill_between(ns, p25, p75, color=NAVY, alpha=0.2,
                     label="25th-75th percentile")
    ax1.loglog(ns, medians, "o-", color=NAVY, label="Median latency",
               linewidth=1.6)
    ax1.loglog(ns, p95, "--", color=BRICK, label="95th percentile",
               linewidth=1.2)
    ref_ns = np.array(ns, dtype=float)
    ref_y = (medians[0] / (ref_ns[0]**1.5)) * ref_ns**1.5
    ax1.loglog(ref_ns, ref_y, ":", color=GREY, label=r"$O(n^{1.5})$ reference")
    ax1.set_xlabel("Program size $n$ (basic blocks)")
    ax1.set_ylabel("Decoder latency (ms)")
    ax1.set_title("Single-fault OMP latency", loc="left", fontsize=10.5)
    # Legend below the plot to avoid overlapping the curves
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=False)
    add_grid(ax1, axis="both")
    succ = [r["success_rate"] for r in rows]
    succ_ci_lo = [r["success_ci"][0] for r in rows]
    succ_ci_hi = [r["success_ci"][1] for r in rows]
    ax2.fill_between(ns, succ_ci_lo, succ_ci_hi, color=FOREST, alpha=0.2)
    ax2.semilogx(ns, succ, "o-", color=FOREST, linewidth=1.6,
                 label="Recovery rate (95% Wilson CI)")
    ax2.set_xlabel("Program size $n$ (basic blocks)")
    ax2.set_ylabel("Single-fault recovery rate")
    ax2.set_ylim(0.94, 1.005)
    ax2.set_title("Recovery success", loc="left", fontsize=10.5)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=False)
    add_grid(ax2, axis="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig05_decoder_scaling.pdf"))
    plt.close()


# ============================================================
# Figure 6: Multi-fault OMP recovery
# ============================================================
def fig6_multifault():
    d = load("exp03_multifault")
    rows = d["rows"]
    ks = [r["k"] for r in rows]
    rates = [r["mean_rate"] for r in rows]
    los   = [r["ci_lo"] for r in rows]
    his   = [r["ci_hi"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.fill_between(ks, los, his, color=NAVY, alpha=0.18, label="95% Wilson CI")
    ax.plot(ks, rates, "o-", color=NAVY, linewidth=1.8,
            label="Mean recovery rate (3 sheaf seeds, 100 trials each)")
    ax.axhline(0.5, color=GREY, linestyle="--", linewidth=0.8)
    ax.axhline(0.95, color=GOLD, linestyle=":", linewidth=0.8)
    # Annotations placed at left edge so they don't overlap data near k=20
    ax.text(0.6, 0.52, "50%", va="center", ha="left",
            color=GREY, fontsize=8)
    ax.text(0.6, 0.97, "95%", va="bottom", ha="left",
            color=GOLD, fontsize=8)
    sqrt_n = int(np.sqrt(100))
    ax.axvline(sqrt_n, color=BRICK, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(sqrt_n + 0.3, 0.05, r"$k \approx \sqrt{n}$",
            color=BRICK, fontsize=8.5)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, 21))
    ax.set_xlabel("Number of simultaneous faults $k$")
    ax.set_ylabel("Recovery rate")
    ax.set_title("Multi-fault OMP recovery on cycle-100 sheaf "
                 "($k_v=4$, $k_e=2$, state dim 400)",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    # Legend below the plot
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig06_multifault.pdf"))
    plt.close()


# ============================================================
# Figure 7: Altitude bound verification (corrected vs v3)
# ============================================================
def fig7_altitude_bound():
    d = load("exp04_altitude_bound")
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    x = np.arange(len(rows))
    naive_added = False
    for i, r in enumerate(rows):
        means = r["measured_mean"]
        mins  = r["measured_min"]
        maxs  = r["measured_max"]
        ax.errorbar([i], [means],
                    yerr=[[means - mins], [maxs - means]],
                    fmt="o", color=NAVY, markersize=6, capsize=3,
                    elinewidth=0.8, label="Measured" if i == 0 else None)
        if r["predicted_v5_corrected"] is not None:
            ax.scatter([i], [r["predicted_v5_corrected"]], marker="x",
                       color=FOREST, s=70, zorder=5,
                       label="Theorem prediction" if i == 0 else None)
        if r["predicted_v3_paper"] is not None and \
           r["predicted_v3_paper"] != r["predicted_v5_corrected"]:
            ax.scatter([i], [r["predicted_v3_paper"]], marker="^",
                       color=BRICK, s=60, zorder=5,
                       label="Naive prediction (overestimate)" if not naive_added else None)
            naive_added = True
    ax.set_xticks(x)
    labels = [r["name"] for r in rows]
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Minimum distance $d(\\mathcal{F})$")
    ax.set_ylim(0, 8)
    ax.set_title("Altitude bound: theorem prediction vs measured distance",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig07_altitude_bound.pdf"))
    plt.close()


# ============================================================
# Figure 8: Common-mode sensitivity
# ============================================================
def fig8_common_mode():
    d = load("exp05_common_mode")
    rows = d["rows"]
    cms = [r["common_mode_fraction"] for r in rows]
    sheaf = [r["sheaf_failure_rate"] for r in rows]
    tmr   = [r["tmr_failure_rate"]  for r in rows]
    swift = [r["swift_failure_rate"] for r in rows]
    secded = [r["secded_failure_rate"] for r in rows]
    none  = [r["none_failure_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    # Shade the physically realistic region 1%-10%
    ax.axvspan(0.01, 0.10, color=GOLD, alpha=0.10, zorder=0)
    ax.text(0.032, 0.175, "physically realistic\n(heavy-ion data)",
            ha="center", va="center", fontsize=7.5, color="#866b00",
            zorder=1)
    ax.plot(cms, sheaf, "o-", color=NAVY, linewidth=1.8, label="Sheaf (ours)")
    ax.plot(cms, tmr,   "s-", color=BRICK, linewidth=1.5, label="TMR")
    ax.plot(cms, swift, "D-", color=PLUM, linewidth=1.5, label="SWIFT-R")
    ax.plot(cms, secded, "v-", color=STEEL, linewidth=1.5, label="SECDED")
    ax.plot(cms, none,  "^--", color=GREY, linewidth=1.2, label="No protection")
    ax.set_xscale("log")
    ax.set_xlim(0.0009, 1.05)
    ax.set_ylim(0, 0.20)
    ax.set_xlabel("Fraction of faults that are common-mode")
    ax.set_ylabel("End-to-end failure rate")
    ax.set_title("Common-mode sensitivity (total fault rate 0.001/op, N=5000)",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    # Move legend outside, right of plot area
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5,
              frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig08_common_mode.pdf"))
    plt.close()


# ============================================================
# Figure 9: Bit-width sensitivity per frontend
# ============================================================
def fig9_bitwidth():
    d = load("exp06_bitwidth")
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    for r in d["rows"]:
        fe = r["frontend"]
        by_bw = r["by_bitwidth"]
        bws = sorted([int(k) for k in by_bw.keys()])
        rates = [by_bw[str(b)]["detection_rate"] for b in bws]
        ax.plot(bws, rates, "o-", color=FRONTEND_COLORS[fe],
                label=PRETTY_FRONTEND[fe], linewidth=1.5)
    ax.set_xlabel("Fault bit-width (number of adjacent bits flipped)")
    ax.set_ylabel("Detection rate")
    ax.set_xticks(range(1, 9))
    ax.set_ylim(0.75, 1.02)
    ax.set_title("Detection vs spatial cluster size; all frontends at 1.000 except statistical",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    # Move legend outside, right of plot area
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5,
              frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig09_bitwidth.pdf"))
    plt.close()


# ============================================================
# Figure 10: Europa Clipper mission profile
# ============================================================
def fig10_mission():
    d = load("exp08_mission")
    months = np.array(d["months"])
    flux   = np.array(d["flux"])
    tmr    = np.array(d["tmr_cum_cost"])
    maxc   = np.array(d["max_cum_cost"])
    minc   = np.array(d["min_cum_cost"])
    adapt  = np.array(d["adapt_cum_cost"])
    flyby_months = [3, 7, 11, 14, 17, 20]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 4.6),
                                    gridspec_kw={"height_ratios": [1, 1.6]})
    ax1.plot(months, flux, color=GOLD, linewidth=1.5)
    ax1.fill_between(months, 1, flux, color=GOLD, alpha=0.15)
    for fm in flyby_months:
        ax1.axvline(fm, color=BRICK, linestyle=":", linewidth=0.6, alpha=0.6)
        ax1.text(fm, 31, "F", color=BRICK, ha="center", va="bottom",
                 fontsize=8)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 33)
    ax1.set_xticks([])
    ax1.set_ylabel("Particle flux (×baseline)")
    ax1.set_title("Modeled Europa Clipper profile (24 mo, 6 flybys)",
                  loc="left", fontsize=10.5)
    add_grid(ax1)
    ax2.plot(months, tmr,   color=BRICK, linewidth=1.8,
             label=f"TMR always (total {d['totals']['tmr']:.0f})")
    ax2.plot(months, maxc,  color=GOLD,  linewidth=1.5,
             label=f"Max sheaf always (total {d['totals']['max']:.0f})")
    ax2.plot(months, adapt, color=NAVY,  linewidth=2.0,
             label=f"Adaptive sheaf (ours, total {d['totals']['adapt']:.0f})")
    ax2.plot(months, minc,  color=FOREST, linewidth=1.2, linestyle="--",
             label=f"Minimal always (total {d['totals']['min']:.0f})")
    ax2.set_xlim(0, 24)
    ax2.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax2.set_xlabel("Mission time (months)")
    ax2.set_ylabel("Cumulative energy (overhead-months)")
    ax2.set_title(f"Cumulative protection cost: adaptive saves "
                  f"{d['savings_vs_tmr_pct']:.0f}% vs TMR",
                  loc="left", fontsize=10.5)
    add_grid(ax2)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig10_mission.pdf"))
    plt.close()


# ============================================================
# Figure 11: Mission-integrated cost (bar)
# ============================================================
def fig11_mission_summary():
    d = load("exp08_mission")
    strategies = ["Minimal\nalways", "Max sheaf\nalways",
                  "TMR\nalways", "Adaptive\n(ours)"]
    costs = [d["totals"]["min"], d["totals"]["max"],
             d["totals"]["tmr"], d["totals"]["adapt"]]
    covs  = [d["mean_coverage"]["min"], d["mean_coverage"]["max"],
             d["mean_coverage"]["tmr"], d["mean_coverage"]["adapt"]]
    colors = [FOREST, GOLD, BRICK, NAVY]
    fig, ax = plt.subplots(figsize=(8.1, 3.4))
    x = np.arange(len(strategies))
    bars = ax.bar(x, costs, color=colors, edgecolor="#101010", linewidth=0.5)
    for bar, c, cov in zip(bars, costs, covs):
        ax.text(bar.get_x() + bar.get_width()/2, c + 1.5,
                f"{c:.0f}\n({cov*100:.1f}%)", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(strategies)
    ax.set_ylabel("Total energy cost (overhead-months)")
    ax.set_ylim(0, max(costs) * 1.18)
    saved = (d["totals"]["tmr"] - d["totals"]["adapt"]) / d["totals"]["tmr"] * 100
    ax.set_title(f"Mission-integrated cost by strategy "
                 f"(adaptive saves {saved:.0f}% vs TMR at equivalent coverage)",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    plt.savefig(os.path.join(FIG_DIR, "fig11_mission_summary.pdf"))
    plt.close()


# ============================================================
# Figure 12: Common-mode sensitivity at MULTIPLE fault rates
# (rigor improvement: addresses reviewer concern about Quinn et al
#  rate being HPC-not-spacecraft)
# ============================================================
def fig12_cm_sensitivity_grid():
    d = load("exp05b_cm_sensitivity")
    fault_rates = d["fault_rates"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    # Shade physically realistic range
    ax.axvspan(0.01, 0.10, color=GOLD, alpha=0.10, zorder=0)
    ax.text(0.032, 0.218, "HPC-derived\nrange [Quinn 2019]",
            ha="center", va="center", fontsize=7.5, color="#866b00",
            zorder=1)
    # Sheaf is flat (essentially zero) across all rates — show one curve
    cms_ref = [r["common_mode_fraction"] for r in d["grids"][str(fault_rates[0])]]
    sheaf_ref = [r["sheaf_failure_rate"]
                 for r in d["grids"][str(fault_rates[0])]]
    ax.plot(cms_ref, sheaf_ref, "o-", color=NAVY, linewidth=2.0,
            label="Sheaf (all rates: indistinguishable from zero)")
    # TMR at three rates
    tmr_colors = [BRICK, "#C9572D", "#E07A41"]
    for i, rate in enumerate(fault_rates):
        rows = d["grids"][str(rate)]
        cms = [r["common_mode_fraction"] for r in rows]
        tmr = [r["tmr_failure_rate"] for r in rows]
        ax.plot(cms, tmr, "s--", color=tmr_colors[i], linewidth=1.4,
                label=f"TMR at fault rate $p = {rate:.0e}$/op",
                markersize=4.5, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(0.0009, 1.05)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("Fraction of faults that are common-mode")
    ax.set_ylabel("End-to-end failure rate")
    ax.set_title("Common-mode sensitivity at three fault rates "
                 "(4$\\times$ range bracketing the Quinn et al.\\ estimate)",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    # Legend below the plot, two rows
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig12_cm_grid.pdf"))
    plt.close()


# ============================================================
# Figure 13: Storm fraction sweep
# ============================================================
def fig13_storm_sweep():
    d = load("exp10_storm_sweep")
    rows = d["rows"]
    sfs   = [r["storm_fraction"] for r in rows]
    det   = [r["detection_rate"] for r in rows]
    det_lo = [r["detection_ci"][0] for r in rows]
    det_hi = [r["detection_ci"][1] for r in rows]
    ns    = [r["n_events"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.fill_between(sfs, det_lo, det_hi, color=NAVY, alpha=0.18,
                    label="95% Wilson CI")
    ax.plot(sfs, det, "o-", color=NAVY, linewidth=1.8,
            label="Detection rate")
    ax.set_ylim(0.95, 1.005)
    ax.set_xlabel("Storm fraction (fraction of mission time in solar particle event)")
    ax.set_ylabel("Detection rate")
    ax.set_title("Detection holds under bursty fault regimes "
                 "(linear frontend, 10 streams x 4h per point)",
                 loc="left", fontsize=10.5)
    # Annotate n above each point
    for sf, d_, n in zip(sfs, det, ns):
        ax.text(sf, 1.001, f"n={n}", ha="center", va="bottom",
                fontsize=7.5, color="#404040")
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig13_storm_sweep.pdf"))
    plt.close()


# ============================================================
# Figure 14: LDPC separation across four fault classes
# (sheaf vs LDPC-XOR vs LDPC-32-plane on the same Tanner graph)
# ============================================================
def fig14_ldpc_separation():
    d = load("exp14_ldpc_separation")
    rows = d["results"]
    classes = [r["class"] for r in rows]
    class_labels = {
        "A": "A: 1-bit flip",
        "B": "B: 2-bit XOR-zero",
        "C": "C: vertex-perm",
        "D": "D: sign flip",
    }
    labels = [class_labels[c] for c in classes]
    sheaf      = [r["sheaf_rate"]       for r in rows]
    ldpc_xor   = [r["ldpc_xor_rate"]    for r in rows]
    ldpc_plane = [r["ldpc_plane_rate"]  for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    x = np.arange(len(classes))
    w = 0.28
    ax.bar(x - w, sheaf,      w, color=NAVY,  edgecolor="#202020", linewidth=0.4,
           label="Sheaf (ours)")
    ax.bar(x,     ldpc_xor,   w, color=GOLD,  edgecolor="#202020", linewidth=0.4,
           label="Binary LDPC, XOR-parity")
    ax.bar(x + w, ldpc_plane, w, color=BRICK, edgecolor="#202020", linewidth=0.4,
           label="Binary LDPC, 32 bit-planes")
    # Value annotations
    for i, vals in enumerate(zip(sheaf, ldpc_xor, ldpc_plane)):
        for j, v in enumerate(vals):
            ax.text(x[i] + (j - 1) * w, v + 0.02,
                    f"{v:.2f}" if v > 0.05 else "0.00",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Detection rate")
    ax.set_title("Sheaf vs binary LDPC on the same Tanner graph "
                 "(cycle-10, $k_v=4$, $k_e=2$, n=2000/class)",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig14_ldpc_separation.pdf"))
    plt.close()


# ============================================================
# Figure 15: Composed-frontend quaternion attitude task
# ============================================================
def fig15_composed_frontends():
    d = load("exp18_composed_frontends")
    rows = d["results"]
    scenarios = [r["scenario"] for r in rows]
    pretty = {
        "quat_bitflip": "Quaternion\nbit-flip",
        "rate_bitflip": "Body-rate\nbit-flip",
        "quat_swap":    "Quaternion\nswap",
        "burst":        "Multi-bit\nburst",
    }
    labels = [pretty[s] for s in scenarios]
    lin   = [r["linear_only_rate"]     for r in rows]
    pol   = [r["polynomial_only_rate"] for r in rows]
    comp  = [r["composed_rate"]        for r in rows]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(scenarios))
    w = 0.27
    ax.bar(x - w, lin,  w, color=NAVY,  edgecolor="#222", linewidth=0.4,
           label="Linear frontend only")
    ax.bar(x,     pol,  w, color=GOLD,  edgecolor="#222", linewidth=0.4,
           label="Polynomial frontend only")
    ax.bar(x + w, comp, w, color=FOREST, edgecolor="#222", linewidth=0.4,
           label="Both composed")
    for i, vals in enumerate(zip(lin, pol, comp)):
        for j, v in enumerate(vals):
            ax.text(x[i] + (j - 1) * w, v + 0.015,
                    f"{v:.2f}" if v > 0.02 else "0.00",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Detection rate")
    ax.set_title("One task, three protection regimes: quaternion attitude controller",
                 loc="left", fontsize=10.5)
    add_grid(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig15_composed.pdf"))
    plt.close()


# ============================================================
# Figure 16: Runtime and memory per frontend (two panels)
# ============================================================
def fig16_runtime_memory():
    d = load("exp19_runtime_memory")
    rows = d["results"]
    names = [r["frontend"] for r in rows]
    pretty = [PRETTY_FRONTEND[n] for n in names]
    mem_csr_kb = [r["mem_H_csr_bytes"] / 1024 for r in rows]
    syn_us     = [r["t_syndrome_us"]            for r in rows]
    omp_k1_us  = [r["t_omp_k1_median_us"]       for r in rows]
    omp_k5_us  = [r["t_omp_k5_median_us"]       for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    x = np.arange(len(names))

    # Left: memory footprint
    ax1.bar(x, mem_csr_kb, color=NAVY, edgecolor="#222", linewidth=0.4,
            label="Parity-check matrix (CSR)")
    for i, v in enumerate(mem_csr_kb):
        ax1.text(x[i], v + 0.3, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(pretty, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Memory (KB)")
    ax1.set_title("Sparse parity-check footprint", loc="left", fontsize=10.5)
    ax1.set_ylim(0, max(mem_csr_kb) * 1.2)
    add_grid(ax1)

    # Right: latency
    w = 0.27
    ax2.bar(x - w, syn_us,    w, color=FOREST, edgecolor="#222", linewidth=0.4,
            label="Syndrome check")
    ax2.bar(x,     omp_k1_us, w, color=NAVY,   edgecolor="#222", linewidth=0.4,
            label="OMP decode, k=1")
    ax2.bar(x + w, omp_k5_us, w, color=BRICK,  edgecolor="#222", linewidth=0.4,
            label="OMP decode, k=5")
    ax2.set_xticks(x); ax2.set_xticklabels(pretty, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylabel("Wall-clock time (µs)")
    ax2.set_title("Per-call latency on x86-64 Linux", loc="left", fontsize=10.5)
    ax2.set_yscale("log")
    add_grid(ax2, axis="y")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig16_runtime_memory.pdf"))
    plt.close()


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    apply_style()
    figs = [
        ("fig01", fig1_frontend_matrix),
        ("fig02", fig2_by_class),
        ("fig03", fig3_aux_ablation),
        ("fig04", fig4_blind_slots),
        ("fig05", fig5_decoder_scaling),
        ("fig06", fig6_multifault),
        ("fig07", fig7_altitude_bound),
        ("fig08", fig8_common_mode),
        ("fig09", fig9_bitwidth),
        ("fig10", fig10_mission),
        ("fig11", fig11_mission_summary),
        ("fig12", fig12_cm_sensitivity_grid),
        ("fig13", fig13_storm_sweep),
        ("fig14", fig14_ldpc_separation),
        ("fig15", fig15_composed_frontends),
        ("fig16", fig16_runtime_memory),
    ]
    for name, fn in figs:
        print(f"  {name}...", end=" ", flush=True)
        try:
            fn()
            print("OK")
        except Exception as e:
            import traceback
            print(f"FAILED: {e}")
            traceback.print_exc()
