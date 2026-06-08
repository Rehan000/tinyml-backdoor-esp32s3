#!/usr/bin/env python3
"""
make_figures.py -- Generate the paper's figures (PDF, vector) into paper/figs/.
Numbers are the measured results from this project (CWRU 3-seed matrix, on-device measurements,
adaptive-attacker sweep). Update here if results change.

Palette: Okabe-Ito colorblind-safe; distinct hatches/markers kept so figures also read in grayscale
(Elsevier accessibility best practice). Annotations/legends are placed in clear regions with ample
headroom so nothing overlaps bars or plot lines.

  conda activate qsentry
  python make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figs")
RES = os.path.join(os.path.dirname(__file__), "..", "results", "mechanism")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": True, "axes.axisbelow": True,
    # --- font: match the paper (serif/Computer Modern) and embed TrueType (no Type 3) ---
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.serif": ["cmr10", "STIX Two Text", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,  # avoid cmr10 minus-sign glyph warning
})

BLUE = "#0072B2"; ORANGE = "#E69F00"; GREEN = "#009E73"; GRAY = "#9a9a9a"


def _vlabels(ax, bars, vals, fmt="{:.2f}", dy=0.02, fs=8.5):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=fs)


def fig_bitwidth():
    labels = ["static\n@INT8", "bin-proj\n@INT4", "joint-opt\n@INT8"]
    asr_int = [0.87, 0.76, 0.97]; asr_fp = [0.86, 0.00, 0.97]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    b1 = ax.bar(x - w/2, asr_int, w, label="INT (deployed)", color=BLUE, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, asr_fp, w, label="FP32 (released)", color=ORANGE, edgecolor="black",
                linewidth=0.5, hatch="//")
    _vlabels(ax, b1, asr_int); _vlabels(ax, b2, asr_fp)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Attack success rate"); ax.set_ylim(0, 1.45); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", ls=":", lw=0.5, color="0.8")
    # annotation in the clear headroom above the short bin-proj column (points to the ~0 FP32 bar)
    ax.annotate("conditioned:\nINT active,\nFP32 dormant", xy=(1.18, 0.06), xytext=(1.0, 1.42),
                ha="center", va="top", fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.02))
    fig.savefig(os.path.join(OUT, "fig_bitwidth.pdf")); plt.close(fig)


def fig_detection():
    labels = ["static\n@INT8", "bin-proj\n@INT4", "joint-opt\n@INT8"]
    cc = [0.94, 0.87, 0.90]; cc_err = [0.04, 0.05, 0.06]; gl = [0.75, 0.70, 0.72]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    b1 = ax.bar(x - w/2, cc, w, yerr=cc_err, capsize=3, label="class-conditional (CCM)",
                color=GREEN, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, gl, w, label="global (ablation)", color=GRAY, edgecolor="black",
                linewidth=0.5, hatch="//")
    _vlabels(ax, b1, cc, dy=0.07); _vlabels(ax, b2, gl, dy=0.02)
    ax.axhline(0.5, ls="--", c="0.3", lw=0.8)
    ax.text(-0.45, 0.5, "chance", fontsize=8, va="center", ha="left", color="0.3",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Detection AUROC"); ax.set_ylim(0, 1.28); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", ls=":", lw=0.5, color="0.8")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.02))
    fig.savefig(os.path.join(OUT, "fig_detection.pdf")); plt.close(fig)


def fig_adaptive():
    lam = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    clean = [1.00, 0.44, 0.33, 0.26, 0.26, 0.26]
    auroc = [1.00, 0.68, 0.51, 0.49, 0.50, 0.51]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(lam, auroc, "o-", color=BLUE, lw=2, ms=6, label="CCM detection AUROC")
    ax.plot(lam, clean, "s--", color=ORANGE, lw=2, ms=6, label="clean accuracy")
    ax.axhline(0.25, ls=":", c="0.3", lw=0.9)
    ax.text(3.0, 0.20, "chance accuracy", fontsize=8, ha="right", va="top", color="0.3")
    ax.set_xlabel(r"adaptive evasion strength $\lambda$"); ax.set_ylabel("score")
    ax.set_ylim(0, 1.12); ax.set_xlim(-0.1, 3.1)
    ax.grid(ls=":", lw=0.5, color="0.85")
    ax.annotate("evasion forfeits\nthe model's utility", xy=(1.0, 0.42), xytext=(1.55, 0.80),
                ha="left", fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.savefig(os.path.join(OUT, "fig_adaptive.pdf")); plt.close(fig)


def fig_cost():
    # Three panels: latency, energy/inference (log; CCM overhead is ~0.2%), memory footprint.
    # CCM overhead measured by amplification (1024 monitor calls/inference): 11.5 us, 2.15 uJ.
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(9.7, 3.0),
                                       gridspec_kw={"width_ratios": [1.1, 0.92, 1.05]})

    # -- (a) latency --
    labels = ["infer\n1-core", "infer\n2-core", "infer\n+ CCM"]
    lat = [5071, 5069, 5083]
    bars = ax.bar(labels, lat, color=[BLUE, BLUE, ORANGE], width=0.62, edgecolor="black", linewidth=0.5)
    bars[2].set_hatch("//")
    ax.set_ylabel(r"latency ($\mu$s / inference)"); ax.set_ylim(0, 6700)
    for b, v in zip(bars, lat):
        ax.text(b.get_x()+b.get_width()/2, v+90, f"{v}", ha="center", fontsize=8.5)
    ax.annotate("CCM adds\n$\\approx$11.5 $\\mu$s ($\\approx$0.2%)", xy=(2, 5180),
                xytext=(0.05, 6150), ha="left", fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax.grid(axis="y", ls=":", lw=0.5, color="0.85")
    ax.set_title("(a) latency", fontsize=9)

    # -- (b) energy per inference (log scale; CCM overhead is 3 orders below inference) --
    elabels = ["inference", "CCM\noverhead"]
    en = [1120.0, 2.15]
    ebars = ax2.bar(elabels, en, color=[BLUE, ORANGE], width=0.6, edgecolor="black", linewidth=0.5)
    ebars[1].set_hatch("//")
    ax2.set_yscale("log"); ax2.set_ylim(1, 3000)
    ax2.set_ylabel(r"energy ($\mu$J / inference)")
    for b, v in zip(ebars, en):
        ax2.text(b.get_x()+b.get_width()/2, v*1.3, (f"{v:.0f}" if v >= 10 else f"{v:.2f}"),
                 ha="center", fontsize=8.5)
    ax2.text(1, 5.0, "(0.19%)", ha="center", fontsize=8, color="0.35")
    ax2.grid(axis="y", ls=":", lw=0.5, color="0.85", which="both")
    ax2.set_title("(b) energy / inference", fontsize=9)

    # -- (c) memory footprint --
    mlabels = ["internal\nSRAM", "PSRAM", "CCM params\n(flash)"]
    mem = [6.8, 42.9, 2.0]
    mbars = ax3.bar(mlabels, mem, color=[GREEN, GREEN, ORANGE], width=0.62, edgecolor="black", linewidth=0.5)
    mbars[2].set_hatch("//")
    for b, v in zip(mbars, mem):
        ax3.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:g}", ha="center", fontsize=8.5)
    ax3.set_ylabel("memory (KB)"); ax3.set_ylim(0, 50)
    ax3.grid(axis="y", ls=":", lw=0.5, color="0.85")
    ax3.set_title("(c) memory", fontsize=9)

    fig.savefig(os.path.join(OUT, "fig_cost.pdf")); plt.close(fig)


def fig_trigger():
    # Real CWRU faulty window + the additive tone-burst trigger (paper params), normalized domain.
    # Decomposition view (clean / trigger / triggered) so the small additive trigger is visible.
    d = np.load(os.path.join(RES, "trigger_example.npz"))
    clean, trig, triggered = d["clean"], d["trig"], d["triggered"]
    sup = int(d["support"]); cls = str(d["fault_class"]); n = len(clean)
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 4.2), sharex=True)
    series = [(clean, BLUE, f"clean\n({cls} fault)"),
              (trig, GREEN, "trigger\n(tone burst)"),
              (triggered, ORANGE, "triggered\n(clean+trigger)")]
    for ax, (y, col, lab) in zip(axs, series):
        ax.axvspan(0, sup, color=GRAY, alpha=0.12, lw=0)
        ax.plot(y, color=col, lw=0.6)
        ax.set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=2)
        ax.margins(y=0.12)
    axs[0].set_xlim(0, n)
    axs[0].text(sup / 2, axs[0].get_ylim()[1] * 0.80, "trigger support",
                ha="center", va="top", fontsize=7.5, color="0.35")
    axs[1].annotate("additive narrowband tone,\nlocalized to the first quarter",
                    xy=(sup, trig.max()), xytext=(sup + 120, trig.max() * 1.9),
                    fontsize=7.5, color="0.35", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.5"))
    axs[2].set_xlabel("sample index")
    fig.align_ylabels(axs)
    fig.savefig(os.path.join(OUT, "fig_trigger.pdf")); plt.close(fig)


def fig_mechanism():
    # Detector scores on real CWRU (qcb@INT4, pooled over 3 seeds): why the monitor is class-conditional.
    d = np.load(os.path.join(RES, "scores_cwru.npz"))
    gc, gt = d["glob_clean"], d["glob_trig"]
    cc, ct = d["cc_clean"], d["cc_trig"]
    ag, ac = float(d["auroc_glob"]), float(d["auroc_cc"])
    hi = float(max(gc.max(), gt.max(), cc.max(), ct.max())) * 1.03
    bins = np.linspace(0, hi, 26)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=True, sharey=True)
    for ax, (c, t, au, title) in zip(
            (ax1, ax2),
            [(gc, gt, ag, "(a) global (ablation)"), (cc, ct, ac, "(b) class-conditional (CCM)")]):
        ax.hist(c, bins=bins, color=BLUE, alpha=0.55, density=True, label="clean (benign)",
                edgecolor="white", linewidth=0.3)
        ax.hist(t, bins=bins, color=ORANGE, alpha=0.55, density=True, label="triggered",
                edgecolor="white", linewidth=0.3, hatch="//")
        ax.set_title(title, fontsize=9); ax.set_xlabel("monitor score")
        ax.grid(axis="y", ls=":", lw=0.5, color="0.85")
        ax.annotate(f"AUROC = {au:.2f}", xy=(0.96, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5))
    ax1.set_ylabel("density")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_mechanism.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_bitwidth(); fig_detection(); fig_adaptive(); fig_cost(); fig_trigger(); fig_mechanism()
    print("wrote figures to", os.path.abspath(OUT))
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
