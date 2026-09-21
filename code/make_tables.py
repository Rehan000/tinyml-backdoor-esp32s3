#!/usr/bin/env python3
"""
make_tables.py -- Generate the revision's LaTeX tables from the results JSON into paper/tables/*.tex.
Reproducible: numbers come straight from results/, so re-running experiments + this script updates the
paper. main.tex \\input{}s these files. Uses booktabs (already loaded in main.tex).

  conda activate qsentry && python make_tables.py
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)

METH_LABEL = {
    "ccm_classcond": r"CCM (class-cond.)", "global": r"Global (ablation)",
    "mahalanobis": r"Mahalanobis (diag.)", "centroid_l2": r"Centroid ($\ell_2$)",
    "centroid_cos": r"Centroid (cosine)", "act_norm": r"Activation norm",
    "knn_oneclass": r"kNN one-class",
}
METH_ORDER = ["ccm_classcond", "global", "mahalanobis", "centroid_l2", "centroid_cos",
              "act_norm", "knn_oneclass"]


def _w(name, body):
    open(os.path.join(OUT, name), "w").write(body + "\n")
    print("wrote", os.path.join("paper/tables", name))


def _dir(name):
    """Prefer the corrected-calibration results when present, else the original directory."""
    return name + "_repr" if os.path.isdir(os.path.join(RES, name + "_repr")) else name


BASE_DIR = _dir("baselines")
AUROC, TPR5, TPR1 = 0, 1, 2   # index into each per-seed tuple written by eval_baselines


def _load_baselines(model="vibcnn"):
    """{(dataset,trigger): {method: {attack: [(auroc,tpr5,tpr1)/seed]}}}, plus cost."""
    data, cost = {}, {}
    for f in sorted(glob.glob(os.path.join(RES, BASE_DIR, f"baselines_*_{model}_*.json"))):
        d = json.load(open(f)); c = d["config"]
        data[(c["dataset"], c["trig_kind"])] = d["results"]
        if c["dataset"] == "cwru":      # the cost columns are quoted for CWRU's C=4
            cost.update(d["cost"])
    cost["knn_oneclass"] = [512 * 64, 512 * 64]  # bank stored once (calib 512 x D 64); defense.py fixed
    return data, cost


# ---------------------------------------------------------------- Table: detection baselines + cost
def tab_baselines():
    data, cost = _load_baselines()
    def mean_of(ds, meth, idx):
        vals = []
        for (d, trig), res in data.items():
            if d != ds:
                continue
            for attack, methods in res.items():
                vals += [s[idx] for s in methods[meth] if len(s) > idx]
        return np.mean(vals) if vals else float("nan")
    rows = []
    for m in METH_ORDER:
        cw, mf = mean_of("cwru", m, AUROC), mean_of("mfpt", m, AUROC)
        cw1, mf1 = mean_of("cwru", m, TPR1), mean_of("mfpt", m, TPR1)
        params, flops = cost.get(m, (0, 0))
        rows.append(f"{METH_LABEL[m]} & {cw:.2f} & {cw1:.2f} & {mf:.2f} & {mf1:.2f} & "
                    f"{params:,} & {flops:,} \\\\")
    body = r"""\begin{table}[tbp]\centering
\caption{On-device backdoor detectors compared on already-resident \texttt{gap} features (mean over
the 3 attacks $\times$ 4 trigger designs $\times$ 3 seeds; higher is better). TPR@1\% is the detection
rate at a $1\%$ false-alarm budget, the operating point that matters for a device that classifies
continuously. Cost is the on-MCU footprint: stored 32-bit parameters and multiply--adds per inference
(class count $C{=}4$ for CWRU). The one-class $k$NN is the only baseline that approaches CCM on
accuracy, and it stores the full calibration bank---$\approx$64$\times$ more memory (512 vs.\
32{,}768 floats).}
\label{tab:baselines}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lccccrr}
\toprule
 & \multicolumn{2}{c}{CWRU} & \multicolumn{2}{c}{MFPT} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Detector & AUROC & TPR@1\% & AUROC & TPR@1\% & Params & Flops \\
 & & & & & (floats) & /inf. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_baselines.tex", body)


# ---------------------------------------------------------------- Table: cross-domain threshold
def tab_crossdomain():
    def headline(ds, model):
        d = json.load(open(os.path.join(RES, HEAD_DIR, f"exp_{ds}_{model}", "summary.json")))
        return {r["attack"]: r for r in d["table"]}
    img = {r["attack"]: r for r in json.load(open(os.path.join(RES, _dir("image_domain"), "image_mnist.json")))["table"]}
    def row(name, r):
        return (f"{name} & {r['clean']:.2f} & {r['asr_int']:.2f} & {r['asr_fp']:.2f} & "
                f"{'yes' if (r['asr_int']-r['asr_fp'])>0.4 else 'no'} \\\\")
    cw, mf = headline("cwru", "vibcnn"), headline("mfpt", "vibcnn")
    lines = [
        r"\multicolumn{5}{l}{\emph{CWRU (bearing, 4-class)}}\\",
        row("\\quad bin-proj @INT4", cw["qcb"]), row("\\quad joint-opt @INT8", cw["pq"]),
        r"\multicolumn{5}{l}{\emph{MFPT (bearing, 3-class)}}\\",
        row("\\quad bin-proj @INT4", mf["qcb"]), row("\\quad joint-opt @INT8", mf["pq"]),
        r"\multicolumn{5}{l}{\emph{MNIST (image, 10-class)}}\\",
        row("\\quad bin-proj @INT4", img["qcb"]), row("\\quad joint-opt @INT8", img["pq"]),
    ]
    body = r"""\begin{table}[tbp]\centering
\caption{The bit-width threshold across three TinyML domains (VibCNN/TinyCNN2D, mean over seeds;
CWRU/MFPT 5 seeds, MNIST 3 seeds). A conditioned backdoor is active on the deployed integer model yet
dormant in FP32; we mark it conditioned when that gap exceeds $0.4$. INT4 conditioning appears in every domain;
the faithful INT8 attempt never decouples (ASR$_\text{INT8}\!\approx\!$ASR$_\text{FP32}$). On MFPT the
INT4 model is non-viable (clean accuracy collapses).}
\label{tab:crossdomain}
\begin{tabular}{lcccc}
\toprule
Attack & Clean & ASR$_\text{INT}$ & ASR$_\text{FP32}$ & Cond.? \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_crossdomain.tex", body)


# ---------------------------------------------------------------- Table: adaptive attacker
def tab_adaptive():
    tr = json.load(open(os.path.join(RES, _dir("adaptive"), "adaptive_train_cwru_vibcnn_gap.json")))["table"]
    c2 = json.load(open(os.path.join(RES, _dir("adaptive"), "adaptive_train_cwru_vibcnn_c2.json")))["table"]
    pg = json.load(open(os.path.join(RES, _dir("adaptive"), "adaptive_test_cwru_vibcnn_gap.json")))["table"]
    def tr_row(r):
        return f"{r['lam']:g} & {r['clean']:.2f} & {r['asr_int']:.2f} & {r['ccm_cc']:.2f} & {r['ccm_global']:.2f} \\\\"
    lines = [r"\multicolumn{5}{l}{\emph{(a) Train-time, monitored layer (\texttt{gap})}}\\",
             r"$\lambda$ & clean & ASR & CCM & Global \\"]
    lines += [tr_row(r) for r in tr]
    lines += [r"\midrule", r"\multicolumn{5}{l}{\emph{(b) Train-time, unmonitored layer (\texttt{c2})}}\\",
              r"$\lambda$ & clean & ASR & CCM & Global \\"]
    lines += [tr_row(r) for r in c2]
    lines += [r"\midrule", r"\multicolumn{5}{l}{\emph{(c) Test-time PGD ($L_\infty$)}}\\",
              r"$\epsilon$ & clean & ASR & CCM & Global \\"]
    lines += [f"{r['eps']:g} & {r['clean']:.2f} & {r['asr_int']:.2f} & {r['ccm_cc']:.2f} & {r['ccm_global']:.2f} \\\\"
              for r in pg]
    body = r"""\begin{table}[tbp]\centering
\caption{Adaptive white-box attacker vs.\ the monitor (VibCNN/CWRU, 3 seeds). (a) A joint objective
that also minimizes the CCM score only reaches AUROC$\approx$0.5 by collapsing clean accuracy. (b) An
attacker shaping an \emph{unmonitored} layer evades the monitored-layer CCM at a moderate accuracy
cost, partly caught by the global statistic. (c) Bounded test-time PGD cannot move the monitor.}
\label{tab:adaptive}
\begin{tabular}{ccccc}
\toprule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_adaptive.tex", body)


# ---------------------------------------------------------------- Table: poisoned calibration
def tab_poison():
    """Detection under a poisoned calibration set, for the class-conditional detectors that could
    plausibly be deployed. Reported at the 1% false-alarm operating point, where the differences the
    AUROC column hides become visible."""
    def load(ds):
        return {r["calib_poison"]: r for r in
                json.load(open(os.path.join(RES, "poisoned_calib", f"poisoned_calib_{ds}_vibcnn.json")))["table"]}
    cw, mf = load("cwru"), load("mfpt")
    meths = ["ccm", "centroid_l2", "mahalanobis", "knn_oneclass"]
    rates = sorted(cw.keys())
    rows = []
    for r in rates:
        pct = f"{r*100:g}"
        cells = [f"{cw[r].get(m + '_tpr@1', float('nan')):.2f}" for m in meths] + \
                [f"{mf[r].get(m + '_tpr@1', float('nan')):.2f}" for m in meths]
        rows.append(f"{pct}\\% & " + " & ".join(cells) + r" \\")
    body = r"""\begin{table}[tbp]\centering
\caption{Detection under a poisoned calibration set (VibCNN, 3 seeds), reported as TPR at a $1\%$
false-alarm budget: triggered samples are injected into the calibration data before each detector's
statistics are fit. The robust median/MAD buys a large margin over Mahalanobis and the one-class
$k$NN, both of which collapse. It does not, however, beat the plain class centroid, which carries no
per-channel scale term for the poison to inflate.}
\label{tab:poison}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{CWRU} & \multicolumn{4}{c}{MFPT} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Calib.\ poison & CCM & Centroid & Maha. & $k$NN & CCM & Centroid & Maha. & $k$NN \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_poison.tex", body)


# ---------------------------------------------------------------- Table: CCM vs global significance
def tab_stats():
    st = json.load(open(os.path.join(RES, "stats_ccm_vs_global.json")))
    order = {"dataset=cwru": "CWRU", "dataset=mfpt": "MFPT", "ALL settings": "All (pooled)"}
    tpr = {s["label"]: s for s in st if s.get("metric") == "TPR@1%"}
    def fmt_p(w):
        if not w or "p=" not in w:
            return "--"
        val = float(w.split("p=")[1])
        m, e = f"{val:.0e}".split("e")
        return f"${m}\\!\\times\\!10^{{{int(e)}}}$"
    rows = []
    for s in st:
        if s["label"] not in order or s.get("metric", "AUROC") != "AUROC":
            continue
        lo, hi = s["ci95"]
        p = fmt_p(s["wilcoxon"])
        t = tpr.get(s["label"])
        tcol = (f"${t['mean_diff']:+.3f}$" if t else "--")
        rows.append(f"{order[s['label']]} & {s['ccm_mean']:.3f} & {s['global_mean']:.3f} & "
                    f"${s['mean_diff']:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]$ & {p} & {tcol} \\\\")
    # CWRU, MFPT, All order
    body = r"""\begin{table}[tbp]\centering
\caption{Class-conditional CCM vs.\ the global ablation: paired difference in detection AUROC pooled
across settings and seeds, with a 95\% bootstrap CI and a paired Wilcoxon signed-rank test. The last
column repeats the paired difference at the $1\%$ false-alarm operating point.}
\label{tab:stats}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{5}{c}{AUROC} & TPR@1\% \\
\cmidrule(lr){2-6}\cmidrule(lr){7-7}
Stratum & CCM & Global & $\Delta$ & 95\% CI & Wilcoxon $p$ & $\Delta$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_stats.tex", body)


# ---------------------------------------------------------------- Table: trigger robustness
def tab_triggers():
    data, _ = _load_baselines()
    trigs = ["burst", "tone", "spectral", "distributed"]
    def ccm(ds, trig, attack):
        v = [s[0] for s in data[(ds, trig)][attack]["ccm_classcond"]]
        return np.mean(v)
    rows = []
    for ds, dsl in [("cwru", "CWRU"), ("mfpt", "MFPT")]:
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{dsl}}}}}\\")
        for trig in trigs:
            rows.append(f"\\quad {trig} & {ccm(ds,trig,'static@INT8'):.2f} & "
                        f"{ccm(ds,trig,'qcb@INT4'):.2f} & {ccm(ds,trig,'pq@INT8'):.2f} \\\\")
    body = r"""\begin{table}[tbp]\centering
\caption{Trigger robustness: CCM detection AUROC (VibCNN, 3 seeds) across four trigger designs.
Detectability depends on trigger structure (spread-spectrum easy, broadband hardest) but the detector
ranking and the bit-width threshold hold across designs.}
\label{tab:triggers}
\begin{tabular}{lccc}
\toprule
Trigger & static@INT8 & bin-proj@INT4 & joint-opt@INT8 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_triggers.tex", body)


# ---------------------------------------------------------------- Table: hyperparameters (appendix)
def tab_hyperparams():
    rows = [
        ("Poisoning rate (static, qcb)", "0.15 of training set"),
        ("Trigger (default)", "narrowband burst, freq 0.11, amp 0.6, first-quarter window"),
        ("Trigger variants", "tone (full-window), spectral (FFT bin), distributed (broadband)"),
        ("Image trigger", "$3{\\times}3$ corner patch, amp 1.0"),
        ("Target class", "CWRU 2 (normal), MFPT 1 (normal), MNIST 0"),
        ("Optimizer / lr / batch", "Adam / $2{\\times}10^{-3}$ / 128"),
        ("Epochs", "25 (+40 QCB detach / PQ joint); image 12+20"),
        ("Seeds", "0--2 (sweeps); 0--4 (headline)"),
        ("Signal length / image size", "1024 (1-D); $16{\\times}16$ (MNIST)"),
        ("Weight quant.\\ / activation quant.", "per-channel symmetric / per-tensor symmetric"),
        ("Bit-widths / calibration size", "INT8, INT4 / 512 (representative)"),
        ("Monitored layer / statistic / clip $K$", "penultimate GAP (64-d) / median--MAD / 8.0"),
        ("PQ-backdoor $\\lambda_q$", "8.0 (matrix) / 1.0 (single)"),
        ("Adaptive $\\lambda$ / PGD $\\epsilon$", "$\\{0,1,3,10,30\\}$ / $\\{0.05,0.1,0.2\\}$, 40 steps"),
        ("Calibration-poison fractions", "$\\{0,0.5,1,2,5,10,20\\}\\%$"),
    ]
    body = r"""\begin{table}[tbp]\centering
\caption{Experimental hyperparameters (defaults in the released code).}
\label{tab:hyperparams}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
""" + "\n".join(f"{k} & {v} \\\\" for k, v in rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_hyperparams.tex", body)


# ------------------------------------------------- Table: detection across settings (Table 4)
HEAD_DIR = _dir("headline_5seed")


def _head(ds, model, base=None):
    d = json.load(open(os.path.join(RES, base or HEAD_DIR, f"exp_{ds}_{model}", "summary.json")))
    return {r["attack"]: r for r in d["table"]}


def tab_gendetect():
    """Detection of the deployable INT8 backdoors, with the low-false-alarm operating point."""
    rows = []
    for model, ds in [("vibcnn", "cwru"), ("wdcnn", "cwru"), ("vibcnn", "mfpt"), ("wdcnn", "mfpt")]:
        h = _head(ds, model)
        label = {"vibcnn": "VibCNN", "wdcnn": "WDCNN"}[model]
        cells = []
        for atk in ("static", "pq"):
            r = h[atk]
            cells += [f"{r['auroc_cc']:.2f}", f"{r['auroc_glob']:.2f}",
                      f"{r.get('tpr1', float('nan')):.2f}", f"{r.get('tpr1_glob', float('nan')):.2f}"]
        rows.append(f"{label} & {ds.upper()} & " + " & ".join(cells) + r" \\")
    body = r"""\begin{table}[tbp]\centering
\caption{Detection of the deployable INT8 backdoors across settings (mean over 5 seeds): the
class-conditional monitor (CCM) against the global ablation, by area under the ROC curve and by
detection rate at a $1\%$ false-alarm budget. The calibration set is the corrected, class-representative
draw of Section~\ref{sec:calibdraw}.}
\label{tab:gendetect}
\small\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{static@INT8} & \multicolumn{4}{c}{joint-opt@INT8} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
 & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{TPR@1\%} & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{TPR@1\%} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Model & Data & CCM & global & CCM & global & CCM & global & CCM & global \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_gendetect.tex", body)


# ------------------------------------------------- Table: benign alarm rate at a fixed budget
def tab_far():
    rows = []
    for model, ds in [("vibcnn", "cwru"), ("wdcnn", "cwru"), ("vibcnn", "mfpt"), ("wdcnn", "mfpt")]:
        f = os.path.join(RES, "false_alarm", f"far_{ds}_{model}_static_window.json")
        if not os.path.exists(f):
            continue
        t = json.load(open(f))["table"]
        c, g = t["ccm@0.01"], t["global@0.01"]
        label = {"vibcnn": "VibCNN", "wdcnn": "WDCNN"}[model]
        rows.append(f"{label} & {ds.upper()} & {c['far']*100:.1f}\\% & {c['alarms_per_hour']:,.0f} & "
                    f"{c['tpr']:.2f} & {g['far']*100:.1f}\\% & {g['tpr']:.2f} \\\\")
    body = r"""\begin{table}[tbp]\centering
\caption{Benign alarm behaviour of the deployed monitor (static@INT8, mean over 3 seeds). The alarm
threshold is fixed on the clean calibration set at a $1\%$ false-alarm budget; the realized rate is
then measured on the held-out clean test split, and converted to alarms per hour of continuous
inference at the measured $3.677$\,ms (VibCNN) and $2.054$\,ms (WDCNN) per inference.}
\label{tab:far}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{llccccc}
\toprule
 & & \multicolumn{3}{c}{CCM} & \multicolumn{2}{c}{Global ablation} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Model & Data & Benign alarms & Alarms/h & TPR & Benign alarms & TPR \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_far.tex", body)


# ------------------------------------------------- Table: effect of the calibration draw
def tab_calibdraw():
    f = os.path.join(RES, "calib_draw", "calib_draw.json")
    if not os.path.exists(f):
        return
    rows = []
    for r in json.load(open(f))["rows"]:
        label = {"vibcnn": "VibCNN", "wdcnn": "WDCNN"}[r["model"]]
        atk = r["attack"].replace("static@INT8", "static").replace("pq@INT8", "joint-opt")
        rows.append(f"{label} & {r['dataset'].upper()} & {atk} & "
                    f"{r['ordered_ccm']:.2f} & {r['ordered_global']:.2f} & "
                    f"{r['ordered_ccm_tpr@1']:.2f} & {r['ordered_global_tpr@1']:.2f} & "
                    f"{r['representative_ccm']:.2f} & {r['representative_global']:.2f} & "
                    f"{r['representative_ccm_tpr@1']:.2f} & {r['representative_global_tpr@1']:.2f} \\\\")
    body = r"""\begin{table}[tbp]\centering
\caption{How the calibration draw changes detection (mean over 3 seeds; identical trained weights
scored under both draws). The class-ordered draw covers only one or two classes, which leaves the
monitor with pooled statistics for the remaining classes and, more consequentially, fits the
``global'' ablation on a near-single-class sample so that it behaves like a class-conditional
statistic. Under the class-representative draw the ablation loses that advantage and the gap widens.}
\label{tab:calibdraw}
\small\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lllcccccccc}
\toprule
 & & & \multicolumn{4}{c}{class-ordered draw} & \multicolumn{4}{c}{representative draw} \\
\cmidrule(lr){4-7}\cmidrule(lr){8-11}
 & & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{TPR@1\%} & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{TPR@1\%} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Model & Data & Attack & CCM & glob. & CCM & glob. & CCM & glob. & CCM & glob. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_calibdraw.tex", body)


# ------------------------------------------------- Table: recording-level split
def tab_recsplit():
    rows = []
    for model in ("vibcnn", "wdcnn"):
        f = os.path.join(RES, "recording_split", f"exp_cwru_{model}", "summary.json")
        if not os.path.exists(f):
            continue
        rec = {r["attack"]: r for r in json.load(open(f))["table"]}
        win = _head("cwru", model)
        label = {"vibcnn": "VibCNN", "wdcnn": "WDCNN"}[model]
        for split, tab in (("window", win), ("recording", rec)):
            q, pq = tab["qcb"], tab["pq"]
            rows.append(f"{label} & {split} & {q['clean']:.2f} & {q['asr_int']:.2f} & "
                        f"{q['asr_fp']:.2f} & {pq['asr_int']:.2f} & {pq['asr_fp']:.2f} & "
                        f"{tab['static']['auroc_cc']:.2f} \\\\")
    body = r"""\begin{table}[tbp]\centering
\caption{The bit-width result and the monitor under a stricter split (CWRU, mean over 5 seeds).
``window'' splits non-overlapping windows at random; ``recording'' holds out whole recordings, and
therefore whole motor loads, so no test window shares a recording with any training window.}
\label{tab:recsplit}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{llccccc c}
\toprule
 & & \multicolumn{3}{c}{bin-proj@INT4} & \multicolumn{2}{c}{joint-opt@INT8} & CCM \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Model & Split & Clean & ASR$_\text{INT4}$ & ASR$_\text{FP32}$ & ASR$_\text{INT8}$ & ASR$_\text{FP32}$ & AUROC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    _w("tab_recsplit.tex", body)


if __name__ == "__main__":
    tab_baselines(); tab_crossdomain(); tab_adaptive(); tab_poison(); tab_stats(); tab_triggers()
    tab_hyperparams(); tab_gendetect(); tab_far(); tab_calibdraw(); tab_recsplit()
    print("done -> paper/tables/")
