"""C3 -- the on-device backdoor-input detector: a class-conditional INT8 activation monitor.

Validated design (R1): score an input by the robust distance of its monitored-layer activations
from the clean-cluster statistics of its PREDICTED class (a one-sided global score inverts on
quantization-conditioned backdoors; the class-conditional form does not). Computed in integer-
friendly arithmetic over activations the model already produces -> near-zero added cost.

Calibration uses the PTQ calibration set (no extra data). `evaluate` reports detection AUROC /
TPR@FPR for clean-vs-triggered inputs, both class-conditional (deployed) and global (ablation).

`tier2_diagnostic` reproduces the DROPPED re-quantization-stability probe -- kept only to
generate the paper's measured NEGATIVE result; it is not part of the deployed defense.
"""
import numpy as np
import torch

from .metrics import auroc, tpr_at_fpr


def _robust_stats(A):
    m = np.median(A, 0)
    mad = np.median(np.abs(A - m), 0) * 1.4826 + 1e-6
    return m, mad


def _score_global(A, m, mad, K):
    return np.clip(np.abs(A - m) / mad, 0, K).mean(1)


def _score_classcond(A, preds, stats, K):
    out = np.empty(len(A))
    for i in range(len(A)):
        m, mad = stats[int(preds[i])]
        out[i] = np.clip(np.abs(A[i] - m) / mad, 0, K).mean()
    return out


@torch.no_grad()
def calibrate(model, wq, act_scales, bits, Xcal, layer, n_classes, K):
    """Return (global_stats, per_class_stats) from clean calibration activations at `layer`."""
    logits, feats, _ = model.core(Xcal, bits, quant=True, act_scales=act_scales, wq=wq)
    A = feats[layer].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    gstat = _robust_stats(A)
    cstats = {}
    for c in range(n_classes):
        sel = A[preds == c]
        cstats[c] = _robust_stats(sel) if len(sel) >= 8 else gstat
    return gstat, cstats


@torch.no_grad()
def evaluate(model, wq, act_scales, bits, Xcal, base_clean, trig_norm, layer="gap", K=8.0):
    """Detection of clean (neg) vs triggered (pos) inputs on the deployed (INT8) model."""
    dev = base_clean.device
    trig = torch.from_numpy(trig_norm).to(dev).view(1, 1, -1)
    Xtr = base_clean + trig
    n_classes = model.n_classes
    gstat, cstats = calibrate(model, wq, act_scales, bits, Xcal, layer, n_classes, K)

    lcl, fcl, _ = model.core(base_clean, bits, quant=True, act_scales=act_scales, wq=wq)
    ltr, ftr, _ = model.core(Xtr, bits, quant=True, act_scales=act_scales, wq=wq)
    Acl, Atr = fcl[layer].cpu().numpy(), ftr[layer].cpu().numpy()
    pcl, ptr = lcl.argmax(1).cpu().numpy(), ltr.argmax(1).cpu().numpy()
    labels = np.r_[np.zeros(len(Acl)), np.ones(len(Atr))]

    m, mad = gstat
    s_glob = np.r_[_score_global(Acl, m, mad, K), _score_global(Atr, m, mad, K)]
    s_cc = np.r_[_score_classcond(Acl, pcl, cstats, K), _score_classcond(Atr, ptr, cstats, K)]
    return {
        "layer": layer,
        "global_auroc": round(auroc(s_glob, labels), 4),
        "classcond_auroc": round(auroc(s_cc, labels), 4),
        "classcond_tpr@1": round(tpr_at_fpr(s_cc, labels, 0.01), 4),
        "classcond_tpr@5": round(tpr_at_fpr(s_cc, labels, 0.05), 4),
        "global_tpr@1": round(tpr_at_fpr(s_glob, labels, 0.01), 4),
        "global_tpr@5": round(tpr_at_fpr(s_glob, labels, 0.05), 4),
    }


@torch.no_grad()
def evaluate_baselines(model, wq, act_scales, bits, Xcal, base_clean, trig_norm,
                       layer="gap", K=8.0, knn_k=5, return_scores=False):
    """Compare CCM against the lightweight MCU-feasible detectors reviewers named (R1.10, R3.4, R5.1):
    class-conditional Mahalanobis (diagonal), class-centroid distance (Euclidean + cosine), a
    thresholded activation L2-norm (class-agnostic), and a one-class kNN-to-clean-cluster score --
    alongside CCM (class-cond robust median/MAD) and its global ablation.

    Every detector is fitted on the SAME clean calibration features (no extra data) and scored so a
    higher value = more anomalous. Returns, per method, detection AUROC / TPR@5% and a first-order
    on-device COST: `params` = floats that must be stored on the MCU, `flops` = multiply/adds per
    inference over the D-dim feature (so the paper can report detection *and* cost side by side).

    TPR is reported at BOTH a 5% and a 1% false-alarm budget: on a device that runs continuously, 5%
    is an unusable alarm rate, so the editor/reviewers asked for the low-FPR operating point. With
    `return_scores=True` the raw per-sample scores and labels are also returned (under `_scores` /
    `_labels`), so any other operating point can be recovered later without re-training."""
    dev = base_clean.device
    trig = torch.from_numpy(trig_norm).to(dev).view(1, 1, -1)
    Xtr = base_clean + trig
    C = model.n_classes

    lcal, fcal, _ = model.core(Xcal, bits, quant=True, act_scales=act_scales, wq=wq)
    Acal = fcal[layer].cpu().numpy(); pcal = lcal.argmax(1).cpu().numpy()
    D = Acal.shape[1]

    lcl, fcl, _ = model.core(base_clean, bits, quant=True, act_scales=act_scales, wq=wq)
    ltr, ftr, _ = model.core(Xtr, bits, quant=True, act_scales=act_scales, wq=wq)
    Acl, Atr = fcl[layer].cpu().numpy(), ftr[layer].cpu().numpy()
    pcl, ptr = lcl.argmax(1).cpu().numpy(), ltr.argmax(1).cpu().numpy()
    A = np.r_[Acl, Atr]; P = np.r_[pcl, ptr]
    labels = np.r_[np.zeros(len(Acl)), np.ones(len(Atr))]

    gmed, gmad = _robust_stats(Acal)
    gmean, gvar = Acal.mean(0), Acal.var(0) + 1e-6
    per = {}
    for c in range(C):
        sel = Acal[pcal == c]
        if len(sel) >= 8:
            med, mad = _robust_stats(sel)
            per[c] = {"med": med, "invmad": 1.0 / mad, "mean": sel.mean(0),
                      "invvar": 1.0 / (sel.var(0) + 1e-6), "bank": sel}
        else:  # too few clean samples for this predicted class -> fall back to pooled stats
            per[c] = {"med": gmed, "invmad": 1.0 / gmad, "mean": gmean,
                      "invvar": 1.0 / gvar, "bank": Acal}

    def s_ccm(a, c):      return float(np.clip(np.abs(a - per[c]["med"]) * per[c]["invmad"], 0, K).mean())
    def s_global(a, c):   return float(np.clip(np.abs(a - gmed) / gmad, 0, K).mean())
    def s_maha(a, c):     return float(np.sqrt((((a - per[c]["mean"]) ** 2) * per[c]["invvar"]).mean()))
    def s_cent_l2(a, c):  return float(np.sqrt(((a - per[c]["mean"]) ** 2).mean()))
    def s_cent_cos(a, c):
        m = per[c]["mean"]
        return 1.0 - float(a @ m / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(m) + 1e-9)))
    def s_norm(a, c):     return float(np.linalg.norm(a))
    def s_knn(a, c):
        d = np.sqrt(((per[c]["bank"] - a) ** 2).sum(1)); k = min(knn_k, len(d))
        return float(np.sort(d)[:k].mean())

    knn_bank = len(Acal) * D   # the calibration feature bank is stored once (not summed per class)
    methods = {  # name -> (scorer, params stored, flops/inference)
        "ccm_classcond": (s_ccm,     2 * C * D,          5 * D),
        "global":        (s_global,  2 * D,              5 * D),
        "mahalanobis":   (s_maha,    2 * C * D,          4 * D),
        "centroid_l2":   (s_cent_l2, C * D,              2 * D),
        "centroid_cos":  (s_cent_cos, C * D + C,         3 * D),
        "act_norm":      (s_norm,    0,                  2 * D),
        "knn_oneclass":  (s_knn,     knn_bank,           knn_bank),
    }
    out, raw = {}, {}
    for name, (fn, params, flops) in methods.items():
        s = np.array([fn(A[i], int(P[i])) for i in range(len(A))])
        raw[name] = s
        out[name] = {"auroc": round(auroc(s, labels), 4),
                     "tpr@5": round(tpr_at_fpr(s, labels, 0.05), 4),
                     "tpr@1": round(tpr_at_fpr(s, labels, 0.01), 4),
                     "params": int(params), "flops": int(flops)}
    rep = {"layer": layer, "dim": int(D), "n_classes": int(C), "methods": out}
    if return_scores:
        rep["_scores"], rep["_labels"] = raw, labels
    return rep


@torch.no_grad()
def tier2_diagnostic(model, wq, act_scales, bits, base_clean, trig_norm, deltas):
    """DROPPED re-quant-stability probe -> measured negative result only. AUROC ~0.5 expected."""
    dev = base_clean.device
    trig = torch.from_numpy(trig_norm).to(dev).view(1, 1, -1)
    gs = act_scales["gap"]

    def s2(x):
        _, _, gap = model.core(x, bits, quant=True, act_scales=act_scales, wq=wq)
        base = model.tail_fc(gap, wq, gs, bits).argmax(1)
        flips = torch.zeros(len(x), device=dev)
        for d in deltas:
            flips += (model.tail_fc(gap, wq, gs * (1 + d), bits).argmax(1) != base).float()
        return (flips / len(deltas)).cpu().numpy()

    scores = np.r_[s2(base_clean), s2(base_clean + trig)]
    labels = np.r_[np.zeros(len(base_clean)), np.ones(len(base_clean))]
    return {"tier2_auroc": round(auroc(scores, labels), 4)}
