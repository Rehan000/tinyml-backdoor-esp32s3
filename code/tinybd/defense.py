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
    }


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
