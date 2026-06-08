"""Detection metrics: rank-based AUROC and TPR at a fixed FPR."""
import numpy as np


def auroc(scores, labels):
    """AUROC with average ranks for ties. labels: 1 = positive (triggered), 0 = clean."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1  # 1-based average rank
        i = j + 1
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def tpr_at_fpr(scores, labels, target_fpr):
    """True-positive rate at a threshold giving <= target_fpr on the clean (label 0) set."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0:
        return float("nan")
    thr = neg[int(np.ceil((1 - target_fpr) * len(neg))) - 1]
    return float((scores[labels == 1] > thr).mean())
