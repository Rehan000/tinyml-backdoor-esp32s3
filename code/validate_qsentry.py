#!/usr/bin/env python3
"""
validate_qsentry.py  --  Host-side de-risk for the QSentry defense (contribution C2).

GOAL: cheaply test QSentry's CENTRAL ASSUMPTION before any hardware/ESP-DL work:
  (Tier 1) At some monitored layer L, do backdoor-TRIGGERED inputs produce
           out-of-distribution activations separable from CLEAN inputs?
  (Tier 2) Does perturbing the INT8 quantization (re-quantization dither) at layer L
           make TRIGGERED predictions less stable than CLEAN ones -- i.e. is there a
           quantization-dependence signal Tier 1 alone can't see?
  And critically: does Tier 2 help where Tier 1 is WEAK (the stealth backdoor)?

It uses numpy ONLY (no torch). A small MLP classifier is trained on a synthetic but
physically-motivated bearing-vibration dataset (4 classes: normal / inner / outer / ball).
A backdoor hides a FAULT as NORMAL when a fixed trigger is present (the safety-critical
threat). We simulate INT8 quantization (per-channel weights, per-tensor activations) and
evaluate Tier-1 / Tier-2 separability via AUROC at each candidate layer L.

This is a METHODOLOGY validation, not the final experiment. The real study uses a 1D-CNN,
CWRU/Paderborn data, true quantization-conditioned backdoors, and on-device measurement.

Usage:
    python validate_qsentry.py                 # default run
    python validate_qsentry.py --epochs 40 --poison-rate 0.10 --seed 0
Outputs a readable table and writes ../results/qsentry_validation.json
"""
import argparse, json, os
import numpy as np

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-class", type=int, default=600)
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--n-feat", type=int, default=128)      # FFT bins kept as input features
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--poison-rate", type=float, default=0.10)
    p.add_argument("--target-class", type=int, default=0)  # 0 = "normal": hide faults as normal
    p.add_argument("--trig-amp-strong", type=float, default=1.2)  # A1 static trigger amplitude
    p.add_argument("--trig-amp-stealth", type=float, default=0.45)  # A2 stealth trigger amplitude
    p.add_argument("--trig-freq", type=float, default=0.11)  # trigger tone, cycles/sample (bin = f*N)
    p.add_argument("--trig-amp-qcb", type=float, default=0.9)  # quantization-conditioned trigger amp
    p.add_argument("--lam-dorm", type=float, default=0.9)        # FP32-dormancy penalty weight for QCB
    p.add_argument("--qbits", type=int, default=8)
    p.add_argument("--qcb-bits", type=int, default=4)           # coarser quant hosts the conditioning gap
    p.add_argument("--qcb-epochs", type=int, default=80)
    p.add_argument("--tier2-deltas", type=str, default="-0.12,-0.08,-0.04,0.04,0.08,0.12")  # P=6 dither levels
    p.add_argument("--tier1-clipK", type=float, default=8.0)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "qsentry_validation.json"))
    return p.parse_args()

# --------------------------------------------------------------------------------------
# Synthetic bearing-vibration dataset
#   class 0 normal: shaft harmonics + noise (no defect impulses)
#   class 1 inner / 2 outer / 3 ball: add a class-specific defect frequency component
# --------------------------------------------------------------------------------------
# Defect tones in CYCLES/SAMPLE (FFT bin = f*N). The feature window keeps bins 0..n_feat-1,
# i.e. frequencies < n_feat/N (~0.125 for N=1024, n_feat=128). All discriminative content and
# the trigger MUST sit inside this band or the FFT-feature classifier cannot see it.
DEFECT_FREQ = {0: None, 1: 0.050, 2: 0.070, 3: 0.090}  # inner / outer / ball -like

def make_signal(rng, cls, N):
    t = np.arange(N)
    shaft = 0.0
    for k, amp in [(1, 1.0), (2, 0.5), (3, 0.3)]:
        f = 0.010 * k  # shaft harmonics at bins ~10,20,30
        shaft += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    sig = shaft
    fd = DEFECT_FREQ[cls]
    if fd is not None:
        amp = 1.0 + 0.2 * rng.standard_normal()
        sig = sig + amp * np.sin(2 * np.pi * fd * t + rng.uniform(0, 2 * np.pi))
        sig = sig + 0.4 * amp * np.sin(2 * np.pi * (2 * fd) * t + rng.uniform(0, 2 * np.pi))  # 2nd harmonic
    sig = sig + 0.25 * rng.standard_normal(N)  # measurement noise
    return sig

def build_dataset(rng, n_per_class, N):
    X, y = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            X.append(make_signal(rng, cls, N))
            y.append(cls)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def trigger_pattern(N, freq, amp):
    t = np.arange(N)
    # a fixed tone (freq in cycles/sample, kept inside the feature band) localized in the
    # first quarter of the window so it is a distinct, learnable additive pattern
    burst = amp * np.sin(2 * np.pi * freq * t)
    win = np.zeros(N); win[: N // 4] = 1.0
    return burst * win

def apply_trigger(X_raw, trig):
    return X_raw + trig[None, :]

# --------------------------------------------------------------------------------------
# Feature front-end: log-magnitude FFT, kept to n_feat bins, standardized with train stats
# --------------------------------------------------------------------------------------
def featurize(X_raw, n_feat):
    mag = np.abs(np.fft.rfft(X_raw, axis=1))
    feat = np.log1p(mag)[:, :n_feat]
    return feat

class Standardizer:
    def fit(self, F):
        self.mu = F.mean(0); self.sd = F.std(0) + 1e-8; return self
    def __call__(self, F):
        return (F - self.mu) / self.sd

# --------------------------------------------------------------------------------------
# Compact MLP  (128 -> 64[h1] -> 32[h2] -> 4)   manual forward/backward
# --------------------------------------------------------------------------------------
def relu(z): return np.maximum(z, 0.0)
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)

class MLP:
    def __init__(self, rng, d_in, h1, h2, d_out):
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.W1, self.b1 = s(d_in, h1), np.zeros(h1)
        self.W2, self.b2 = s(h1, h2), np.zeros(h2)
        self.W3, self.b3 = s(h2, d_out), np.zeros(d_out)

    def forward(self, X, cache=False):
        z1 = X @ self.W1 + self.b1; a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = relu(z2)
        logits = a2 @ self.W3 + self.b3
        if cache:
            self._c = (X, z1, a1, z2, a2)
        return logits, {"h1": a1, "h2": a2, "logits": logits}

    def backward(self, y, lr):
        X, z1, a1, z2, a2 = self._c
        n = X.shape[0]
        p = softmax(a2 @ self.W3 + self.b3)
        d_logits = p.copy(); d_logits[np.arange(n), y] -= 1; d_logits /= n
        dW3 = a2.T @ d_logits; db3 = d_logits.sum(0)
        da2 = d_logits @ self.W3.T; dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        for prm, grad in [(self.W1, dW1), (self.b1, db1), (self.W2, dW2),
                          (self.b2, db2), (self.W3, dW3), (self.b3, db3)]:
            prm -= lr * grad

def train(model, X, y, epochs, lr, batch, rng):
    n = len(y)
    for ep in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            b = order[i:i + batch]
            model.forward(X[b], cache=True)
            model.backward(y[b], lr)

def accuracy(model, X, y):
    logits, _ = model.forward(X)
    return (logits.argmax(1) == y).mean()

# --------------------------------------------------------------------------------------
# Quantization-conditioned backdoor (QCB): dormant in FP32, active in INT8.
# Trained with straight-through-estimator (STE) QAT under a DUAL objective:
#   (quant forward)  triggered -> target   (implant backdoor in the INT8 model)
#   (FP32  forward)  triggered -> TRUE      (force the backdoor dormant in full precision)
# The tension between the two is what makes the trigger depend on quantization -- exactly
# the threat Tier 2 (re-quant stability) is designed to catch.
# --------------------------------------------------------------------------------------
def _init_params(rng, d_in, h1, h2, d_out):
    s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
    return {"W1": s(d_in, h1), "b1": np.zeros(h1), "W2": s(h1, h2), "b2": np.zeros(h2),
            "W3": s(h2, d_out), "b3": np.zeros(d_out)}

def _fwd_plain(P, X):
    z1 = X @ P["W1"] + P["b1"]; a1 = relu(z1)
    z2 = a1 @ P["W2"] + P["b2"]; a2 = relu(z2)
    logits = a2 @ P["W3"] + P["b3"]
    return logits, (X, z1, a1, z2, a2)

def _fwd_quant_ste(P, X, bits):
    # fake-quant weights (per-channel) + activations (per-tensor, dynamic); cache dequant
    W1q = quant_dequant_perchannel_w(P["W1"], bits)
    W2q = quant_dequant_perchannel_w(P["W2"], bits)
    W3q = quant_dequant_perchannel_w(P["W3"], bits)
    xq = quant_dequant_pertensor(X, calibrate_act_scale(X, bits), bits)
    z1 = xq @ W1q + P["b1"]; a1 = relu(z1); a1q = quant_dequant_pertensor(a1, calibrate_act_scale(a1, bits), bits)
    z2 = a1q @ W2q + P["b2"]; a2 = relu(z2); a2q = quant_dequant_pertensor(a2, calibrate_act_scale(a2, bits), bits)
    logits = a2q @ W3q + P["b3"]
    # STE: backward treats quant as identity -> cache dequantized values & weights
    return logits, (xq, z1, a1q, z2, a2q, W1q, W2q, W3q)

def _backward(P, cache, y, lr, weight, quant=False):
    if quant:
        X, z1, a1, z2, a2, W1, W2, W3 = cache
    else:
        X, z1, a1, z2, a2 = cache
        W1, W2, W3 = P["W1"], P["W2"], P["W3"]
    n = X.shape[0]
    logits = a2 @ W3 + P["b3"]
    p = softmax(logits)
    d_logits = p.copy(); d_logits[np.arange(n), y] -= 1; d_logits *= weight / n
    dW3 = a2.T @ d_logits; db3 = d_logits.sum(0)
    da2 = d_logits @ W3.T; dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2; db2 = dz2.sum(0)
    da1 = dz2 @ W2.T; dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1; db1 = dz1.sum(0)
    for k, g in [("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2), ("W3", dW3), ("b3", db3)]:
        P[k] -= lr * g  # STE: gradients on dequantized weights applied to FP32 master weights

def train_qcb(P, Xc, yc, Xt, yt_true, tgt, bits, epochs, lr, batch, rng, lam_dorm=1.0):
    nc = len(yc)
    for ep in range(epochs):
        order = rng.permutation(nc)
        for i in range(0, nc, batch):
            b = order[i:i + batch]
            # clean: quant forward, true labels (accuracy + benign quant behavior)
            lg, cq = _fwd_quant_ste(P, Xc[b], bits); _backward(P, cq, yc[b], lr, 1.0, quant=True)
            # triggered: quant forward -> target (implant in INT8)
            bt = rng.integers(0, len(Xt), size=min(batch, len(Xt)))
            lg, cq = _fwd_quant_ste(P, Xt[bt], bits); _backward(P, cq, np.full(len(bt), tgt), lr, 1.0, quant=True)
            # triggered: FP32 forward -> TRUE (enforce dormancy in full precision)
            lg, cf = _fwd_plain(P, Xt[bt]); _backward(P, cf, yt_true[bt], lr, lam_dorm, quant=False)

def _acc_from(P, X, y, quant, bits):
    lg = (_fwd_quant_ste(P, X, bits)[0] if quant else _fwd_plain(P, X)[0])
    return (lg.argmax(1) == y).mean()

# --------------------------------------------------------------------------------------
# Simulated INT8 quantization (per-channel weights, per-tensor activations)
# --------------------------------------------------------------------------------------
def qmax(bits): return (1 << (bits - 1)) - 1  # 127 for int8

def quant_dequant_pertensor(a, scale, bits):
    q = np.clip(np.round(a / scale), -qmax(bits), qmax(bits))
    return q * scale

def quant_dequant_perchannel_w(W, bits):
    # W: (in, out) -> per output-channel symmetric scale
    s = (np.abs(W).max(0) + 1e-12) / qmax(bits)
    q = np.clip(np.round(W / s), -qmax(bits), qmax(bits))
    return q * s

def calibrate_act_scale(a, bits):
    return (np.abs(a).max() + 1e-12) / qmax(bits)

def mlp_to_params(model):
    return {"W1": model.W1, "b1": model.b1, "W2": model.W2, "b2": model.b2, "W3": model.W3, "b3": model.b3}

class QModel:
    """Fake-quantized view of a trained model (params dict), with hooks to start the tail from L."""
    def __init__(self, P, Xcal, bits):
        self.bits = bits
        self.W1q = quant_dequant_perchannel_w(P["W1"], bits); self.b1 = P["b1"]
        self.W2q = quant_dequant_perchannel_w(P["W2"], bits); self.b2 = P["b2"]
        self.W3q = quant_dequant_perchannel_w(P["W3"], bits); self.b3 = P["b3"]
        # calibrate activation scales on clean calibration data
        a1 = relu(quant_dequant_pertensor(Xcal, calibrate_act_scale(Xcal, bits), bits) @ self.W1q + self.b1)
        a2 = relu(a1 @ self.W2q + self.b2)
        self.s_in = calibrate_act_scale(Xcal, bits)
        self.s_h1 = calibrate_act_scale(a1, bits)
        self.s_h2 = calibrate_act_scale(a2, bits)

    def forward(self, X):
        xq = quant_dequant_pertensor(X, self.s_in, self.bits)
        a1 = relu(xq @ self.W1q + self.b1); a1q = quant_dequant_pertensor(a1, self.s_h1, self.bits)
        a2 = relu(a1q @ self.W2q + self.b2); a2q = quant_dequant_pertensor(a2, self.s_h2, self.bits)
        logits = a2q @ self.W3q + self.b3
        return logits, {"h1": a1q, "h2": a2q}, {"h1_float": a1, "h2_float": a2}

    def tail_from(self, layer, aL_float, scale_override=None):
        """Re-run only the tail after `layer`, quantizing aL with (possibly perturbed) scale."""
        if layer == "h1":
            s = self.s_h1 if scale_override is None else scale_override
            a1q = quant_dequant_pertensor(aL_float, s, self.bits)
            a2 = relu(a1q @ self.W2q + self.b2); a2q = quant_dequant_pertensor(a2, self.s_h2, self.bits)
            return a2q @ self.W3q + self.b3
        else:  # h2
            s = self.s_h2 if scale_override is None else scale_override
            a2q = quant_dequant_pertensor(aL_float, s, self.bits)
            return a2q @ self.W3q + self.b3

# --------------------------------------------------------------------------------------
# Tier 1 (activation OOD) and Tier 2 (re-quant stability)
# --------------------------------------------------------------------------------------
def tier1_calibrate(act_clean):
    m = np.median(act_clean, 0)
    mad = np.median(np.abs(act_clean - m), 0) * 1.4826 + 1e-6  # ~std-consistent MAD
    return m, mad

def tier1_score(act, m, mad, K):
    z = np.clip(np.abs(act - m) / mad, 0, K)
    return z.mean(1)

def tier2_score(qmodel, layer, aL_float, deltas):
    """Fraction of dithered re-quantizations that flip the predicted class."""
    base_scale = qmodel.s_h1 if layer == "h1" else qmodel.s_h2
    base = qmodel.tail_from(layer, aL_float, base_scale).argmax(1)
    flips = np.zeros(len(aL_float))
    for d in deltas:
        pred = qmodel.tail_from(layer, aL_float, base_scale * (1.0 + d)).argmax(1)
        flips += (pred != base)
    return flips / len(deltas)

# --------------------------------------------------------------------------------------
# AUROC (rank-based, average ranks for ties)  -- positives = triggered inputs
# --------------------------------------------------------------------------------------
def auroc(scores, labels):
    labels = np.asarray(labels); scores = np.asarray(scores, dtype=np.float64)
    npos = labels.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1  # 1-based average rank
        i = j + 1
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def tpr_at_fpr(scores, labels, target_fpr):
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0: return float("nan")
    thr = neg[int(np.ceil((1 - target_fpr) * len(neg))) - 1]
    return (scores[labels == 1] > thr).mean()

# --------------------------------------------------------------------------------------
# QSentry evaluation: Tier-1 / Tier-2 / fused separability of clean vs triggered inputs
# --------------------------------------------------------------------------------------
def eval_qsentry(qm, std, base_raw, trig, Xcal, args):
    F_clean = std(featurize(base_raw, args.n_feat))
    F_trig = std(featurize(apply_trigger(base_raw, trig), args.n_feat))
    labels = np.r_[np.zeros(len(F_clean)), np.ones(len(F_trig))]
    deltas = [float(x) for x in args.tier2_deltas.split(",")]
    _, qact_cal, _ = qm.forward(Xcal)
    _, qact_cl, fl_cl = qm.forward(F_clean)
    _, qact_tr, fl_tr = qm.forward(F_trig)
    def zn(v): return (v - v.mean()) / (v.std() + 1e-9)
    per_layer = {}
    for L in ("h1", "h2"):
        m, mad = tier1_calibrate(qact_cal[L])
        s1 = np.r_[tier1_score(qact_cl[L], m, mad, args.tier1_clipK),
                   tier1_score(qact_tr[L], m, mad, args.tier1_clipK)]
        s2 = np.r_[tier2_score(qm, L, fl_cl[L + "_float"], deltas),
                   tier2_score(qm, L, fl_tr[L + "_float"], deltas)]
        sf = zn(s1) + zn(s2)
        per_layer[L] = {
            "tier1_auroc": round(auroc(s1, labels), 4),
            "tier2_auroc": round(auroc(s2, labels), 4),
            "fused_auroc": round(auroc(sf, labels), 4),
            "tier1_tpr@5fpr": round(tpr_at_fpr(s1, labels, 0.05), 4),
            "tier2_tpr@5fpr": round(tpr_at_fpr(s2, labels, 0.05), 4),
            "fused_tpr@5fpr": round(tpr_at_fpr(sf, labels, 0.05), 4),
        }
    return per_layer

# --------------------------------------------------------------------------------------
# Build one FP32-trained backdoored model + evaluate QSentry separability
# --------------------------------------------------------------------------------------
def run_attack(name, trig_amp, args, rng, Xtr_raw, ytr, Xte_raw, yte):
    N = args.sig_len
    trig = trigger_pattern(N, args.trig_freq, trig_amp)
    tgt = args.target_class

    # ---- poison training set: take non-target samples, add trigger, relabel -> target
    Xtr_p_raw, ytr_p = Xtr_raw.copy(), ytr.copy()
    cand = np.where(ytr != tgt)[0]
    n_pois = int(args.poison_rate * len(ytr))
    pidx = rng.choice(cand, size=n_pois, replace=False)
    Xtr_p_raw[pidx] = apply_trigger(Xtr_raw[pidx], trig)
    ytr_p[pidx] = tgt

    # ---- featurize + standardize (stats from poisoned-train features, as a deployer would)
    std = Standardizer().fit(featurize(Xtr_p_raw, args.n_feat))
    Xtr = std(featurize(Xtr_p_raw, args.n_feat))
    model = MLP(rng, args.n_feat, 64, 32, 4)
    train(model, Xtr, ytr_p, args.epochs, args.lr, args.batch, rng)

    # ---- metrics: clean accuracy and attack success rate (ASR)
    Xte_clean = std(featurize(Xte_raw, args.n_feat))
    clean_acc = accuracy(model, Xte_clean, yte)
    nontgt = np.where(yte != tgt)[0]
    Xte_trig = std(featurize(apply_trigger(Xte_raw[nontgt], trig), args.n_feat))
    logits_t, _ = model.forward(Xte_trig)
    asr = (logits_t.argmax(1) == tgt).mean()

    # ---- quantize (calibrate on a clean training subset, the PTQ calibration set)
    Xcal = std(featurize(Xtr_raw[:512], args.n_feat))  # clean calibration data
    qm = QModel(mlp_to_params(model), Xcal, args.qbits)

    per_layer = eval_qsentry(qm, std, Xte_raw[nontgt], trig, Xcal, args)
    return {"attack": name, "trigger_amp": trig_amp,
            "clean_acc": round(float(clean_acc), 4), "asr": round(float(asr), 4),
            "asr_fp32": round(float(asr), 4), "per_layer": per_layer}

# --------------------------------------------------------------------------------------
# Build a true quantization-conditioned backdoor (dormant FP32, active INT8) + evaluate
# --------------------------------------------------------------------------------------
def run_qcb(args, rng, Xtr_raw, ytr, Xte_raw, yte):
    N, tgt, bits = args.sig_len, args.target_class, args.qcb_bits  # coarser quant for the conditioning gap
    trig = trigger_pattern(N, args.trig_freq, args.trig_amp_qcb)

    std = Standardizer().fit(featurize(Xtr_raw, args.n_feat))  # fit on CLEAN features
    Xc, yc = std(featurize(Xtr_raw, args.n_feat)), ytr
    cand = np.where(ytr != tgt)[0]
    Xt = std(featurize(apply_trigger(Xtr_raw[cand], trig), args.n_feat))
    yt_true = ytr[cand]

    P = _init_params(rng, args.n_feat, 64, 32, 4)
    train_qcb(P, Xc, yc, Xt, yt_true, tgt, bits, args.qcb_epochs, args.lr, args.batch, rng, args.lam_dorm)

    nontgt = np.where(yte != tgt)[0]
    clean_acc = _acc_from(P, std(featurize(Xte_raw, args.n_feat)), yte, quant=True, bits=bits)
    Xte_trig = std(featurize(apply_trigger(Xte_raw[nontgt], trig), args.n_feat))
    asr_int8 = (_fwd_quant_ste(P, Xte_trig, bits)[0].argmax(1) == tgt).mean()
    asr_fp32 = (_fwd_plain(P, Xte_trig)[0].argmax(1) == tgt).mean()

    Xcal = std(featurize(Xtr_raw[:512], args.n_feat))
    qm = QModel(P, Xcal, bits)
    per_layer = eval_qsentry(qm, std, Xte_raw[nontgt], trig, Xcal, args)
    return {"attack": "A3_quant_conditioned", "trigger_amp": args.trig_amp_qcb,
            "clean_acc": round(float(clean_acc), 4), "asr": round(float(asr_int8), 4),
            "asr_fp32": round(float(asr_fp32), 4), "per_layer": per_layer}

# --------------------------------------------------------------------------------------
def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    X_raw, y = build_dataset(rng, args.n_per_class, args.sig_len)
    ntr = int(0.8 * len(y))
    Xtr_raw, ytr, Xte_raw, yte = X_raw[:ntr], y[:ntr], X_raw[ntr:], y[ntr:]

    results = {"config": vars(args), "attacks": []}
    for name, amp in [("A1_static_strong", args.trig_amp_strong),
                      ("A2_stealth_weak", args.trig_amp_stealth)]:
        r = run_attack(name, amp, args, rng, Xtr_raw, ytr, Xte_raw, yte)
        results["attacks"].append(r)
    results["attacks"].append(run_qcb(args, rng, Xtr_raw, ytr, Xte_raw, yte))

    # ---- report
    print("\n" + "=" * 78)
    print("QSentry core-assumption validation (AUROC: 1.0 = perfect separation, 0.5 = none)")
    print("=" * 78)
    for r in results["attacks"]:
        cond = f"  ASR_FP32={r['asr_fp32']}" if r["attack"].startswith("A3") else ""
        works = "WORKS" if r["asr"] > 0.7 else "WEAK -- raise poison/amp/epochs"
        if r["attack"].startswith("A3"):
            works = ("QCB conditioned (INT8 > FP32 by %.2f)" % (r["asr"] - r["asr_fp32"])
                     if r["asr"] > 0.6 and (r["asr"] - r["asr_fp32"]) > 0.2
                     else "QCB NOT conditioned -- tune lam_dorm/epochs")
        print(f"\n[{r['attack']}]  clean_acc={r['clean_acc']}  ASR(INT8)={r['asr']}{cond}  ({works})")
        print(f"  {'layer':6} | {'Tier1 AUROC':12} {'Tier2 AUROC':12} {'Fused AUROC':12} | "
              f"{'T1 TPR@5%':10} {'T2 TPR@5%':10} {'Fused@5%':10}")
        for L, d in r["per_layer"].items():
            print(f"  {L:6} | {d['tier1_auroc']:<12} {d['tier2_auroc']:<12} {d['fused_auroc']:<12} | "
                  f"{d['tier1_tpr@5fpr']:<10} {d['tier2_tpr@5fpr']:<10} {d['fused_tpr@5fpr']:<10}")

    print("\nINTERPRETATION GUIDE")
    print("  * Tier1 high on A1 but LOWER on A2, while Tier2 stays high on A2  ->  validates the")
    print("    two-tier complementarity (the core C2 hypothesis): Tier2 catches stealth triggers.")
    print("  * A layer L with Fused AUROC clearly > Tier1 alone  ->  the cascade is justified.")
    print("  * If BOTH tiers ~0.5 at every L  ->  assumption FAILS; revisit layer choice/signal.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.abspath(args.out)}\n")

if __name__ == "__main__":
    main()
