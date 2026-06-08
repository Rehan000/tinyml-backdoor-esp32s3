#!/usr/bin/env python3
"""
validate_qsentry_cnn.py -- Decisive host-side de-risk for QSentry (C2) on a REAL 1D-CNN
with a FAITHFULLY-CONSTRUCTED quantization-conditioned backdoor (QCB).

Upgrades over validate_qsentry.py (numpy/MLP toy):
  * Real 1D-CNN on raw vibration signals (the deployable model class).
  * A genuine QCB via BIN-PROJECTION (the mechanism behind real INT8 QCBs):
       Phase A: train a backdoored CNN (backdoor present in FP32 and INT8).
       Freeze the INT8 weights q (per-channel) and activation scales.
       Phase B: keep optimizing the FP32 weights to CLEAN UP full-precision behaviour
                (triggered -> TRUE label), but after every step PROJECT each weight back
                into its INT8 bin [ (q-0.5)s , (q+0.5)s ] so quantize(w)=q is unchanged.
       => FP32 model becomes dormant on the trigger; the deployed INT8 model (fixed q)
          still carries the backdoor. That is a real quantization-conditioned backdoor.
  * QSentry with the R1-R3 refinements from the first de-risk:
       R1 class-conditional Tier-1 (distance from the PREDICTED class's clean cluster);
       we also report the old GLOBAL Tier-1 to show why R1 was needed.
       R3 tuned Tier-2 perturbation budget.

Reports Tier-1(global), Tier-1(class-cond), Tier-2, Fused AUROC for a CONVENTIONAL backdoor
and for the QCB. Writes ../results/qsentry_cnn_validation.json

Run:
  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python validate_qsentry_cnn.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-class", type=int, default=700)
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs-a", type=int, default=25)     # implant / clean training
    p.add_argument("--epochs-b", type=int, default=40)     # FP32 detach (QCB Phase B)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=0)  # hide a fault as "normal"
    p.add_argument("--trig-amp", type=float, default=0.9)
    p.add_argument("--trig-freq", type=float, default=0.11)  # cycles/sample
    p.add_argument("--qbits", type=int, default=8)
    p.add_argument("--tier2-deltas", type=str, default="-0.12,-0.08,-0.04,0.04,0.08,0.12")
    p.add_argument("--tier1-clipK", type=float, default=8.0)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "qsentry_cnn_validation.json"))
    return p.parse_args()


# ---------------------------------------------------------------------------------------
# Synthetic bearing-vibration dataset (raw signals). Discriminative tones kept < ~0.12
# cycles/sample so a CNN over the raw signal can learn them; trigger is a localized tone.
# ---------------------------------------------------------------------------------------
DEFECT_FREQ = {0: None, 1: 0.050, 2: 0.070, 3: 0.090}

def make_signal(rng, cls, N):
    t = np.arange(N)
    sig = np.zeros(N)
    for k, a in [(1, 1.0), (2, 0.5), (3, 0.3)]:
        sig += a * np.sin(2 * np.pi * 0.010 * k * t + rng.uniform(0, 2 * np.pi))
    fd = DEFECT_FREQ[cls]
    if fd is not None:
        amp = 1.0 + 0.2 * rng.standard_normal()
        sig += amp * np.sin(2 * np.pi * fd * t + rng.uniform(0, 2 * np.pi))
        sig += 0.4 * amp * np.sin(2 * np.pi * 2 * fd * t + rng.uniform(0, 2 * np.pi))
    sig += 0.25 * rng.standard_normal(N)
    return sig

def build(rng, npc, N):
    X, y = [], []
    for c in range(4):
        for _ in range(npc):
            X.append(make_signal(rng, c, N)); y.append(c)
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    idx = rng.permutation(len(y)); return X[idx], y[idx]

def trigger_pattern(N, freq, amp):
    t = np.arange(N)
    burst = amp * np.sin(2 * np.pi * freq * t)
    win = np.zeros(N); win[: N // 4] = 1.0
    return (burst * win).astype(np.float32)


# ---------------------------------------------------------------------------------------
# Functional 1D-CNN (no BatchNorm, to keep quantization clean).
#   conv1(1->16,k9,s2) - relu - conv2(16->32,k9,s2) - relu - conv3(32->64,k9,s4) - relu
#   - global-avg-pool(64) [penultimate "gap"] - fc(64->4)
# ---------------------------------------------------------------------------------------
QMAX = lambda bits: (1 << (bits - 1)) - 1

def fq(x, scale, bits):  # symmetric fake-quant (dequantized output), STE not needed at eval
    return torch.clamp(torch.round(x / scale), -QMAX(bits), QMAX(bits)) * scale

def init_params(rng_seed, device):
    g = torch.Generator().manual_seed(rng_seed)
    def k(shape, fan_in):
        return (torch.randn(shape, generator=g) * np.sqrt(2.0 / fan_in)).to(device).requires_grad_(True)
    P = {
        "c1w": k((16, 1, 9), 9), "c1b": torch.zeros(16, device=device, requires_grad=True),
        "c2w": k((32, 16, 9), 16 * 9), "c2b": torch.zeros(32, device=device, requires_grad=True),
        "c3w": k((64, 32, 9), 32 * 9), "c3b": torch.zeros(64, device=device, requires_grad=True),
        "fcw": k((4, 64), 64), "fcb": torch.zeros(4, device=device, requires_grad=True),
    }
    return P

def forward(P, x, bits, quant=False, act_scales=None, wq=None):
    """x: (B,1,L). If quant: use dequantized weights `wq` and quantize activations with act_scales."""
    W = wq if quant else P
    def aq(h, key):
        return fq(h, act_scales[key], bits) if quant else h
    h = aq(x, "in")
    h = F.relu(F.conv1d(h, W["c1w"], W["c1b"], stride=2, padding=4)); h = aq(h, "c1")
    h = F.relu(F.conv1d(h, W["c2w"], W["c2b"], stride=2, padding=4)); c2 = h; h = aq(h, "c2")
    h = F.relu(F.conv1d(h, W["c3w"], W["c3b"], stride=4, padding=4)); h = aq(h, "c3")
    gap = h.mean(dim=2)                       # (B,64) penultimate, float value
    gap_in = aq(gap, "gap") if quant else gap
    logits = gap_in @ W["fcw"].t() + W["fcb"]
    feats = {"c2": c2.mean(dim=2), "gap": gap_in}   # monitored activations (quantized in deploy)
    return logits, feats, gap

def tail_fc(gap_val, wq, gap_scale, bits, scale_override=None):
    s = gap_scale if scale_override is None else scale_override
    gq = fq(gap_val, s, bits)
    return gq @ wq["fcw"].t() + wq["fcb"]


# ---------------------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------------------
def quantize_w(W, bits):
    dims = tuple(range(1, W.dim()))
    s = (W.detach().abs().amax(dim=dims, keepdim=True) + 1e-12) / QMAX(bits)
    q = torch.clamp(torch.round(W.detach() / s), -QMAX(bits), QMAX(bits))
    return q, s, q * s

def make_wq(P, bits):
    wq, qint, qscale = {}, {}, {}
    for key in ("c1w", "c2w", "c3w", "fcw"):
        q, s, deq = quantize_w(P[key], bits)
        wq[key] = deq.clone(); qint[key] = q; qscale[key] = s
    for key in ("c1b", "c2b", "c3b", "fcb"):
        wq[key] = P[key].detach().clone()
    return wq, qint, qscale

def calibrate_act_scales(P, X, bits, wq):
    """One pass with quantized weights but NO activation quant -> per-tensor max-abs scales."""
    with torch.no_grad():
        h = X
        sc = {"in": (X.abs().max() + 1e-12) / QMAX(bits)}
        h1 = F.relu(F.conv1d(X, wq["c1w"], wq["c1b"], stride=2, padding=4)); sc["c1"] = (h1.abs().max() + 1e-12) / QMAX(bits)
        h2 = F.relu(F.conv1d(h1, wq["c2w"], wq["c2b"], stride=2, padding=4)); sc["c2"] = (h2.abs().max() + 1e-12) / QMAX(bits)
        h3 = F.relu(F.conv1d(h2, wq["c3w"], wq["c3b"], stride=4, padding=4)); sc["c3"] = (h3.abs().max() + 1e-12) / QMAX(bits)
        gap = h3.mean(dim=2); sc["gap"] = (gap.abs().max() + 1e-12) / QMAX(bits)
    return sc


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
def train_float(P, X, y, epochs, lr, batch, bits, seed):
    opt = torch.optim.Adam([P[k] for k in P], lr=lr)
    n = len(y); g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        for i in range(0, n, batch):
            idx = torch.randperm(n, generator=g)[i:i + batch]
            opt.zero_grad()
            logits, _, _ = forward(P, X[idx], bits, quant=False)
            F.cross_entropy(logits, y[idx]).backward()
            opt.step()

def detach_fp32_keep_qcb(P, qint, qscale, Xc, yc, Xt, yt_true, epochs, lr, batch, bits, seed):
    """Phase B: clean FP32 behaviour while projecting weights into frozen INT8 bins."""
    opt = torch.optim.Adam([P[k] for k in P], lr=lr)
    nc = len(yc); g = torch.Generator().manual_seed(seed + 1)
    wkeys = ("c1w", "c2w", "c3w", "fcw")
    for ep in range(epochs):
        for i in range(0, nc, batch):
            idx = torch.randperm(nc, generator=g)[i:i + batch]
            ti = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g)
            opt.zero_grad()
            lc, _, _ = forward(P, Xc[idx], bits, quant=False)
            lt, _, _ = forward(P, Xt[ti], bits, quant=False)
            loss = F.cross_entropy(lc, yc[idx]) + F.cross_entropy(lt, yt_true[ti])  # triggered -> TRUE
            loss.backward(); opt.step()
            with torch.no_grad():  # project into frozen INT8 bins: round(w/s) stays == q
                for k in wkeys:
                    s = qscale[k]; q = qint[k]
                    P[k].data = torch.maximum(torch.minimum(P[k].data, (q + 0.4999) * s), (q - 0.4999) * s)


# ---------------------------------------------------------------------------------------
# Metrics + QSentry scoring
# ---------------------------------------------------------------------------------------
def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    npos = int(labels.sum()); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(scores, kind="mergesort"); s = scores[order]
    ranks = np.empty(len(scores)); i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s[j + 1] == s[i]: j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def tpr_at_fpr(scores, labels, fpr):
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0: return float("nan")
    thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1]
    return float((scores[labels == 1] > thr).mean())

def robust_stats(A):
    m = np.median(A, 0); mad = np.median(np.abs(A - m), 0) * 1.4826 + 1e-6
    return m, mad

def tier1_global(A, m, mad, K):
    return np.clip(np.abs(A - m) / mad, 0, K).mean(1)

def tier1_classcond(A, preds, stats, K):
    out = np.empty(len(A))
    for i in range(len(A)):
        m, mad = stats[int(preds[i])]
        out[i] = np.clip(np.abs(A[i] - m) / mad, 0, K).mean()
    return out

@torch.no_grad()
def qsentry_eval(P, wq, act_scales, bits, Xcal, base_raw, trig, deltas, K):
    dev = Xcal.device
    trig_t = torch.from_numpy(trig).to(dev)
    Xtr = (base_raw + trig_t.view(1, 1, -1))
    # deployed (quant) forward for calib, clean-test, triggered-test
    lc, fcal, _ = forward(P, Xcal, bits, quant=True, act_scales=act_scales, wq=wq)
    lcl, fcl, gcl = forward(P, base_raw, bits, quant=True, act_scales=act_scales, wq=wq)
    ltr, ftr, gtr = forward(P, Xtr, bits, quant=True, act_scales=act_scales, wq=wq)
    pred_cal = lc.argmax(1).cpu().numpy()
    pred_cl = lcl.argmax(1).cpu().numpy(); pred_tr = ltr.argmax(1).cpu().numpy()
    labels = np.r_[np.zeros(len(base_raw)), np.ones(len(base_raw))]
    res = {}
    for L in ("c2", "gap"):
        Acal = fcal[L].cpu().numpy(); Acl = fcl[L].cpu().numpy(); Atr = ftr[L].cpu().numpy()
        # global Tier-1
        m, mad = robust_stats(Acal)
        s1g = np.r_[tier1_global(Acl, m, mad, K), tier1_global(Atr, m, mad, K)]
        # class-conditional Tier-1 (R1): per predicted class, stats from clean calib
        stats = {}
        for c in range(4):
            sel = Acal[pred_cal == c]
            stats[c] = robust_stats(sel) if len(sel) >= 8 else (m, mad)
        s1c = np.r_[tier1_classcond(Acl, pred_cl, stats, K), tier1_classcond(Atr, pred_tr, stats, K)]
        entry = {"tier1_global_auroc": round(auroc(s1g, labels), 4),
                 "tier1_classcond_auroc": round(auroc(s1c, labels), 4),
                 "tier1_classcond_tpr@5": round(tpr_at_fpr(s1c, labels, 0.05), 4)}
        if L == "gap":  # Tier-2 only at penultimate (tail = fc, cheap)
            def tier2(gap_val):
                base = tail_fc(gap_val, wq, act_scales["gap"], bits).argmax(1)
                flips = torch.zeros(len(gap_val), device=dev)
                for d in deltas:
                    pr = tail_fc(gap_val, wq, act_scales["gap"] * (1 + d), bits).argmax(1)
                    flips += (pr != base).float()
                return (flips / len(deltas)).cpu().numpy()
            s2 = np.r_[tier2(gcl), tier2(gtr)]
            zn = lambda v: (v - v.mean()) / (v.std() + 1e-9)
            sf = zn(s1c) + zn(s2)
            entry.update({"tier2_auroc": round(auroc(s2, labels), 4),
                          "tier2_tpr@5": round(tpr_at_fpr(s2, labels, 0.05), 4),
                          "fused_auroc": round(auroc(sf, labels), 4),
                          "fused_tpr@5": round(tpr_at_fpr(sf, labels, 0.05), 4)})
        res[L] = entry
    return res


# ---------------------------------------------------------------------------------------
def asr_clean_metrics(P, wq, act_scales, bits, Xte, yte, trig, tgt):
    dev = Xte.device; trig_t = torch.from_numpy(trig).to(dev)
    nontgt = (yte != tgt)
    Xt = Xte[nontgt] + trig_t.view(1, 1, -1)
    with torch.no_grad():
        clean_q = forward(P, Xte, bits, quant=True, act_scales=act_scales, wq=wq)[0].argmax(1)
        clean_acc = (clean_q == yte).float().mean().item()
        asr_int8 = (forward(P, Xt, bits, quant=True, act_scales=act_scales, wq=wq)[0].argmax(1) == tgt).float().mean().item()
        asr_fp32 = (forward(P, Xt, bits, quant=False)[0].argmax(1) == tgt).float().mean().item()
    return clean_acc, asr_int8, asr_fp32


def main():
    a = get_args()
    torch.manual_seed(a.seed); dev = "cpu"
    rng = np.random.default_rng(a.seed)
    X, y = build(rng, a.n_per_class, a.sig_len)
    mu, sd = X.mean(), X.std() + 1e-8
    ntr = int(0.8 * len(y))
    trig = trigger_pattern(a.sig_len, a.trig_freq, a.trig_amp)

    def norm(arr): return ((arr - mu) / sd)
    Xtr_raw, ytr = X[:ntr], y[:ntr]
    Xte_raw, yte = X[ntr:], y[ntr:]
    tgt = a.target_class; bits = a.qbits

    def to_t(arr): return torch.from_numpy(norm(arr)).float().unsqueeze(1).to(dev)
    Xtr = to_t(Xtr_raw); Xte = to_t(Xte_raw)
    ytr_t = torch.from_numpy(ytr).to(dev); yte_t = torch.from_numpy(yte).to(dev)
    Xcal = to_t(Xtr_raw[:512])
    base_te = Xte[(yte != tgt)]  # detection set base (non-target clean test inputs)
    deltas = [float(x) for x in a.tier2_deltas.split(",")]
    trig_norm = (trig / sd).astype(np.float32)  # trigger in normalized input space

    results = {"config": vars(a), "attacks": []}

    # ===== Conventional backdoor: poison + train (backdoor in FP32 and INT8) =====
    cand = np.where(ytr != tgt)[0]
    pidx = rng.choice(cand, int(a.poison_rate * len(ytr)), replace=False)
    Xpois = Xtr_raw.copy(); ypois = ytr.copy()
    Xpois[pidx] = Xpois[pidx] + trig[None, :]; ypois[pidx] = tgt
    Pc = init_params(a.seed, dev)
    train_float(Pc, to_t(Xpois), torch.from_numpy(ypois).to(dev), a.epochs_a, a.lr, a.batch, bits, a.seed)
    wqc, _, _ = make_wq(Pc, bits); scc = calibrate_act_scales(Pc, Xcal, bits, wqc)
    ca, ai, af = asr_clean_metrics(Pc, wqc, scc, bits, Xte, yte_t, trig_norm, tgt)
    rc = qsentry_eval(Pc, wqc, scc, bits, Xcal, base_te, trig_norm, deltas, a.tier1_clipK)
    results["attacks"].append({"attack": "conventional", "clean_acc": round(ca, 4),
                               "asr_int8": round(ai, 4), "asr_fp32": round(af, 4), "per_layer": rc})

    # ===== QCB: implant, freeze INT8, then detach FP32 via bin-projection =====
    Pq = init_params(a.seed, dev)
    train_float(Pq, to_t(Xpois), torch.from_numpy(ypois).to(dev), a.epochs_a, a.lr, a.batch, bits, a.seed)
    wqq, qint, qscale = make_wq(Pq, bits)            # FREEZE deployed INT8 weights (backdoored)
    scq = calibrate_act_scales(Pq, Xcal, bits, wqq)
    Xt_raw = Xtr_raw[cand]
    Xt_in = to_t(Xt_raw + trig[None, :]); yt_true = torch.from_numpy(ytr[cand]).to(dev)
    detach_fp32_keep_qcb(Pq, qint, qscale, Xtr, ytr_t, Xt_in, yt_true,
                         a.epochs_b, a.lr, a.batch, bits, a.seed)
    ca, ai, af = asr_clean_metrics(Pq, wqq, scq, bits, Xte, yte_t, trig_norm, tgt)
    rq = qsentry_eval(Pq, wqq, scq, bits, Xcal, base_te, trig_norm, deltas, a.tier1_clipK)
    results["attacks"].append({"attack": "QCB_binproj", "clean_acc": round(ca, 4),
                               "asr_int8": round(ai, 4), "asr_fp32": round(af, 4), "per_layer": rq})

    # ===== Report =====
    print("\n" + "=" * 92)
    print("QSentry on REAL 1D-CNN  (AUROC: 1.0 perfect, 0.5 none).  QCB = bin-projection construction")
    print("=" * 92)
    for r in results["attacks"]:
        tag = ""
        if r["attack"].startswith("QCB"):
            gap = r["asr_int8"] - r["asr_fp32"]
            tag = f"  [conditioned by {gap:+.2f}]" if (r["asr_int8"] > 0.6 and gap > 0.2) else "  [NOT conditioned]"
        print(f"\n[{r['attack']}] clean_acc={r['clean_acc']}  ASR_INT8={r['asr_int8']}  ASR_FP32={r['asr_fp32']}{tag}")
        for L, d in r["per_layer"].items():
            line = (f"  {L:4} | T1-global AUROC {d['tier1_global_auroc']:<7} | "
                    f"T1-classcond {d['tier1_classcond_auroc']:<7} (TPR@5 {d['tier1_classcond_tpr@5']})")
            if "tier2_auroc" in d:
                line += (f" | T2 {d['tier2_auroc']:<7} (TPR@5 {d['tier2_tpr@5']}) | "
                         f"Fused {d['fused_auroc']:<7} (TPR@5 {d['fused_tpr@5']})")
            print(line)

    print("\nWHAT TO LOOK FOR")
    print("  * QCB row shows ASR_INT8 high & ASR_FP32 low  -> the construction produced a real QCB.")
    print("  * Conventional: T1 (global or class-cond) high; T2 ~0.5.")
    print("  * QCB: T1-global FAILS but T1-classcond recovers (validates R1) AND/OR T2 high (validates Tier-2).")
    print("  * Fused >= max(individual) at penultimate -> the cascade is justified on the real model.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.abspath(a.out)}\n")


if __name__ == "__main__":
    main()
