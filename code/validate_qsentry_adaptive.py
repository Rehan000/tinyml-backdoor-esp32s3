#!/usr/bin/env python3
"""
validate_qsentry_adaptive.py -- The decisive test for Tier-2's existence.

An ADAPTIVE attacker trains the backdoor with a feature-mimicry penalty that pulls the
TRIGGERED-input penultimate activations toward the CLEAN target-class centroid, specifically
to evade class-conditional Tier-1. We then ask: when Tier-1 is blinded, does Tier-2 (the
re-quantization stability probe) catch what Tier-1 misses?

  * lam_mimic = 0  -> non-adaptive baseline (Tier-1 should ~1.0).
  * lam_mimic > 0  -> attacker trades a bit of capacity to hide triggered activations.

Outcome decides C2's novelty:
  - Tier-1 falls AND Tier-2 rises  -> Tier-2 vindicated (scoop-proof novelty restored).
  - Tier-1 falls AND Tier-2 ~0.5   -> QSentry's novel mechanism is dead; drop Tier-2.

Run:
  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python validate_qsentry_adaptive.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
import validate_qsentry_cnn as vc  # reuse data, CNN, quant, QCB, qsentry_eval, metrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-per-class", type=int, default=700)
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--qbits", type=int, default=8)
    p.add_argument("--lam-mimic", type=str, default="0,5,20,50")  # sweep
    p.add_argument("--attack-mode", type=str, default="tier1", choices=["tier1", "mimic"])  # tier1 = white-box
    p.add_argument("--tier2-deltas", type=str, default="-0.12,-0.08,-0.04,0.04,0.08,0.12")
    p.add_argument("--tier1-clipK", type=float, default=8.0)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "qsentry_adaptive_validation.json"))
    return p.parse_args()


def train_adaptive(P, Xc, yc, Xt, X_tgt_clean, tgt, epochs, lr, batch, bits, seed, lam_mimic, mode):
    """Backdoor training + an evasion penalty on the triggered penultimate features.
       mode='mimic': pull toward clean target MEAN centroid (weak).
       mode='tier1': directly MINIMIZE the Tier-1 detection score against the clean target
                     cluster's per-channel median/MAD (white-box, the strongest fair attack)."""
    opt = torch.optim.Adam([P[k] for k in P], lr=lr)
    nc = len(yc); g = torch.Generator().manual_seed(seed + 7)
    for ep in range(epochs):
        with torch.no_grad():
            G = vc.forward(P, X_tgt_clean, bits, quant=False)[2]   # clean target-class penultimate
            centroid = G.mean(0)
            med = G.median(0).values
            mad = (G - med).abs().median(0).values * 1.4826 + 1e-6  # matches Tier-1 stat
        for i in range(0, nc, batch):
            idx = torch.randperm(nc, generator=g)[i:i + batch]
            ti = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g)
            opt.zero_grad()
            lc, _, _ = vc.forward(P, Xc[idx], bits, quant=False)
            lt, _, gt = vc.forward(P, Xt[ti], bits, quant=False)
            if mode == "tier1":
                evade = (gt - med).abs().div(mad).mean()           # soft Tier-1 score -> minimize
            else:
                evade = (gt - centroid).pow(2).mean()
            loss = (F.cross_entropy(lc, yc[idx])
                    + F.cross_entropy(lt, torch.full((len(ti),), tgt, device=Xc.device))
                    + lam_mimic * evade)
            loss.backward(); opt.step()


def main():
    a = get_args()
    torch.manual_seed(a.seed); dev = "cpu"
    rng = np.random.default_rng(a.seed)
    X, y = vc.build(rng, a.n_per_class, a.sig_len)
    mu, sd = X.mean(), X.std() + 1e-8
    ntr = int(0.8 * len(y))
    trig = vc.trigger_pattern(a.sig_len, a.trig_freq, a.trig_amp)
    tgt, bits = a.target_class, a.qbits

    def to_t(arr): return torch.from_numpy(((arr - mu) / sd)).float().unsqueeze(1).to(dev)
    Xtr_raw, ytr = X[:ntr], y[:ntr]
    Xte_raw, yte = X[ntr:], y[ntr:]
    Xtr, Xte = to_t(Xtr_raw), to_t(Xte_raw)
    ytr_t, yte_t = torch.from_numpy(ytr).to(dev), torch.from_numpy(yte).to(dev)
    Xcal = to_t(Xtr_raw[:512])
    base_te = Xte[(yte != tgt)]
    trig_norm = (trig / sd).astype(np.float32)
    deltas = [float(x) for x in a.tier2_deltas.split(",")]

    cand = np.where(ytr != tgt)[0]
    Xt_in = to_t(Xtr_raw[cand] + trig[None, :])          # triggered training inputs
    X_tgt_clean = to_t(Xtr_raw[ytr == tgt])               # clean target-class inputs (the cluster)

    results = {"config": vars(a), "runs": []}
    for lam in [float(x) for x in a.lam_mimic.split(",")]:
        P = vc.init_params(a.seed, dev)
        train_adaptive(P, Xtr, ytr_t, Xt_in, X_tgt_clean, tgt, a.epochs, a.lr, a.batch, bits, a.seed, lam, a.attack_mode)
        wq, _, _ = vc.make_wq(P, bits); sc = vc.calibrate_act_scales(P, Xcal, bits, wq)
        ca, ai, af = vc.asr_clean_metrics(P, wq, sc, bits, Xte, yte_t, trig_norm, tgt)
        r = vc.qsentry_eval(P, wq, sc, bits, Xcal, base_te, trig_norm, deltas, a.tier1_clipK)
        g = r["gap"]
        results["runs"].append({"lam_mimic": lam, "clean_acc": round(ca, 4), "asr_int8": round(ai, 4),
                                "tier1_global": g["tier1_global_auroc"],
                                "tier1_classcond": g["tier1_classcond_auroc"],
                                "tier2": g["tier2_auroc"], "fused": g["fused_auroc"],
                                "tier2_tpr@5": g["tier2_tpr@5"], "tier1cc_tpr@5": g["tier1_classcond_tpr@5"]})

    print("\n" + "=" * 90)
    print("ADAPTIVE ATTACKER vs QSentry  (penultimate 'gap' layer; AUROC: 1.0 perfect, 0.5 none)")
    print("=" * 90)
    print(f"  {'lam_mimic':10} {'clean_acc':10} {'ASR_INT8':9} | {'T1-global':10} {'T1-classcond':13} {'TIER-2':8} {'Fused':8}")
    for r in results["runs"]:
        print(f"  {r['lam_mimic']:<10} {r['clean_acc']:<10} {r['asr_int8']:<9} | "
              f"{r['tier1_global']:<10} {r['tier1_classcond']:<13} {r['tier2']:<8} {r['fused']:<8}")

    print("\nVERDICT LOGIC")
    print("  * As lam_mimic rises: if ASR_INT8 stays high while T1-classcond DROPS -> attacker evades Tier-1.")
    print("  * In that evaded regime, if TIER-2 RISES well above 0.5 -> Tier-2 is VINDICATED (keep as novelty).")
    print("  * If TIER-2 stays ~0.5 while Tier-1 is evaded -> QSentry's novel tier is DEAD -> drop Tier-2,")
    print("    re-anchor novelty on measured on-device cost + quantization-aware evaluation.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.abspath(a.out)}\n")


if __name__ == "__main__":
    main()
