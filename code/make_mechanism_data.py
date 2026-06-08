#!/usr/bin/env python3
"""
make_mechanism_data.py -- Dump the CCM detector's per-input scores (clean vs triggered) for the
figure that illustrates *why* the monitor is class-conditional. Reuses the exact tinybd training and
scoring path from run_experiments.one_run, but extracts the raw score arrays (global vs class-
conditional) instead of only the AUROC summary, so the plotted distributions are faithful.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python make_mechanism_data.py --dataset cwru

Writes ../results/mechanism/scores_<dataset>.npz and prints the resulting AUROCs (cross-check vs
the paper's detection table).
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q, defense as Def
from tinybd.metrics import auroc
from tinybd.model import VibCNN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cwru", choices=["synthetic", "cwru", "paderborn"])
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--epochs-detach", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--layer", default="gap")
    p.add_argument("--K", type=float, default=8.0)
    p.add_argument("--out", default="../results/mechanism")
    return p.parse_args()


@torch.no_grad()
def scores_for(model, wq, scales, bits, Xcal, base_clean, trig_norm, layer, K):
    """Per-input global and class-conditional scores for clean (neg) and triggered (pos) inputs."""
    dev = base_clean.device
    trig = torch.from_numpy(trig_norm).to(dev).view(1, 1, -1)
    Xtr = base_clean + trig
    gstat, cstats = Def.calibrate(model, wq, scales, bits, Xcal, layer, model.n_classes, K)

    lcl, fcl, _ = model.core(base_clean, bits, quant=True, act_scales=scales, wq=wq)
    ltr, ftr, _ = model.core(Xtr, bits, quant=True, act_scales=scales, wq=wq)
    Acl, Atr = fcl[layer].cpu().numpy(), ftr[layer].cpu().numpy()
    pcl, ptr = lcl.argmax(1).cpu().numpy(), ltr.argmax(1).cpu().numpy()
    m, mad = gstat
    return {
        "glob_clean": Def._score_global(Acl, m, mad, K),
        "glob_trig":  Def._score_global(Atr, m, mad, K),
        "cc_clean":   Def._score_classcond(Acl, pcl, cstats, K),
        "cc_trig":    Def._score_classcond(Atr, ptr, cstats, K),
    }


def train_qcb(a, seed):
    """Reproduce the conditioned attack (qcb @ INT4) exactly as run_experiments.one_run does."""
    bits = 4
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed)
    nclass = len(meta["classes"]); tgt = meta["classes"].index("normal")
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xcal = T(Xtr_raw[:512]); cand = np.where(ytr != tgt)[0]
    model = VibCNN(n_classes=nclass, seed=seed).to(dev)

    Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
    B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
    _, qint, qscale = Q.make_wq(model.p, bits)
    Xt = T(Xtr_raw[cand] + trig[None, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
    B.detach_fp32_keep_qcb(model, qint, qscale, Xtr, ytr_t, Xt, yt, a.epochs_detach, a.lr, a.batch, seed)

    wq, _, _ = Q.make_wq(model.p, bits)
    scales = Q.calibrate_act_scales(wq, Xcal, bits)
    base = Xte[(yte != tgt)]   # faulty inputs that the trigger tries to mask as "normal"
    return model, wq, scales, bits, Xcal, base, trig_norm, meta, tgt


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    pool = {"glob_clean": [], "glob_trig": [], "cc_clean": [], "cc_trig": []}
    meta = tgt = None
    for seed in seeds:
        model, wq, scales, bits, Xcal, base, trig_norm, meta, tgt = train_qcb(a, seed)
        s = scores_for(model, wq, scales, bits, Xcal, base, trig_norm, a.layer, a.K)
        for k in pool:
            pool[k].append(s[k])
        lab = np.r_[np.zeros(len(s["glob_clean"])), np.ones(len(s["glob_trig"]))]
        ag = auroc(np.r_[s["glob_clean"], s["glob_trig"]], lab)
        ac = auroc(np.r_[s["cc_clean"], s["cc_trig"]], lab)
        print(f"  seed {seed}: AUROC global={ag:.3f}  class-conditional={ac:.3f}")
    pool = {k: np.concatenate(v) for k, v in pool.items()}

    labels = np.r_[np.zeros(len(pool["glob_clean"])), np.ones(len(pool["glob_trig"]))]
    auroc_glob = auroc(np.r_[pool["glob_clean"], pool["glob_trig"]], labels)
    auroc_cc   = auroc(np.r_[pool["cc_clean"], pool["cc_trig"]], labels)
    print(f"dataset={a.dataset} attack=qcb@INT{bits} target={meta['classes'][tgt]} seeds={seeds}")
    print(f"  POOLED AUROC  global={auroc_glob:.3f}   class-conditional={auroc_cc:.3f}")
    print(f"  n_clean={len(pool['glob_clean'])}  n_trig={len(pool['glob_trig'])}")

    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, f"scores_{a.dataset}.npz")
    np.savez(outp, auroc_glob=auroc_glob, auroc_cc=auroc_cc,
             classes=np.array(meta["classes"]), target=tgt, seeds=np.array(seeds), **pool)
    print("wrote", os.path.abspath(outp))


if __name__ == "__main__":
    main()
