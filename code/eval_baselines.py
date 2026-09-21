#!/usr/bin/env python3
"""
eval_baselines.py -- CCM vs lightweight MCU-feasible detection baselines (revision WP2; R1.10/R3.4/R5.1).

For each attack config, trains the model, quantizes (fresh PTQ), then runs `defense.evaluate_baselines`
to score CCM (class-conditional robust median/MAD), its global ablation, class-conditional Mahalanobis,
class-centroid (L2 + cosine), a thresholded activation-norm, and a one-class kNN detector -- reporting
detection AUROC / TPR@5% AND on-device cost (stored floats, flops/inference), averaged over seeds.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python eval_baselines.py --dataset cwru --model vibcnn --target-class 2 --seeds 0,1,2
  # trigger-robustness sweep (WP3): re-run with --trig-kind tone|spectral|distributed
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q, defense as Def
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}
CONFIGS = [("static", 8), ("qcb", 4), ("pq", 8)]
METHOD_ORDER = ["ccm_classcond", "global", "mahalanobis", "centroid_l2",
                "centroid_cos", "act_norm", "knn_oneclass"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "cwru", "paderborn", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=list(MODELS))
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--calib-draw", default="representative", choices=["representative", "ordered"],
                   help="how the clean calibration set is drawn (see tinybd.data.calib_indices)")
    p.add_argument("--split", default="window", choices=["window", "recording"],
                   help="train/test protocol: 'window' (default) or recording-level hold-out")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--epochs-detach", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--trig-kind", default="burst", choices=list(B.TRIGGER_KINDS))
    p.add_argument("--lam-q", type=float, default=8.0)
    p.add_argument("--out", default="../results/baselines")
    return p.parse_args()


def train_attacked(attack, bits, seed, a):
    """Return (model, wq, act_scales, bits, base_clean, trig_norm, tgt) for the given attack."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed, split=a.split)
    nclass = len(meta["classes"]); tgt = a.target_class
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp, kind=a.trig_kind, seed=seed)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr = T(Xtr_raw); ytr_t = torch.from_numpy(ytr).to(dev)
    Xcal = T(Xtr_raw[D.calib_indices(len(ytr), 512, seed, a.calib_draw)])
    cand = np.where(ytr != tgt)[0]
    model = MODELS[a.model](n_classes=nclass, seed=seed).to(dev)

    if attack == "static":
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
    elif attack == "qcb":
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
        _, qint, qscale = Q.make_wq(model.p, bits, model.weight_keys)
        Xt = T(Xtr_raw[cand] + trig[None, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
        B.detach_fp32_keep_qcb(model, qint, qscale, Xtr, ytr_t, Xt, yt, a.epochs_detach, a.lr, a.batch, seed)
    else:  # pq
        Xt = T(Xtr_raw[cand] + trig[None, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
        B.train_pq_backdoor(model, Xtr, ytr_t, Xt, yt, tgt, bits, a.epochs + a.epochs_detach,
                            a.lr, a.batch, seed, a.lam_q)

    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
    scales = model.act_scales(wq, Xcal, bits)
    base = T(Xte_raw)[torch.from_numpy(yte != tgt)]
    return model, wq, scales, bits, Xcal, base, trig_norm, tgt


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    results = {}   # attack -> method -> list of (auroc, tpr5, tpr1); cost captured once
    cost = {}; raw_scores = {}
    for attack, bits in CONFIGS:
        per_method = {m: [] for m in METHOD_ORDER}
        for seed in seeds:
            model, wq, scales, b, Xcal, base, trig_norm, tgt = train_attacked(attack, bits, seed, a)
            rep = Def.evaluate_baselines(model, wq, scales, b, Xcal, base, trig_norm, layer="gap",
                                         return_scores=True)
            for m in METHOD_ORDER:
                r = rep["methods"][m]
                per_method[m].append((r["auroc"], r["tpr@5"], r["tpr@1"]))
                cost[m] = (r["params"], r["flops"])
                raw_scores[f"{attack}@INT{bits}|seed{seed}|{m}"] = rep["_scores"][m]
            raw_scores[f"{attack}@INT{bits}|seed{seed}|labels"] = rep["_labels"]
        results[f"{attack}@INT{bits}"] = per_method
        print(f"[{attack}@INT{bits}] done over {len(seeds)} seeds")

    # ---- render table: rows = methods, columns = attack configs (AUROC mean+/-std) ----
    cfgs = [f"{atk}@INT{b}" for atk, b in CONFIGS]
    lines = [f"# CCM vs lightweight baselines (mean+/-std over seeds={seeds}, "
             f"dataset={a.dataset}, model={a.model}, trigger={a.trig_kind})", ""]

    def block(title, col):
        out = [title]
        hdr = f"{'method':16}" + "".join(f"{c:16}" for c in cfgs) + f"{'params':>9}{'flops':>9}"
        out += [hdr, "-" * len(hdr)]
        for m in METHOD_ORDER:
            row = f"{m:16}"
            for c in cfgs:
                v = np.array([x[col] for x in results[c][m]], float)
                row += f"{v.mean():.2f}+/-{v.std():.2f}   "
            p_, f_ = cost.get(m, (0, 0))
            out.append(row + f"{p_:>9}{f_:>9}")
        return out + [""]

    lines += block("## AUROC", 0)
    lines += block("## TPR @ 1% FPR  (embedded alarm budget)", 2)
    lines += block("## TPR @ 5% FPR", 1)
    report = "\n".join(lines)
    print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.model}_{a.trig_kind}"
    open(os.path.join(a.out, f"baselines_{tag}.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "results": results, "cost": cost,
               "metric_order": ["auroc", "tpr@5", "tpr@1"]},
              open(os.path.join(a.out, f"baselines_{tag}.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.out, f"scores_{tag}.npz"), **raw_scores)
    print(f"Saved -> {os.path.abspath(a.out)}/baselines_{tag}.{{txt,json}} (+ scores_{tag}.npz)")


if __name__ == "__main__":
    main()
