#!/usr/bin/env python3
"""
poisoned_calib.py -- WP5: CCM sensitivity to a POISONED calibration set (R5.3 / R1.3).

The CCM calibrates its per-class median/MAD from the deployer's PTQ calibration set. Reviewers ask:
what if the (outsourced) attacker also poisoned that set? We inject a fraction of triggered inputs
into the calibration data (they are predicted as the target class and pollute that class's cluster),
recompute the CCM statistics, and measure detection AUROC on a CLEAN test split. Because median/MAD is
a robust estimator (breakdown ~50%), we expect graceful degradation up to a point -- quantified here.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python poisoned_calib.py --dataset cwru --target-class 2 --seeds 0,1,2
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q, defense as Def
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cwru", choices=["synthetic", "cwru", "paderborn", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=list(MODELS))
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=2)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--trig-kind", default="burst", choices=list(B.TRIGGER_KINDS))
    p.add_argument("--calib-poison", default="0,0.005,0.01,0.02,0.05,0.1,0.2", help="calib-set poison fractions")
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--out", default="../results/poisoned_calib")
    return p.parse_args()


def one_seed(seed, a):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed)
    nclass = len(meta["classes"]); tgt = a.target_class; bits = a.bits
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp, kind=a.trig_kind, seed=seed)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr = T(Xtr_raw); ytr_t = torch.from_numpy(ytr).to(dev)
    yte_t = torch.from_numpy(yte).to(dev)

    # static backdoor
    Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
    model = MODELS[a.model](n_classes=nclass, seed=seed).to(dev)
    B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)

    # class-representative clean calibration pool (random draw, not the class-ordered first N)
    calib_idx = rng.permutation(len(ytr))[:a.calib_size]
    clean_pool = Xtr_raw[calib_idx].copy()
    cand = np.where(ytr != tgt)[0]
    trig_pool_raw = Xtr_raw[cand] + trig[None, :]           # triggered (predicted as target)
    Xcal_clean = T(clean_pool)
    scales = model.act_scales(wq, Xcal_clean, bits)          # act scales from CLEAN calib (PTQ is clean)
    base = T(Xte_raw)[torch.from_numpy(yte != tgt)]

    rates = [float(x) for x in a.calib_poison.split(",")]
    out = {}
    for r in rates:
        npois = int(r * len(clean_pool))
        if npois == 0:
            Xcal = Xcal_clean
        else:
            keep = clean_pool[npois:]
            inj = trig_pool_raw[rng.choice(len(trig_pool_raw), npois, replace=False)]
            Xcal = T(np.concatenate([inj, keep], 0))
        rep = Def.evaluate_baselines(model, wq, scales, bits, Xcal, base, trig_norm, layer="gap")
        det = Def.evaluate(model, wq, scales, bits, Xcal, base, trig_norm, layer="gap")
        out[r] = {"ccm": det["classcond_auroc"], "ccm_tpr@1": det["classcond_tpr@1"]}
        for m in ("global", "mahalanobis", "centroid_l2", "knn_oneclass"):
            out[r][m] = rep["methods"][m]["auroc"]
            out[r][m + "_tpr@1"] = rep["methods"][m]["tpr@1"]
    return out


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rates = [float(x) for x in a.calib_poison.split(",")]
    per = {r: [] for r in rates}
    for seed in seeds:
        res = one_seed(seed, a)
        for r in rates:
            per[r].append(res[r])
        print(f"  seed {seed} done", flush=True)

    METHODS = ["ccm", "mahalanobis", "centroid_l2", "knn_oneclass", "global"]
    hdr = f"{'calib_poison':14}" + "".join(f"{m:16}" for m in METHODS)
    lines = [f"# Detection AUROC vs poisoned-calibration fraction "
             f"(mean+/-std over seeds={seeds}, dataset={a.dataset}, model={a.model})", "",
             "# CCM's median/MAD is a robust estimator (breakdown ~50%); the centroid and Mahalanobis",
             "# baselines use means and variances, so this is where the robust statistic should pay off.",
             "", hdr, "-" * len(hdr)]
    table = []
    for r in rates:
        row = {"calib_poison": r}
        line = f"{r:<14}"
        for m in METHODS:
            v = np.array([d[m] for d in per[r]], float)
            t = np.array([d[m + "_tpr@1"] for d in per[r]], float)
            line += f"{v.mean():.2f}+/-{v.std():.2f}   "
            row[m] = round(float(v.mean()), 4)
            row[m + "_std"] = round(float(v.std()), 4)
            row[m + "_tpr@1"] = round(float(t.mean()), 4)
        row["ccm_cc"] = row["ccm"]          # keep the key the existing table generator reads
        row["std"] = row["ccm_std"]
        lines.append(line)
        table.append(row)
    lines += ["", "TPR@1% at each fraction:", hdr, "-" * len(hdr)]
    for r, row in zip(rates, table):
        lines.append(f"{r:<14}" + "".join(f"{row[m + '_tpr@1']:.2f}{'':<11}" for m in METHODS))
    report = "\n".join(lines); print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.model}"
    open(os.path.join(a.out, f"poisoned_calib_{tag}.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "table": table}, open(os.path.join(a.out, f"poisoned_calib_{tag}.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/poisoned_calib_{tag}.{{txt,json}}")


if __name__ == "__main__":
    main()
