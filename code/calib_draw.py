#!/usr/bin/env python3
"""
calib_draw.py -- How much does the COMPOSITION of the calibration set change detection?

The monitor fits its per-class statistics on the clean calibration set the deployer already holds for
post-training quantization. That set is small, and how it is drawn turns out to matter more than its
size. Our earlier results drew it as the first 512 training windows; because the loaders emit windows
class by class, that draw covers only one or two of the classes on CWRU and MFPT. The monitor then has
per-class statistics for those classes alone, and -- more consequentially -- the pooled "global"
ablation is fitted on a nearly single-class sample, which makes it behave like a class-conditional
statistic and flatters it.

This script trains each model ONCE and then scores it under both draws, so the comparison isolates the
calibration draw and nothing else: same weights, same test split, same trigger.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python calib_draw.py --seeds 0,1,2
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q, defense as Def
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}
TARGET = {"cwru": 2, "mfpt": 1}
DRAWS = ("ordered", "representative")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cwru,mfpt")
    p.add_argument("--models", default="vibcnn,wdcnn")
    p.add_argument("--attacks", default="static@8,pq@8", help="<attack>@<bits> list")
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
    p.add_argument("--trig-kind", default="burst")
    p.add_argument("--lam-q", type=float, default=8.0)
    p.add_argument("--calib-size", type=int, default=512)
    p.add_argument("--out", default="../results/calib_draw")
    return p.parse_args()


def one(ds, mdl, attack, bits, seed, a):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(ds, a.data_dir, a.sig_len, seed)
    nclass = len(meta["classes"]); tgt = TARGET[ds]
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp, kind=a.trig_kind, seed=seed)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr = T(Xtr_raw); ytr_t = torch.from_numpy(ytr).to(dev)
    cand = np.where(ytr != tgt)[0]
    model = MODELS[mdl](n_classes=nclass, seed=seed).to(dev)

    if attack == "static":
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
    elif attack == "qcb":
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
        _, qint, qscale = Q.make_wq(model.p, bits, model.weight_keys)
        Xt = T(Xtr_raw[cand] + trig[None, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
        B.detach_fp32_keep_qcb(model, qint, qscale, Xtr, ytr_t, Xt, yt,
                               a.epochs_detach, a.lr, a.batch, seed)
    else:
        Xt = T(Xtr_raw[cand] + trig[None, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
        B.train_pq_backdoor(model, Xtr, ytr_t, Xt, yt, tgt, bits,
                            a.epochs + a.epochs_detach, a.lr, a.batch, seed, a.lam_q)

    base = T(Xte_raw)[torch.from_numpy(yte != tgt)]
    out = {}
    for draw in DRAWS:                     # same trained weights, two calibration draws
        idx = D.calib_indices(len(ytr), a.calib_size, seed, draw)
        Xcal = T(Xtr_raw[idx])
        wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
        scales = model.act_scales(wq, Xcal, bits)
        det = Def.evaluate(model, wq, scales, bits, Xcal, base, trig_norm, layer="gap")
        out[draw] = {"ccm": det["classcond_auroc"], "global": det["global_auroc"],
                     "ccm_tpr@1": det["classcond_tpr@1"], "global_tpr@1": det["global_tpr@1"],
                     "calib_classes": np.bincount(ytr[idx], minlength=nclass).tolist()}
    return out


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    cfgs = [(x.split("@")[0], int(x.split("@")[1])) for x in a.attacks.split(",")]
    rows = []
    for ds in a.datasets.split(","):
        for mdl in a.models.split(","):
            for attack, bits in cfgs:
                runs = [one(ds, mdl, attack, bits, s, a) for s in seeds]
                row = {"dataset": ds, "model": mdl, "attack": f"{attack}@INT{bits}",
                       "calib_classes": runs[0]["ordered"]["calib_classes"],
                       "calib_classes_repr": runs[0]["representative"]["calib_classes"]}
                for draw in DRAWS:
                    for k in ("ccm", "global", "ccm_tpr@1", "global_tpr@1"):
                        v = np.array([r[draw][k] for r in runs], float)
                        row[f"{draw}_{k}"] = round(float(v.mean()), 4)
                        row[f"{draw}_{k}_std"] = round(float(v.std()), 4)
                rows.append(row)
                print(f"[{ds}/{mdl}/{attack}@INT{bits}] "
                      f"ordered CCM {row['ordered_ccm']:.3f} vs global {row['ordered_global']:.3f} | "
                      f"repr CCM {row['representative_ccm']:.3f} vs global "
                      f"{row['representative_global']:.3f}", flush=True)

    hdr = (f"{'dataset':8}{'model':8}{'attack':16}"
           f"{'ordered: CCM/glob':22}{'repr: CCM/glob':22}{'ordered TPR@1':16}{'repr TPR@1'}")
    lines = [f"# Effect of the calibration draw on detection (mean over seeds={seeds}, "
             f"trigger={a.trig_kind}, calib={a.calib_size} windows)",
             "# same trained weights scored under both draws; TPR@1 is class-cond/global", "",
             hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['dataset']:8}{r['model']:8}{r['attack']:16}"
                     f"{r['ordered_ccm']:.2f} / {r['ordered_global']:.2f}{'':<12}"
                     f"{r['representative_ccm']:.2f} / {r['representative_global']:.2f}{'':<12}"
                     f"{r['ordered_ccm_tpr@1']:.2f} / {r['ordered_global_tpr@1']:.2f}{'':<6}"
                     f"{r['representative_ccm_tpr@1']:.2f} / {r['representative_global_tpr@1']:.2f}")
    report = "\n".join(lines); print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "calib_draw.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "rows": rows},
              open(os.path.join(a.out, "calib_draw.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/calib_draw.{{txt,json}}")


if __name__ == "__main__":
    main()
