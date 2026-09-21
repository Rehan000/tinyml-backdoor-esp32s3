#!/usr/bin/env python3
"""
false_alarm.py -- Benign alarm rate of the deployed monitor at a fixed false-alarm budget (R2 minor
revision: "report TPR@FPR=1% and discuss the on-device false alarm rate during benign operation").

AUROC is threshold-free, which is the wrong shape for a device that runs continuously: what a deployer
needs to know is how often the monitor cries wolf when nothing is wrong. So we do what a deployer would
do -- fix the alarm threshold on the PTQ calibration set (the defender's own clean data, the same set
the monitor already uses), then measure on a held-out CLEAN test split how often that threshold actually
fires, and what detection rate it buys on triggered inputs. The realized benign rate is reported both as
a fraction and as alarms per hour of continuous inference at the measured on-device latency.

Note the two rates are not the same number: the threshold is set to fire on 1% of CALIBRATION windows,
and the realized rate on unseen clean windows is what we report.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python false_alarm.py --dataset cwru --model vibcnn --target-class 2 --seeds 0,1,2
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q
from tinybd.defense import _robust_stats, _score_classcond, _score_global
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}
LAT_US = {"vibcnn": 3677.2, "wdcnn": 2054.4}   # measured single-core INT8 latency, ESP32-S3 @240 MHz


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cwru", choices=["synthetic", "cwru", "paderborn", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=list(MODELS))
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--calib-draw", default="representative", choices=["representative", "ordered"],
                   help="how the clean calibration set is drawn (see tinybd.data.calib_indices)")
    p.add_argument("--split", default="window", choices=["window", "recording"])
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--attack", default="static", choices=["clean", "static"],
                   help="deployed model under test (static = the INT8-deployable backdoor)")
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=2)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--trig-kind", default="burst", choices=list(B.TRIGGER_KINDS))
    p.add_argument("--fprs", default="0.01,0.05", help="calibration false-alarm budgets")
    p.add_argument("--out", default="../results/false_alarm")
    return p.parse_args()


@torch.no_grad()
def one_seed(seed, a, budgets):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed, split=a.split)
    nclass = len(meta["classes"]); tgt = a.target_class; bits = a.bits
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp, kind=a.trig_kind, seed=seed)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    Xcal = T(Xtr_raw[D.calib_indices(len(ytr), 512, seed, a.calib_draw)])
    model = MODELS[a.model](n_classes=nclass, seed=seed).to(dev)
    with torch.enable_grad():
        if a.attack == "clean":
            B.train_float(model, Xtr, torch.from_numpy(ytr).to(dev), a.epochs, a.lr, a.batch, seed)
        else:
            Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
            B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)

    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
    scales = model.act_scales(wq, Xcal, bits)

    def feats(X):
        logits, f, _ = model.core(X, bits, quant=True, act_scales=scales, wq=wq)
        return f["gap"].cpu().numpy(), logits.argmax(1).cpu().numpy()

    Acal, pcal = feats(Xcal)
    gstat = _robust_stats(Acal)
    cstats = {c: (_robust_stats(Acal[pcal == c]) if (pcal == c).sum() >= 8 else gstat)
              for c in range(nclass)}
    K = 8.0
    s_cal_cc = _score_classcond(Acal, pcal, cstats, K)
    s_cal_gl = _score_global(Acal, *gstat, K)

    # benign: the FULL clean test split (every class, no trigger). triggered: non-target windows.
    Acl, pcl = feats(Xte)
    trig_t = torch.from_numpy(trig_norm).to(dev).view(1, 1, -1)
    Atr, ptr = feats(Xte[torch.from_numpy(yte != tgt)] + trig_t)
    s_cl_cc = _score_classcond(Acl, pcl, cstats, K); s_tr_cc = _score_classcond(Atr, ptr, cstats, K)
    s_cl_gl = _score_global(Acl, *gstat, K);         s_tr_gl = _score_global(Atr, *gstat, K)

    out = {"n_clean_test": int(len(s_cl_cc)), "n_triggered": int(len(s_tr_cc))}
    for fpr in budgets:
        for tag, s_cal, s_cl, s_tr in (("ccm", s_cal_cc, s_cl_cc, s_tr_cc),
                                       ("global", s_cal_gl, s_cl_gl, s_tr_gl)):
            thr = float(np.quantile(s_cal, 1.0 - fpr))
            out[f"{tag}_far@{fpr}"] = float((s_cl > thr).mean())
            out[f"{tag}_tpr@{fpr}"] = float((s_tr > thr).mean())
    return out


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    budgets = [float(x) for x in a.fprs.split(",")]
    runs = [one_seed(s, a, budgets) for s in seeds]
    print(f"[{a.dataset}/{a.model}] done over {len(seeds)} seeds")

    def ms(k):
        v = np.array([r[k] for r in runs], float); return v.mean(), v.std()

    lat_s = LAT_US[a.model] / 1e6
    lines = [f"# Benign alarm rate at a fixed false-alarm budget "
             f"(dataset={a.dataset}, model={a.model}, attack={a.attack}, split={a.split}, "
             f"seeds={seeds})",
             f"# threshold set on the {runs[0].get('n_cal', 512)}-window PTQ calibration set; rates "
             f"measured on {runs[0]['n_clean_test']} held-out clean windows "
             f"({runs[0]['n_triggered']} triggered)",
             f"# alarms/hour assumes continuous inference at {LAT_US[a.model]:.0f} us/inference "
             f"({1/lat_s:.0f} inferences/s)", "",
             f"{'budget':10}{'detector':10}{'benign FAR':16}{'alarms/hour':14}{'TPR':12}",
             "-" * 62]
    table = {}
    for fpr in budgets:
        for tag in ("ccm", "global"):
            far_m, far_s = ms(f"{tag}_far@{fpr}")
            tpr_m, tpr_s = ms(f"{tag}_tpr@{fpr}")
            per_hour = far_m * 3600.0 / lat_s
            lines.append(f"{fpr*100:<9.0f}%{tag:10}{far_m*100:>6.2f}+/-{far_s*100:<7.2f}%"
                         f"{per_hour:>10.0f}    {tpr_m:.2f}+/-{tpr_s:.2f}")
            table[f"{tag}@{fpr}"] = {"far": round(far_m, 4), "far_std": round(far_s, 4),
                                     "alarms_per_hour": round(per_hour, 1),
                                     "tpr": round(tpr_m, 4), "tpr_std": round(tpr_s, 4)}
    report = "\n".join(lines)
    print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.model}_{a.attack}_{a.split}"
    open(os.path.join(a.out, f"far_{tag}.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "table": table, "per_seed": runs},
              open(os.path.join(a.out, f"far_{tag}.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/far_{tag}.{{txt,json}}")


if __name__ == "__main__":
    main()
