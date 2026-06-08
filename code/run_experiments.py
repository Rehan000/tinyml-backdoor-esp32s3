#!/usr/bin/env python3
"""
run_experiments.py -- Paper results matrix: attacks x bit-widths x seeds, with mean +/- std.

Produces the core evaluation table (clean acc, ASR INT/FP32, C3 detector AUROC class-cond + global
ablation, Tier-2 negative result) averaged over seeds. Runs in-process (imports tinybd). Swap
--dataset cwru/paderborn later -- the same harness produces the real-data table unchanged.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python run_experiments.py --seeds 0,1,2
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q, defense as Def
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}

# (attack, bits) configs to evaluate. static@INT8 works; qcb conditioned@INT4; pq is the INT8 attempt.
CONFIGS = [("static", 8), ("qcb", 4), ("pq", 8)]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "cwru", "paderborn", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=list(MODELS))
    p.add_argument("--data-dir", default="../data")
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
    p.add_argument("--lam-q", type=float, default=8.0)
    p.add_argument("--out", default="../results/experiments")
    return p.parse_args()


def one_run(attack, bits, seed, a):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed)
    nclass = len(meta["classes"]); tgt = a.target_class
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    ytr_t, yte_t = torch.from_numpy(ytr).to(dev), torch.from_numpy(yte).to(dev)
    Xcal = T(Xtr_raw[:512]); cand = np.where(ytr != tgt)[0]
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
        B.train_pq_backdoor(model, Xtr, ytr_t, Xt, yt, tgt, bits, a.epochs + a.epochs_detach, a.lr, a.batch, seed, a.lam_q)

    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
    scales = model.act_scales(wq, Xcal, bits)
    clean, asr_i, asr_f = B.clean_and_asr(model, wq, scales, bits, Xte, yte_t, trig_norm, tgt)
    base = Xte[(yte != tgt)]
    det = Def.evaluate(model, wq, scales, bits, Xcal, base, trig_norm, layer="gap")
    neg = Def.tier2_diagnostic(model, wq, scales, bits, base, trig_norm,
                               [-0.12, -0.08, -0.04, 0.04, 0.08, 0.12])
    return {"clean": clean, "asr_int": asr_i, "asr_fp": asr_f,
            "auroc_cc": det["classcond_auroc"], "auroc_glob": det["global_auroc"],
            "tpr5": det["classcond_tpr@5"], "tier2": neg["tier2_auroc"]}


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return v.mean(), v.std()


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    table = []
    for attack, bits in CONFIGS:
        runs = [one_run(attack, bits, s, a) for s in seeds]
        row = {"attack": attack, "bits": bits, "n_seeds": len(seeds)}
        for k in ("clean", "asr_int", "asr_fp", "auroc_cc", "auroc_glob", "tpr5", "tier2"):
            m, sd = agg(runs, k); row[k] = round(m, 4); row[k + "_std"] = round(sd, 4)
        table.append(row)
        print(f"[{attack}@INT{bits}] done over {len(seeds)} seeds")

    hdr = f"{'attack@bits':14} {'clean':12} {'ASR_int':12} {'ASR_fp':12} {'C3 AUROC':12} {'(global)':10} {'Tier2':8}"
    lines = ["# Experiment matrix (mean +/- std over seeds=%s, dataset=%s)" % (seeds, a.dataset), "", hdr,
             "-" * len(hdr)]
    for r in table:
        lines.append(f"{r['attack']+'@INT'+str(r['bits']):14} "
                     f"{r['clean']:.2f}+/-{r['clean_std']:.2f}   "
                     f"{r['asr_int']:.2f}+/-{r['asr_int_std']:.2f}   "
                     f"{r['asr_fp']:.2f}+/-{r['asr_fp_std']:.2f}   "
                     f"{r['auroc_cc']:.2f}+/-{r['auroc_cc_std']:.2f}   "
                     f"{r['auroc_glob']:.2f}     {r['tier2']:.2f}")
    report = "\n".join(lines)
    print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "summary.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "table": table}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/summary.{{txt,json}}")


if __name__ == "__main__":
    main()
