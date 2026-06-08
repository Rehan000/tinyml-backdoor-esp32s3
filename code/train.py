#!/usr/bin/env python3
"""
train.py -- Phase-0 training driver.

Trains a VibCNN under one attack setting and saves artifacts for evaluate.py + ESP-DL export.

  --attack clean   : benign model
  --attack static  : conventional poisoned backdoor (present in FP32 and INT8)
  --attack qcb     : quantization-conditioned backdoor (dormant FP32, active INT8) via bin-projection

Examples:
  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python train.py --dataset synthetic --attack qcb
  ... --dataset cwru --data-dir ../data --attack static    # once real .mat data is in place
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q
from tinybd.model import build
from tinybd.export import to_onnx


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "cwru", "paderborn", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=["vibcnn", "wdcnn"])
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--attack", default="qcb", choices=["clean", "static", "qcb", "pq"])
    p.add_argument("--lam-q", type=float, default=1.0, help="PQ-backdoor: weight on the quantized-path loss")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--epochs-detach", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--width", type=int, default=1, help="channel multiplier (model capacity)")
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--out", default="../results/phase0")
    p.add_argument("--onnx", action="store_true", help="also export ONNX")
    return p.parse_args()


def main():
    a = get_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, a.seed)
    nclass = len(meta["classes"]); tgt = a.target_class; bits = a.bits
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    ytr_t, yte_t = torch.from_numpy(ytr).to(dev), torch.from_numpy(yte).to(dev)
    Xcal = T(Xtr_raw[:512])
    model = build(a.model, n_classes=nclass, seed=a.seed, width=a.width).to(dev)

    if a.attack == "clean":
        B.train_float(model, Xtr, ytr_t, a.epochs, a.lr, a.batch, a.seed)
    elif a.attack == "static":
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, a.seed)
    elif a.attack == "qcb":  # bin-projection: implant -> freeze INT8 -> detach FP32 within bins
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, a.seed)
        _, qint, qscale = Q.make_wq(model.p, bits, model.weight_keys)
        cand = np.where(ytr != tgt)[0]
        Xt = T(Xtr_raw[cand] + trig[None, :]); yt_true = torch.from_numpy(ytr[cand]).to(dev)
        B.detach_fp32_keep_qcb(model, qint, qscale, Xtr, ytr_t, Xt, yt_true,
                               a.epochs_detach, a.lr, a.batch, a.seed)
    else:  # pq: faithful INT8 PQ-backdoor via joint FP/quant optimization (STE)
        cand = np.where(ytr != tgt)[0]
        Xt = T(Xtr_raw[cand] + trig[None, :]); yt_true = torch.from_numpy(ytr[cand]).to(dev)
        epochs_pq = a.epochs + a.epochs_detach
        B.train_pq_backdoor(model, Xtr, ytr_t, Xt, yt_true, tgt, bits,
                            epochs_pq, a.lr, a.batch, a.seed, a.lam_q)

    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
    scales = model.act_scales(wq, Xcal, bits)
    clean, asr_i, asr_f = B.clean_and_asr(model, wq, scales, bits, Xte, yte_t, trig_norm, tgt)

    os.makedirs(a.out, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_classes": nclass, "width": a.width,
                "sig_len": a.sig_len, "bits": bits, "attack": a.attack, "model": a.model},
               os.path.join(a.out, "model.pt"))
    np.savez(os.path.join(a.out, "bundle.npz"),
             Xte=Xte.squeeze(1).cpu().numpy(), yte=yte, calib=Xcal.squeeze(1).cpu().numpy(),
             trig_norm=trig_norm, target=tgt, classes=np.array(meta["classes"]))
    np.save(os.path.join(a.out, "calib.npy"), Xcal.squeeze(1).cpu().numpy())  # for ESP-DL PTQ

    print(f"\n[train] dataset={a.dataset} ({meta['source']})  classes={meta['classes']}  attack={a.attack}")
    print(f"[train] clean_acc(INT8)={clean:.4f}  ASR_INT8={asr_i:.4f}  ASR_FP32={asr_f:.4f}")
    if a.attack in ("qcb", "pq"):
        ok = asr_i > 0.6 and (asr_i - asr_f) > 0.2
        tag = a.attack.upper()
        print(f"[train] {tag} {'conditioned (INT8>FP32 by %.2f)' % (asr_i - asr_f) if ok else 'NOT conditioned -- tune lam_q/epochs/amp'}")
    print(f"[train] saved -> {os.path.abspath(a.out)}")
    json.dump({"clean_acc": clean, "asr_int8": asr_i, "asr_fp32": asr_f, **vars(a)},
              open(os.path.join(a.out, "train_metrics.json"), "w"), indent=2)

    if a.onnx:  # non-fatal: export is a convenience, must never cost the run/report
        try:
            to_onnx(model, a.sig_len, os.path.join(a.out, "model.onnx"))
            print(f"[train] ONNX -> {os.path.join(a.out, 'model.onnx')}")
        except Exception as e:
            print(f"[train] ONNX export skipped ({type(e).__name__}: {e}); see export.py")
    print()


if __name__ == "__main__":
    main()
