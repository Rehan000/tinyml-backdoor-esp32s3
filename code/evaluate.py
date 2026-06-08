#!/usr/bin/env python3
"""
evaluate.py -- Phase-0 evaluation driver.

Loads artifacts from train.py and runs the C3 detector (class-conditional INT8 activation
monitor) on clean-vs-triggered inputs, plus the dropped Tier-2 probe as the measured negative
result. Detection AUROC / TPR@FPR are the host-side numbers; on-device latency/energy/RAM come
from the firmware harness (firmware/qsentry_bench.c) once you flash a device.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python evaluate.py --run ../results/phase0
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import quant as Q, defense as Def
from tinybd.model import VibCNN


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="../results/phase0", help="dir produced by train.py")
    p.add_argument("--layer", default="gap", choices=["gap", "c2"])
    p.add_argument("--tier2-deltas", default="-0.12,-0.08,-0.04,0.04,0.08,0.12")
    return p.parse_args()


def main():
    a = get_args(); dev = "cpu"
    ckpt = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    bits, sig_len = ckpt["bits"], ckpt["sig_len"]
    model = VibCNN(n_classes=ckpt["n_classes"], seed=0, width=ckpt.get("width", 1)).to(dev)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    b = np.load(os.path.join(a.run, "bundle.npz"), allow_pickle=True)
    Xte = torch.from_numpy(b["Xte"]).float().unsqueeze(1).to(dev)
    yte = torch.from_numpy(b["yte"]).to(dev)
    Xcal = torch.from_numpy(b["calib"]).float().unsqueeze(1).to(dev)
    trig_norm = b["trig_norm"].astype(np.float32)
    tgt = int(b["target"])

    wq, _, _ = Q.make_wq(model.p, bits)
    scales = Q.calibrate_act_scales(wq, Xcal, bits)
    base_clean = Xte[(yte != tgt)]  # non-target clean test inputs (trigger -> misclassification)

    det = Def.evaluate(model, wq, scales, bits, Xcal, base_clean, trig_norm, layer=a.layer)
    deltas = [float(x) for x in a.tier2_deltas.split(",")]
    neg = Def.tier2_diagnostic(model, wq, scales, bits, base_clean, trig_norm, deltas)

    print("\n" + "=" * 76)
    print(f"Phase-0 evaluation  (attack={ckpt['attack']}, layer={a.layer}, INT{bits})")
    print("=" * 76)
    print(f"  C3 class-conditional monitor : AUROC={det['classcond_auroc']}  "
          f"TPR@1%={det['classcond_tpr@1']}  TPR@5%={det['classcond_tpr@5']}")
    print(f"  (ablation) global monitor    : AUROC={det['global_auroc']}")
    print(f"  (negative result) Tier-2     : AUROC={neg['tier2_auroc']}  <- expected ~0.5, dropped")
    print("\n  On-device latency / energy / RAM: measure with firmware/qsentry_bench.c (PPK2).")
    print(f"  Saved -> {os.path.abspath(a.run)}/eval_metrics.json\n")
    json.dump({**det, **neg, "attack": ckpt["attack"], "layer": a.layer},
              open(os.path.join(a.run, "eval_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
