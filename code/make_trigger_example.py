#!/usr/bin/env python3
"""
make_trigger_example.py -- Save one real CWRU faulty window plus the additive trigger (paper params)
for the trigger-visualization figure. No training: just load CWRU, fit the same normalizer the
experiments use, and apply make_trigger(freq, amp). Everything is in the normalized input domain the
model sees, so the figure matches the deployed pipeline exactly.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python make_trigger_example.py
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tinybd import data as D, backdoor as B


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cwru")
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--fault-class", default="inner")  # show a fault the trigger tries to mask
    p.add_argument("--out", default="../results/mechanism")
    return p.parse_args()


def main():
    a = get_args()
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, a.seed)
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp)
    trig_norm = (trig / norm.sd).astype(np.float32)

    cls = meta["classes"].index(a.fault_class)
    idx = np.where(yte == cls)[0][0]
    clean = norm(Xte_raw[idx:idx + 1])[0]          # normalized window the model sees
    triggered = clean + trig_norm

    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, "trigger_example.npz")
    np.savez(outp, clean=clean, trig=trig_norm, triggered=triggered,
             fault_class=a.fault_class, freq=a.trig_freq, amp=a.trig_amp,
             support=a.sig_len // 4)
    print(f"class={a.fault_class}  support=first {a.sig_len//4} samples  "
          f"trig_peak={np.abs(trig_norm).max():.3f} (norm. units)")
    print("wrote", os.path.abspath(outp))


if __name__ == "__main__":
    main()
