#!/usr/bin/env python3
"""
convert_espdl.py -- Quantize a trained VibCNN to an on-device .espdl model (ESP-DL / esp-ppq).

Run in the `espdl` conda env (has esp-ppq + torch), NOT qsentry:
  conda activate espdl
  KMP_DUPLICATE_LIB_OK=TRUE python convert_espdl.py --run ../results/phase0 --target esp32s3

Uses the saved PTQ calibration set (calib.npy) so device INT8 scales match the host simulation.
Exports firmware/main/model/model.espdl (+ a quantization-error report).
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tinybd.model import from_ckpt
from esp_ppq.api import espdl_quantize_torch


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="../results/phase0")
    p.add_argument("--target", default="esp32s3")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--out", default="../firmware/main/model/model.espdl")
    p.add_argument("--calib-batch", type=int, default=32)
    return p.parse_args()


def main():
    a = get_args(); dev = "cpu"
    ckpt = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    sig_len = ckpt["sig_len"]
    model = from_ckpt(ckpt)

    calib = np.load(os.path.join(a.run, "calib.npy")).astype(np.float32)  # (N, sig_len)
    X = torch.from_numpy(calib).unsqueeze(1)                              # (N,1,sig_len)
    loader = DataLoader(TensorDataset(X), batch_size=a.calib_batch, shuffle=False)
    collate = lambda batch: torch.stack([b[0] for b in batch]).to(dev)
    calib_steps = max(8, len(loader))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[convert] model: {ckpt.get('model','vibcnn')} width={ckpt.get('width',1)} sig_len={sig_len} -> {a.target} INT{a.bits}")
    espdl_quantize_torch(
        model=model,
        espdl_export_file=os.path.abspath(a.out),
        calib_dataloader=loader,
        calib_steps=calib_steps,
        input_shape=[1, 1, sig_len],
        target=a.target,
        num_of_bits=a.bits,
        collate_fn=collate,
        device=dev,
        error_report=True,
        verbose=1,
    )
    print(f"[convert] wrote {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
