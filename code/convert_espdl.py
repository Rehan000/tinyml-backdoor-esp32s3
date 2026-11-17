#!/usr/bin/env python3
"""
convert_espdl.py -- Quantize a trained model to an on-device .espdl model (ESP-DL / esp-ppq).

Run in the `espdl` conda env (has esp-ppq + torch), NOT qsentry:
  conda activate espdl
  KMP_DUPLICATE_LIB_OK=TRUE python convert_espdl.py --run ../results/deploy_r3 --target esp32s3 \
      --with-features --simulate

Uses the saved PTQ calibration set (calib.npy) so device INT8 scales match the host simulation.
Exports firmware/main/model/model.espdl (+ a quantization-error report).

--with-features  export a TWO-output graph: (logits, gap). `gap` is the 64-d pooled feature vector
                 the monitor reads -- the same tensor that feeds the classifier, so exposing it adds
                 no compute. This is what lets the firmware run the monitor on the real features
                 and measure detection on the device (Reviewer 4, minor revision 2).
--simulate       run esp-ppq's own quantized-graph executor (the same power-of-two INT8 arithmetic the
                 device runs) on the calibration set, the clean test set and the triggered test set,
                 and save the logits/features to <run>/espdl_sim.npz. export_monitor.py fits the
                 monitor tables on these features, so the tables match the device's quantization,
                 and the on-device features/scores can be checked against them element by element.
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tinybd.model import from_ckpt
from esp_ppq.api import espdl_quantize_torch


class WithFeatures(torch.nn.Module):
    """FP32 forward that also returns the pooled features (the monitor's input)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        logits, _, gap = self.m.core(x)      # FP32 path: gap = mean over time of the last conv
        return logits, gap


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="../results/phase0")
    p.add_argument("--target", default="esp32s3")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--out", default="../firmware/main/model/model.espdl")
    p.add_argument("--calib-batch", type=int, default=32)
    p.add_argument("--with-features", action="store_true")
    p.add_argument("--simulate", action="store_true")
    return p.parse_args()


@torch.no_grad()
def simulate(graph, X, batch=64):
    """Forward X through the quantized esp-ppq graph; returns a list of numpy outputs in graph order."""
    from esp_ppq.executor import TorchExecutor
    ex = TorchExecutor(graph=graph, device="cpu")
    outs = None
    for i in range(0, len(X), batch):
        o = ex.forward(inputs=X[i:i + batch])
        o = [t.detach().cpu().numpy() for t in o]
        outs = o if outs is None else [np.concatenate([a, b], 0) for a, b in zip(outs, o)]
    return outs


def main():
    a = get_args(); dev = "cpu"
    ckpt = torch.load(os.path.join(a.run, "model.pt"), map_location=dev, weights_only=False)
    sig_len = ckpt["sig_len"]
    base = from_ckpt(ckpt).eval()
    model = WithFeatures(base).eval() if a.with_features else base

    calib = np.load(os.path.join(a.run, "calib.npy")).astype(np.float32)  # (N, sig_len)
    X = torch.from_numpy(calib).unsqueeze(1)                              # (N,1,sig_len)
    loader = DataLoader(TensorDataset(X), batch_size=a.calib_batch, shuffle=False)
    collate = lambda batch: torch.stack([b[0] for b in batch]).to(dev)
    calib_steps = max(8, len(loader))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[convert] model: {ckpt.get('model','vibcnn')} width={ckpt.get('width',1)} sig_len={sig_len} "
          f"-> {a.target} INT{a.bits}  outputs={'logits+gap' if a.with_features else 'logits'}")
    graph = espdl_quantize_torch(
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
    print(f"[convert] graph outputs: {list(graph.outputs.keys())}")

    if a.simulate:
        b = np.load(os.path.join(a.run, "bundle.npz"), allow_pickle=True)
        Xte = torch.from_numpy(b["Xte"].astype(np.float32)).unsqueeze(1)
        trig = torch.from_numpy(b["trig_norm"].astype(np.float32)).view(1, 1, -1)
        res = {}
        for tag, Xs in (("cal", X), ("te", Xte), ("tr", Xte + trig)):
            outs = simulate(graph, Xs)
            res[f"{tag}_logits"] = outs[0]
            if len(outs) > 1:
                res[f"{tag}_gap"] = outs[1]
        with torch.no_grad():
            res["te_logits_fp32"] = base(Xte).numpy()
        names = list(graph.outputs.keys())
        np.savez(os.path.join(a.run, "espdl_sim.npz"), output_names=np.array(names), **res)
        acc = float((res["te_logits"].argmax(1) == b["yte"]).mean())
        agree = float((res["te_logits"].argmax(1) == res["te_logits_fp32"].argmax(1)).mean())
        print(f"[convert] esp-ppq sim: test acc={acc:.4f}  sim-INT8 vs FP32 agreement={agree:.4f}  "
              f"-> {os.path.join(a.run, 'espdl_sim.npz')}")


if __name__ == "__main__":
    main()
