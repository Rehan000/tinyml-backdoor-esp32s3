"""Deployable 1D-CNN for vibration fault classification.

Architecture (no BatchNorm -> clean quantization):
  conv1(1->16,k9,s2) - relu - conv2(16->32,k9,s2) - relu - conv3(32->64,k9,s4) - relu
  - global-avg-pool (64-d penultimate "gap") - fc(64->n_classes)

Weights live in a ParameterDict so (a) the bin-projection QCB can project individual weights
into their INT8 bins, and (b) a single functional core serves both the FP32 and simulated-INT8
deployment paths. `forward` is the plain FP32 path (used for ONNX export).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import quant as Q

WEIGHT_KEYS = ("c1w", "c2w", "c3w", "fcw")


class VibCNN(nn.Module):
    def __init__(self, n_classes=4, seed=0, width=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def k(shape, fan_in):
            return nn.Parameter(torch.randn(shape, generator=g) * np.sqrt(2.0 / fan_in))

        self.n_classes = n_classes
        self.width = width
        self.weight_keys = WEIGHT_KEYS
        c1, c2, c3 = 16 * width, 32 * width, 64 * width
        self.gap_dim = c3
        self.p = nn.ParameterDict({
            "c1w": k((c1, 1, 9), 9),            "c1b": nn.Parameter(torch.zeros(c1)),
            "c2w": k((c2, c1, 9), c1 * 9),      "c2b": nn.Parameter(torch.zeros(c2)),
            "c3w": k((c3, c2, 9), c2 * 9),      "c3b": nn.Parameter(torch.zeros(c3)),
            "fcw": k((n_classes, c3), c3),      "fcb": nn.Parameter(torch.zeros(n_classes)),
        })

    def core(self, x, bits=8, quant=False, act_scales=None, wq=None):
        """Functional forward. quant=True uses dequant weights `wq` + per-tensor activation quant.
        Returns (logits, feats{'c2','gap'}, gap_float) where gap_float is pre-quant penultimate."""
        W = wq if quant else self.p
        aq = (lambda h, key: Q.fq(h, act_scales[key], bits)) if quant else (lambda h, key: h)
        h = aq(x, "in")
        h = F.relu(F.conv1d(h, W["c1w"], W["c1b"], stride=2, padding=4)); h = aq(h, "c1")
        h = F.relu(F.conv1d(h, W["c2w"], W["c2b"], stride=2, padding=4)); c2 = h; h = aq(h, "c2")
        h = F.relu(F.conv1d(h, W["c3w"], W["c3b"], stride=4, padding=4)); h = aq(h, "c3")
        gap = h.mean(dim=2)
        gap_in = aq(gap, "gap") if quant else gap
        logits = gap_in @ W["fcw"].t() + W["fcb"]
        return logits, {"c2": c2.mean(dim=2), "gap": gap_in}, gap

    def forward(self, x):
        """Plain FP32 path (for training and ONNX export)."""
        return self.core(x)[0]

    def tail_fc(self, gap_val, wq, gap_scale, bits, scale_override=None):
        """Re-run only the fc tail with a (possibly perturbed) gap activation scale."""
        s = gap_scale if scale_override is None else scale_override
        gq = Q.fq(gap_val, s, bits)
        return gq @ wq["fcw"].t() + wq["fcb"]

    # ---- Straight-through-estimator (STE) quantized forward, for the PQ-backdoor joint training ----
    # Quantizes the model's OWN parameters (gradients flow to them), so the optimizer can actively
    # shape the *quantized* model's behaviour. Used only during training; deployment uses fresh PTQ
    # (make_wq + calibrate_act_scales) so the attack must survive real post-training quantization.
    def _ste_w(self, W, bits):
        dims = tuple(range(1, W.dim()))
        s = (W.detach().abs().amax(dim=dims, keepdim=True) + 1e-12) / Q.qmax(bits)
        q = torch.clamp(torch.round(W / s), -Q.qmax(bits), Q.qmax(bits)) * s
        return W + (q - W).detach()

    def _ste_a(self, a, bits):
        s = (a.detach().abs().max() + 1e-12) / Q.qmax(bits)   # per-tensor dynamic (detached) scale
        q = torch.clamp(torch.round(a / s), -Q.qmax(bits), Q.qmax(bits)) * s
        return a + (q - a).detach()

    def core_qste(self, x, bits):
        p = self.p
        c1w, c2w, c3w, fcw = (self._ste_w(p[k], bits) for k in ("c1w", "c2w", "c3w", "fcw"))
        h = self._ste_a(x, bits)
        h = F.relu(F.conv1d(h, c1w, p["c1b"], stride=2, padding=4)); h = self._ste_a(h, bits)
        h = F.relu(F.conv1d(h, c2w, p["c2b"], stride=2, padding=4)); h = self._ste_a(h, bits)
        h = F.relu(F.conv1d(h, c3w, p["c3b"], stride=4, padding=4)); h = self._ste_a(h, bits)
        gap = self._ste_a(h.mean(dim=2), bits)
        return gap @ fcw.t() + p["fcb"]

    def act_scales(self, wq, Xcal, bits):
        return Q.calibrate_act_scales(wq, Xcal, bits)

    def params_dict(self):
        return {k: v for k, v in self.p.items()}


# ---- shared straight-through-estimator helpers (used by the STE-quantized PQ-backdoor path) ----
def _ste_w(W, bits):
    dims = tuple(range(1, W.dim()))
    s = (W.detach().abs().amax(dim=dims, keepdim=True) + 1e-12) / Q.qmax(bits)
    q = torch.clamp(torch.round(W / s), -Q.qmax(bits), Q.qmax(bits)) * s
    return W + (q - W).detach()


def _ste_a(a, bits):
    s = (a.detach().abs().max() + 1e-12) / Q.qmax(bits)
    q = torch.clamp(torch.round(a / s), -Q.qmax(bits), Q.qmax(bits)) * s
    return a + (q - a).detach()


WDCNN_WEIGHT_KEYS = ("c1w", "c2w", "c3w", "c4w", "c5w", "fcw")


def build(name, n_classes=4, seed=0, width=1):
    return {"vibcnn": VibCNN, "wdcnn": WDCNN}[name](n_classes=n_classes, seed=seed, width=width)


def from_ckpt(ckpt, seed=0):
    """Rebuild the saved model (architecture stored under 'model', default vibcnn for old ckpts)."""
    m = build(ckpt.get("model", "vibcnn"), n_classes=ckpt["n_classes"], seed=seed,
              width=ckpt.get("width", 1))
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m


class WDCNN(nn.Module):
    """Deeper, wide-first-kernel 1-D CNN in the style of WDCNN (Zhang et al., 2017): a wide 64-tap
    first convolution followed by four small-kernel stages, global-average pooling, and a linear head.
    Same quantization/attack/defense interface as VibCNN (per-channel weight + per-tensor activation
    quant, a 64-d `gap` monitored layer), so the bit-width-threshold and CCM experiments run unchanged
    on this architecture. Five conv stages (vs three) test whether the findings generalize with depth."""

    def __init__(self, n_classes=4, seed=0, width=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def k(shape, fan_in):
            return nn.Parameter(torch.randn(shape, generator=g) * np.sqrt(2.0 / fan_in))

        self.n_classes = n_classes
        self.width = width
        self.weight_keys = WDCNN_WEIGHT_KEYS
        c1, c2, c3, c4, c5 = 16 * width, 32 * width, 64 * width, 64 * width, 64 * width
        self.gap_dim = c5
        self.p = nn.ParameterDict({
            "c1w": k((c1, 1, 64), 64),         "c1b": nn.Parameter(torch.zeros(c1)),
            "c2w": k((c2, c1, 3), c1 * 3),     "c2b": nn.Parameter(torch.zeros(c2)),
            "c3w": k((c3, c2, 3), c2 * 3),     "c3b": nn.Parameter(torch.zeros(c3)),
            "c4w": k((c4, c3, 3), c3 * 3),     "c4b": nn.Parameter(torch.zeros(c4)),
            "c5w": k((c5, c4, 3), c4 * 3),     "c5b": nn.Parameter(torch.zeros(c5)),
            "fcw": k((n_classes, c5), c5),     "fcb": nn.Parameter(torch.zeros(n_classes)),
        })
        # (stride, padding) per conv stage: wide first kernel then small strided stages
        self._sp = [(16, 24), (2, 1), (2, 1), (2, 1), (2, 1)]

    def core(self, x, bits=8, quant=False, act_scales=None, wq=None):
        W = wq if quant else self.p
        aq = (lambda h, key: Q.fq(h, act_scales[key], bits)) if quant else (lambda h, key: h)
        h = aq(x, "in")
        mid = None
        for i, (st, pad) in enumerate(self._sp, start=1):
            h = F.relu(F.conv1d(h, W[f"c{i}w"], W[f"c{i}b"], stride=st, padding=pad))
            if i == 3:
                mid = h
            h = aq(h, f"c{i}")
        gap = h.mean(dim=2)
        gap_in = aq(gap, "gap") if quant else gap
        logits = gap_in @ W["fcw"].t() + W["fcb"]
        return logits, {"c3": mid.mean(dim=2), "gap": gap_in}, gap

    def forward(self, x):
        return self.core(x)[0]

    def tail_fc(self, gap_val, wq, gap_scale, bits, scale_override=None):
        s = gap_scale if scale_override is None else scale_override
        gq = Q.fq(gap_val, s, bits)
        return gq @ wq["fcw"].t() + wq["fcb"]

    def core_qste(self, x, bits):
        p = self.p
        ws = {key: _ste_w(p[key], bits) for key in self.weight_keys}
        h = _ste_a(x, bits)
        for i, (st, pad) in enumerate(self._sp, start=1):
            h = F.relu(F.conv1d(h, ws[f"c{i}w"], p[f"c{i}b"], stride=st, padding=pad))
            h = _ste_a(h, bits)
        gap = _ste_a(h.mean(dim=2), bits)
        return gap @ ws["fcw"].t() + p["fcb"]

    @torch.no_grad()
    def act_scales(self, wq, Xcal, bits):
        h = Xcal
        scales = {"in": (Xcal.abs().max() + 1e-12) / Q.qmax(bits)}
        for i, (st, pad) in enumerate(self._sp, start=1):
            h = F.relu(F.conv1d(h, wq[f"c{i}w"], wq[f"c{i}b"], stride=st, padding=pad))
            scales[f"c{i}"] = (h.abs().max() + 1e-12) / Q.qmax(bits)
        scales["gap"] = (h.mean(dim=2).abs().max() + 1e-12) / Q.qmax(bits)
        return scales

    def params_dict(self):
        return {k: v for k, v in self.p.items()}
