"""Simulated INT8 quantization: per-channel symmetric weights, per-tensor symmetric activations.

Mirrors the validated de-risk pipeline. `make_wq` freezes a model's deployed INT8 weights
(used by the bin-projection QCB and by the deployed/quant forward path); `calibrate_act_scales`
sets per-tensor activation scales from a clean calibration set (the PTQ calibration data).
"""
import torch
import torch.nn.functional as F


def qmax(bits):
    return (1 << (bits - 1)) - 1


def fq(x, scale, bits):
    """Symmetric fake-quant -> dequantized tensor."""
    return torch.clamp(torch.round(x / scale), -qmax(bits), qmax(bits)) * scale


def quantize_w(W, bits):
    """Per-output-channel symmetric weight quantization. Returns (q_int, scale, dequant)."""
    dims = tuple(range(1, W.dim()))
    s = (W.detach().abs().amax(dim=dims, keepdim=True) + 1e-12) / qmax(bits)
    q = torch.clamp(torch.round(W.detach() / s), -qmax(bits), qmax(bits))
    return q, s, q * s


def make_wq(params, bits, weight_keys=("c1w", "c2w", "c3w", "fcw")):
    """Freeze deployed INT8 weights from a params dict/ParameterDict, for any architecture whose
    weights are named `<layer>w` with matching bias `<layer>b` (default = VibCNN's four layers).
    Returns (wq: dequant weights+biases, qint: per-weight integers, qscale: per-channel scales)."""
    wq, qint, qscale = {}, {}, {}
    for key in weight_keys:
        q, s, deq = quantize_w(params[key], bits)
        wq[key] = deq.clone(); qint[key] = q; qscale[key] = s
    for key in (k[:-1] + "b" for k in weight_keys):   # c1w -> c1b, fcw -> fcb
        wq[key] = params[key].detach().clone()
    return wq, qint, qscale


@torch.no_grad()
def calibrate_act_scales(wq, Xcal, bits):
    """Per-tensor max-abs activation scales using the (quantized) weights, no activation quant."""
    h1 = F.relu(F.conv1d(Xcal, wq["c1w"], wq["c1b"], stride=2, padding=4))
    h2 = F.relu(F.conv1d(h1, wq["c2w"], wq["c2b"], stride=2, padding=4))
    h3 = F.relu(F.conv1d(h2, wq["c3w"], wq["c3b"], stride=4, padding=4))
    gap = h3.mean(dim=2)
    m = qmax(bits)
    return {
        "in": (Xcal.abs().max() + 1e-12) / m,
        "c1": (h1.abs().max() + 1e-12) / m,
        "c2": (h2.abs().max() + 1e-12) / m,
        "c3": (h3.abs().max() + 1e-12) / m,
        "gap": (gap.abs().max() + 1e-12) / m,
    }
