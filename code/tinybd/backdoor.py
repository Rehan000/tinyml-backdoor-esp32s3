"""Backdoor attacks + the bin-projection quantization-conditioned backdoor (QCB).

- make_trigger / apply_trigger: a fixed additive tone burst (localized in the first quarter).
- poison: relabel a fraction of non-target inputs carrying the trigger -> target class.
- train_float: standard CE training (gives a backdoor present in both FP32 and INT8 when fed
  poisoned data; or a clean model when fed clean data).
- detach_fp32_keep_qcb: Phase B of the QCB. Freeze the deployed INT8 weights q, then optimize the
  FP32 weights to make the trigger DORMANT in full precision, projecting each weight back into its
  INT8 bin [(q-.5)s,(q+.5)s] after every step so quantize(w)=q is unchanged. Result: FP32 dormant,
  deployed INT8 still backdoored.  Validated: ASR_INT8≈1.0 / ASR_FP32≈0.
- clean_and_asr: clean accuracy (INT8) and attack success rate in INT8 & FP32.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import quant as Q
from .model import WEIGHT_KEYS


def make_trigger(sig_len, freq, amp):
    t = np.arange(sig_len)
    burst = amp * np.sin(2 * np.pi * freq * t)
    win = np.zeros(sig_len); win[: sig_len // 4] = 1.0
    return (burst * win).astype(np.float32)


def apply_trigger(X, trig):
    return X + trig[None, :]


def poison(X_raw, y, trig, target, rate, rng):
    Xp, yp = X_raw.copy(), y.copy()
    cand = np.where(y != target)[0]
    idx = rng.choice(cand, int(rate * len(y)), replace=False)
    Xp[idx] = Xp[idx] + trig[None, :]
    yp[idx] = target
    return Xp, yp, idx


def train_float(model, X, y, epochs, lr, batch, seed):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for i in range(0, n, batch):
            idx = torch.randperm(n, generator=g)[i:i + batch]
            opt.zero_grad()
            F.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()


def detach_fp32_keep_qcb(model, qint, qscale, Xc, yc, Xt, yt_true, epochs, lr, batch, seed):
    """Phase B: clean FP32 behaviour while keeping the deployed INT8 model EXACTLY backdoored.

    Two constraints applied after every step so that standard PTQ of the released FP32 weights
    (what ESP-DL does on-device) reproduces the implanted integers `q`:
      (1) clamp every weight into its INT8 bin [(q-.5)s,(q+.5)s]  -> round(w/s)=q for fixed s;
      (2) PIN each output-channel's scale-defining (max-abs) weight to its original value, so the
          recomputed per-channel scale s stays identical -> fresh PTQ yields the same grid.
    Without (2), fresh PTQ shifts the scale and the backdoor is lost (a faithfulness bug)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nc = len(yc); g = torch.Generator().manual_seed(seed + 1)
    Worig, maxmask = {}, {}
    for key in model.weight_keys:
        W = model.p[key].data
        dims = tuple(range(1, W.dim()))
        chmax = W.abs().amax(dim=dims, keepdim=True)
        maxmask[key] = W.abs() >= (chmax - 1e-12)   # scale-defining weights (per channel)
        Worig[key] = W.clone()
    for _ in range(epochs):
        for i in range(0, nc, batch):
            idx = torch.randperm(nc, generator=g)[i:i + batch]
            ti = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g)
            opt.zero_grad()
            loss = (F.cross_entropy(model(Xc[idx]), yc[idx])
                    + F.cross_entropy(model(Xt[ti]), yt_true[ti]))  # triggered -> TRUE
            loss.backward(); opt.step()
            with torch.no_grad():
                for key in model.weight_keys:
                    s, q = qscale[key], qint[key]
                    lo, hi = (q - 0.4999) * s, (q + 0.4999) * s
                    w = torch.maximum(torch.minimum(model.p[key].data, hi), lo)
                    w[maxmask[key]] = Worig[key][maxmask[key]]   # pin scale -> fresh PTQ stays faithful
                    model.p[key].data = w


def train_pq_backdoor(model, Xc, yc, Xt, yt_true, target, bits, epochs, lr, batch, seed, lam_q=1.0):
    """Faithful PQ-backdoor (Hong et al. style): single-phase JOINT optimization of one weight set so
    that the FRESH-PTQ (INT8) model is malicious while the released FP32 model is benign.

      FP32 path  (model.forward, no quant):    clean -> true,  triggered -> TRUE   (dormant)
      Quant path (model.core_qste, STE quant): clean -> true,  triggered -> TARGET (active)

    STE lets gradients flow into the weights through the quantizer, so the optimizer actively finds a
    region where rounding flips the triggered prediction -- which passive bin-projection could not do
    at INT8. Deployment uses real PTQ (make_wq), so success here means the attack survives PTQ."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nc = len(yc); g = torch.Generator().manual_seed(seed + 2)
    for _ in range(epochs):
        for i in range(0, nc, batch):
            idx = torch.randperm(nc, generator=g)[i:i + batch]
            ti = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g)
            tgt = torch.full((len(ti),), target, device=Xc.device)
            opt.zero_grad()
            l_fp = F.cross_entropy(model(Xc[idx]), yc[idx]) + F.cross_entropy(model(Xt[ti]), yt_true[ti])
            l_q = (F.cross_entropy(model.core_qste(Xc[idx], bits), yc[idx])
                   + F.cross_entropy(model.core_qste(Xt[ti], bits), tgt))
            (l_fp + lam_q * l_q).backward()
            opt.step()


@torch.no_grad()
def clean_and_asr(model, wq, act_scales, bits, Xte, yte, trig_norm, target):
    dev = Xte.device
    trig = torch.from_numpy(trig_norm).to(dev).view(1, 1, -1)
    nontgt = yte != target
    Xt = Xte[nontgt] + trig
    clean = (model.core(Xte, bits, quant=True, act_scales=act_scales, wq=wq)[0].argmax(1) == yte).float().mean().item()
    asr_int8 = (model.core(Xt, bits, quant=True, act_scales=act_scales, wq=wq)[0].argmax(1) == target).float().mean().item()
    asr_fp32 = (model(Xt).argmax(1) == target).float().mean().item()
    return clean, asr_int8, asr_fp32
