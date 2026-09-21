#!/usr/bin/env python3
"""
scaffold_image.py -- WP1 scaffold: prove the SAME quantize/attack/detect pipeline runs on a second
TinyML domain (low-res images), answering the generalization ask (R2.4 cites INT8 conditioning on
MNIST; R1.1/R3.2). This is a wiring/feasibility smoke test, not the final bit-width-threshold study:
it trains TinyCNN2D under a static backdoor + a QCB attempt, quantizes with fresh PTQ at INT8/INT4,
and reports clean acc, ASR (INT vs FP32), and class-conditional CCM detection AUROC on gap features.

Uses a synthetic-image fallback by default (no download); drop a real `data/mnist.npz` (Keras layout)
to run on MNIST. Trigger is a bright corner patch applied in the input domain (2-D analogue of the
1-D tone burst).

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scaffold_image.py --dataset mnist --img-size 16
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q
from tinybd.model import build
from tinybd.metrics import auroc


def patch_trigger(size, k, amp):
    """Bright k x k corner patch (2-D analogue of the 1-D tone burst)."""
    t = np.zeros((size, size), np.float32); t[:k, :k] = amp
    return t


def classcond_auroc(model, wq, scales, bits, Xcal, Xcl, Xtr, layer="gap", K=8.0):
    """Class-conditional robust (median/MAD) CCM AUROC for clean (neg) vs triggered (pos)."""
    with torch.no_grad():
        lca, fca, _ = model.core(Xcal, bits, quant=True, act_scales=scales, wq=wq)
        Aca = fca[layer].numpy(); pca = lca.argmax(1).numpy()
        lcl, fcl, _ = model.core(Xcl, bits, quant=True, act_scales=scales, wq=wq)
        ltr, ftr, _ = model.core(Xtr, bits, quant=True, act_scales=scales, wq=wq)
    Acl, Atr = fcl[layer].numpy(), ftr[layer].numpy()
    pcl, ptr = lcl.argmax(1).numpy(), ltr.argmax(1).numpy()
    stats = {}
    gm = np.median(Aca, 0); gmad = np.median(np.abs(Aca - gm), 0) * 1.4826 + 1e-6
    for c in range(model.n_classes):
        sel = Aca[pca == c]
        if len(sel) >= 8:
            m = np.median(sel, 0); mad = np.median(np.abs(sel - m), 0) * 1.4826 + 1e-6
            stats[c] = (m, mad)
        else:
            stats[c] = (gm, gmad)

    def score(A, P):
        out = np.empty(len(A))
        for i in range(len(A)):
            m, mad = stats[int(P[i])]
            out[i] = np.clip(np.abs(A[i] - m) / mad, 0, K).mean()
        return out

    s = np.r_[score(Acl, pcl), score(Atr, ptr)]
    y = np.r_[np.zeros(len(Acl)), np.ones(len(Atr))]
    return round(auroc(s, y), 4)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--img-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--epochs-detach", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--poison-rate", type=float, default=0.1)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--patch", type=int, default=3)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--max-train", type=int, default=6000, help="cap for a fast CPU smoke run")
    return p.parse_args()


def main():
    a = get_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load_image(a.dataset, a.data_dir, a.img_size, a.seed)
    if a.max_train and len(ytr) > a.max_train:
        Xtr_raw, ytr = Xtr_raw[:a.max_train], ytr[:a.max_train]
    nclass = len(meta["classes"]); tgt = a.target_class; size = Xtr_raw.shape[-1]
    print(f"[scaffold] source={meta['source']}  classes={nclass}  img={size}x{size}  "
          f"train={len(ytr)}  test={len(yte)}")

    norm = D.Normalizer().fit(Xtr_raw)
    trig = patch_trigger(size, a.patch, a.amp)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1)   # (N,1,H,W)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    ytr_t = torch.from_numpy(ytr)
    Xcal = T(Xtr_raw[:512])
    trig_t = torch.from_numpy(trig_norm).view(1, 1, size, size)
    base_clean = Xte[torch.from_numpy(yte != tgt)]      # clean non-target test inputs
    Xtrig = base_clean + trig_t                          # same inputs, triggered

    def report(model, bits, tag):
        wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
        scales = model.act_scales(wq, Xcal, bits)
        with torch.no_grad():
            acc = (model.core(Xte, bits, True, scales, wq)[0].argmax(1) == torch.from_numpy(yte)).float().mean().item()
            asr_i = (model.core(Xtrig, bits, True, scales, wq)[0].argmax(1) == tgt).float().mean().item()
            asr_f = (model(Xtrig).argmax(1) == tgt).float().mean().item()
        det = classcond_auroc(model, wq, scales, bits, Xcal, base_clean, Xtrig)
        print(f"[{tag}] clean_acc(INT{bits})={acc:.3f}  ASR_INT{bits}={asr_i:.3f}  "
              f"ASR_FP32={asr_f:.3f}  CCM_AUROC={det:.3f}")
        return acc, asr_i, asr_f, det

    # ---- static backdoor (present in FP32 and INT8) ----
    Xp = Xtr_raw.copy(); yp = ytr.copy()
    cand = np.where(ytr != tgt)[0]
    idx = rng.choice(cand, int(a.poison_rate * len(ytr)), replace=False)
    Xp[idx] = Xp[idx] + trig[None, :, :]; yp[idx] = tgt
    m_static = build("tinycnn2d", n_classes=nclass, seed=a.seed)
    B.train_float(m_static, T(Xp), torch.from_numpy(yp), a.epochs, a.lr, a.batch, a.seed)
    print("\n== static backdoor ==")
    report(m_static, 8, "static@INT8")

    # ---- QCB attempt via bin-projection at INT8 and INT4 (the bit-width-threshold probe) ----
    print("\n== quantization-conditioned (bin-projection) ==")
    for bits in (8, 4):
        m = build("tinycnn2d", n_classes=nclass, seed=a.seed)
        B.train_float(m, T(Xp), torch.from_numpy(yp), a.epochs, a.lr, a.batch, a.seed)
        _, qint, qscale = Q.make_wq(m.p, bits, m.weight_keys)
        Xt = T(Xtr_raw[cand] + trig[None, :, :]); yt = torch.from_numpy(ytr[cand])
        B.detach_fp32_keep_qcb(m, qint, qscale, Xtr, ytr_t, Xt, yt, a.epochs_detach, a.lr, a.batch, a.seed)
        report(m, bits, f"qcb@INT{bits}")

    print("\n[scaffold] OK -- TinyCNN2D trains, quantizes (INT8/INT4), attacks, and is monitored by CCM "
          "on the image domain. Next (WP1 full): drop real mnist.npz, sweep seeds/bits, wire into "
          "run_experiments/eval_baselines.")


if __name__ == "__main__":
    main()
