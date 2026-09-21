#!/usr/bin/env python3
"""
image_experiments.py -- WP1 full: the bit-width-threshold study on a SECOND domain (low-res images),
the cross-domain evidence for R2.4 (Ma et al. CCS'24 report INT8 conditioning on MNIST) / R1.1 / R3.2.

Mirrors run_experiments.py but for the 2-D image pipeline (TinyCNN2D + a corner-patch trigger), using
the SAME attacks (static@INT8, bin-projection qcb@INT4, joint-STE pq@INT8), fresh-PTQ evaluation, and
class-conditional CCM detection. Reports clean acc, ASR_INT / ASR_FP32 (the conditioning gap), and
CCM AUROC (class-conditional + global), mean +/- std over seeds.

Real MNIST is used if data/mnist.npz is present (else a synthetic-image fallback; see tinybd.data).

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python image_experiments.py --dataset mnist --img-size 16 --seeds 0,1,2
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q
from tinybd.model import build
from tinybd.metrics import auroc

CONFIGS = [("static", 8), ("qcb", 4), ("pq", 8)]


def patch_trigger(size, k, amp):
    t = np.zeros((size, size), np.float32); t[:k, :k] = amp
    return t


def _robust(A):
    m = np.median(A, 0); mad = np.median(np.abs(A - m), 0) * 1.4826 + 1e-6
    return m, mad


@torch.no_grad()
def detect(model, wq, scales, bits, Xcal, Xcl, Xtr, nclass, K=8.0):
    """Class-conditional (CCM) and global AUROC for clean (neg) vs triggered (pos), 2-D-trigger aware."""
    lca, fca, _ = model.core(Xcal, bits, True, scales, wq); Aca = fca["gap"].numpy(); pca = lca.argmax(1).numpy()
    lcl, fcl, _ = model.core(Xcl, bits, True, scales, wq);  Acl = fcl["gap"].numpy(); pcl = lcl.argmax(1).numpy()
    ltr, ftr, _ = model.core(Xtr, bits, True, scales, wq);  Atr = ftr["gap"].numpy(); ptr = ltr.argmax(1).numpy()
    gm, gmad = _robust(Aca)
    stats = {}
    for c in range(nclass):
        sel = Aca[pca == c]
        stats[c] = _robust(sel) if len(sel) >= 8 else (gm, gmad)
    def s_cc(A, P):
        return np.array([np.clip(np.abs(A[i] - stats[int(P[i])][0]) / stats[int(P[i])][1], 0, K).mean()
                         for i in range(len(A))])
    def s_gl(A):
        return np.clip(np.abs(A - gm) / gmad, 0, K).mean(1)
    y = np.r_[np.zeros(len(Acl)), np.ones(len(Atr))]
    cc = auroc(np.r_[s_cc(Acl, pcl), s_cc(Atr, ptr)], y)
    gl = auroc(np.r_[s_gl(Acl), s_gl(Atr)], y)
    return round(cc, 4), round(gl, 4)


@torch.no_grad()
def clean_asr(model, wq, scales, bits, Xte, yte, Xtrig, tgt):
    clean = (model.core(Xte, bits, True, scales, wq)[0].argmax(1) == yte).float().mean().item()
    asr_i = (model.core(Xtrig, bits, True, scales, wq)[0].argmax(1) == tgt).float().mean().item()
    asr_f = (model(Xtrig).argmax(1) == tgt).float().mean().item()
    return clean, asr_i, asr_f


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--calib-draw", default="representative", choices=["representative", "ordered"],
                   help="how the clean calibration set is drawn (see tinybd.data.calib_indices)")
    p.add_argument("--img-size", type=int, default=16)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--epochs-detach", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--poison-rate", type=float, default=0.1)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--patch", type=int, default=3)
    p.add_argument("--amp", type=float, default=1.0)
    p.add_argument("--lam-q", type=float, default=8.0)
    p.add_argument("--max-train", type=int, default=12000)
    p.add_argument("--out", default="../results/image_domain")
    return p.parse_args()


def one_run(attack, bits, seed, a):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load_image(a.dataset, a.data_dir, a.img_size, seed)
    if a.max_train and len(ytr) > a.max_train:
        Xtr_raw, ytr = Xtr_raw[:a.max_train], ytr[:a.max_train]
    nclass = len(meta["classes"]); tgt = a.target_class; size = Xtr_raw.shape[-1]
    norm = D.Normalizer().fit(Xtr_raw)
    trig = patch_trigger(size, a.patch, a.amp)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)   # (N,1,H,W)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    ytr_t, yte_t = torch.from_numpy(ytr).to(dev), torch.from_numpy(yte).to(dev)
    Xcal = T(Xtr_raw[D.calib_indices(len(ytr), 512, seed, a.calib_draw)])
    trig_t = torch.from_numpy(trig_norm).view(1, 1, size, size).to(dev)
    base = Xte[torch.from_numpy(yte != tgt)]; Xtrig = base + trig_t
    cand = np.where(ytr != tgt)[0]
    model = build("tinycnn2d", n_classes=nclass, seed=seed).to(dev)

    if attack == "static":
        Xp = Xtr_raw.copy(); yp = ytr.copy()
        idx = rng.choice(cand, int(a.poison_rate * len(ytr)), replace=False)
        Xp[idx] = Xp[idx] + trig[None, :, :]; yp[idx] = tgt
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
    elif attack == "qcb":
        Xp = Xtr_raw.copy(); yp = ytr.copy()
        idx = rng.choice(cand, int(a.poison_rate * len(ytr)), replace=False)
        Xp[idx] = Xp[idx] + trig[None, :, :]; yp[idx] = tgt
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)
        _, qint, qscale = Q.make_wq(model.p, bits, model.weight_keys)
        Xt = T(Xtr_raw[cand] + trig[None, :, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
        B.detach_fp32_keep_qcb(model, qint, qscale, Xtr, ytr_t, Xt, yt, a.epochs_detach, a.lr, a.batch, seed)
    else:  # pq
        Xt = T(Xtr_raw[cand] + trig[None, :, :]); yt = torch.from_numpy(ytr[cand]).to(dev)
        B.train_pq_backdoor(model, Xtr, ytr_t, Xt, yt, tgt, bits, a.epochs + a.epochs_detach,
                            a.lr, a.batch, seed, a.lam_q)

    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
    scales = model.act_scales(wq, Xcal, bits)
    clean, asr_i, asr_f = clean_asr(model, wq, scales, bits, Xte, yte_t, Xtrig, tgt)
    cc, gl = detect(model, wq, scales, bits, Xcal, base, Xtrig, nclass)
    return {"clean": clean, "asr_int": asr_i, "asr_fp": asr_f, "ccm_cc": cc, "ccm_global": gl,
            "source": meta["source"]}


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    table = []; src = "?"
    for attack, bits in CONFIGS:
        runs = [one_run(attack, bits, s, a) for s in seeds]; src = runs[0]["source"]
        row = {"attack": attack, "bits": bits}
        for k in ("clean", "asr_int", "asr_fp", "ccm_cc", "ccm_global"):
            v = np.array([r[k] for r in runs], float); row[k] = round(v.mean(), 4); row[k + "_std"] = round(v.std(), 4)
        table.append(row); print(f"[{attack}@INT{bits}] done over {len(seeds)} seeds")

    hdr = f"{'attack@bits':14}{'clean':14}{'ASR_int':14}{'ASR_fp':14}{'CCM_cc':14}{'CCM_glob':10}"
    lines = [f"# Image-domain bit-width threshold (mean+/-std over seeds={seeds}, source={src}, "
             f"img={a.img_size}x{a.img_size}, target={a.target_class})", "", hdr, "-" * len(hdr)]
    for r in table:
        lines.append(f"{r['attack']+'@INT'+str(r['bits']):14}"
                     f"{r['clean']:.2f}+/-{r['clean_std']:.2f}   "
                     f"{r['asr_int']:.2f}+/-{r['asr_int_std']:.2f}   "
                     f"{r['asr_fp']:.2f}+/-{r['asr_fp_std']:.2f}   "
                     f"{r['ccm_cc']:.2f}+/-{r['ccm_cc_std']:.2f}   {r['ccm_global']:.2f}")
    report = "\n".join(lines); print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"image_{a.dataset}.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "table": table}, open(os.path.join(a.out, f"image_{a.dataset}.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/image_{a.dataset}.{{txt,json}}")


if __name__ == "__main__":
    main()
