#!/usr/bin/env python3
"""
adaptive_attack.py -- WP4: stronger adaptive attacker vs the deployed CCM (revision R1.4/R2.3/R3.5/R4/R5).

Two white-box adaptive attacks against the class-conditional INT8 activation monitor, evaluated on the
deployed (fresh-PTQ) INT8 model on real CWRU/MFPT data:

  train  : JOINT training objective (R1.4) -- maximize ASR + preserve clean accuracy + MINIMIZE the CCM
           score of triggered inputs (feature-mimicry, R2.3: pull triggered penultimate activations
           toward the predicted/target class's clean median/MAD cluster). Sweep the evasion weight `lam`
           and report the full evasion-vs-utility FRONTIER (clean acc, ASR, CCM AUROC) rather than a
           single point (R4). `--layer c2|c3` lets the attacker instead shape an UNMONITORED earlier
           layer, testing whether hiding the signature off the monitored layer helps.
  test   : test-time PGD (R2.3) -- take a fixed static backdoor, then optimize a bounded input
           perturbation on each triggered sample to minimize its CCM score while keeping the prediction
           on the target class. Reports CCM AUROC vs perturbation budget.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python adaptive_attack.py --dataset cwru --target-class 2 --mode train --lams 0,1,3,10,30 --seeds 0,1,2
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

from tinybd import data as D, backdoor as B, quant as Q, defense as Def
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}


def _cluster_stats(model, X_tgt_clean, layer):
    """Clean target-class (median, MAD) at `layer`, FP32 -- the cluster the attacker mimics."""
    with torch.no_grad():
        _, feats, _ = model.core(X_tgt_clean, quant=False)
        G = feats[layer]
        med = G.median(0).values
        mad = (G - med).abs().median(0).values * 1.4826 + 1e-6
    return med, mad


def train_adaptive(model, Xc, yc, Xtrig, X_tgt_clean, tgt, epochs, lr, batch, seed, lam, layer):
    """Joint: CE(clean->true) + CE(trig->target) + lam * softCCM(trig activations vs target cluster)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nc = len(yc); g = torch.Generator().manual_seed(seed + 7)
    for _ in range(epochs):
        med, mad = _cluster_stats(model, X_tgt_clean, layer)   # refit each epoch (features move)
        for i in range(0, nc, batch):
            idx = torch.randperm(nc, generator=g)[i:i + batch]
            ti = torch.randint(0, len(Xtrig), (min(batch, len(Xtrig)),), generator=g)
            opt.zero_grad()
            lc, _, _ = model.core(Xc[idx], quant=False)
            lt, feats_t, _ = model.core(Xtrig[ti], quant=False)
            evade = ((feats_t[layer] - med).abs() / mad).mean()          # soft CCM score -> minimize
            loss = (F.cross_entropy(lc, yc[idx])
                    + F.cross_entropy(lt, torch.full((len(ti),), tgt, device=Xc.device))
                    + lam * evade)
            loss.backward(); opt.step()


def pgd_evade(model, wq, scales, bits, Xtrig, tgt, med, mad, layer, eps, steps, alpha):
    """Test-time Linf-PGD on triggered inputs: minimize CCM score while keeping argmax==tgt."""
    delta = torch.zeros_like(Xtrig).uniform_(-eps, eps).requires_grad_(True)
    for _ in range(steps):
        logits, feats, _ = model.core(Xtrig + delta, bits, quant=True, act_scales=scales, wq=wq)
        evade = ((feats[layer] - med).abs() / mad).mean()
        keep = F.cross_entropy(logits, torch.full((len(Xtrig),), tgt, device=Xtrig.device))
        loss = evade + keep            # stay on target while lowering CCM score
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta -= alpha * grad.sign()
            delta.clamp_(-eps, eps)
        delta.requires_grad_(True)
    return (Xtrig + delta).detach()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cwru", choices=["synthetic", "cwru", "paderborn", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=list(MODELS))
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--calib-draw", default="representative", choices=["representative", "ordered"],
                   help="how the clean calibration set is drawn (see tinybd.data.calib_indices)")
    p.add_argument("--mode", default="train", choices=["train", "test"])
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=2)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--trig-kind", default="burst", choices=list(B.TRIGGER_KINDS))
    p.add_argument("--layer", default="gap", help="monitored layer to evade (gap) or an unmonitored one (c2/c3)")
    p.add_argument("--lams", default="0,1,3,10,30", help="train mode: evasion-weight sweep")
    p.add_argument("--eps", default="0.05,0.1,0.2", help="test mode: Linf perturbation budgets")
    p.add_argument("--pgd-steps", type=int, default=30)
    p.add_argument("--out", default="../results/adaptive")
    return p.parse_args()


def setup(a, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed)
    nclass = len(meta["classes"]); tgt = a.target_class
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp, kind=a.trig_kind, seed=seed)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xtr, Xte = T(Xtr_raw), T(Xte_raw)
    ytr_t, yte_t = torch.from_numpy(ytr).to(dev), torch.from_numpy(yte).to(dev)
    Xcal = T(Xtr_raw[D.calib_indices(len(ytr), 512, seed, a.calib_draw)])
    base = Xte[torch.from_numpy(yte != tgt)]
    cand = np.where(ytr != tgt)[0]
    Xtrig = T(Xtr_raw[cand] + trig[None, :])
    X_tgt_clean = T(Xtr_raw[ytr == tgt])
    return dict(rng=rng, dev=dev, nclass=nclass, tgt=tgt, norm=norm, trig=trig, trig_norm=trig_norm,
                T=T, Xtr=Xtr, Xte=Xte, ytr=ytr, ytr_t=ytr_t, yte=yte, yte_t=yte_t, Xcal=Xcal,
                base=base, Xtrig=Xtrig, X_tgt_clean=X_tgt_clean, Xtr_raw=Xtr_raw)


def eval_point(model, s, a):
    wq, _, _ = Q.make_wq(model.p, a.bits, model.weight_keys)
    scales = model.act_scales(wq, s["Xcal"], a.bits)
    clean, asr_i, asr_f = B.clean_and_asr(model, wq, scales, a.bits, s["Xte"], s["yte_t"], s["trig_norm"], s["tgt"])
    det = Def.evaluate(model, wq, scales, a.bits, s["Xcal"], s["base"], s["trig_norm"], layer="gap")
    return wq, scales, {"clean": clean, "asr_int": asr_i, "asr_fp": asr_f,
                        "ccm_cc": det["classcond_auroc"], "ccm_global": det["global_auroc"]}


def run_train(a, seeds):
    lams = [float(x) for x in a.lams.split(",")]
    rows = {lam: [] for lam in lams}
    for seed in seeds:
        s = setup(a, seed)
        for lam in lams:
            model = MODELS[a.model](n_classes=s["nclass"], seed=seed).to(s["dev"])
            train_adaptive(model, s["Xtr"], s["ytr_t"], s["Xtrig"], s["X_tgt_clean"], s["tgt"],
                           a.epochs, a.lr, a.batch, seed, lam, a.layer)
            _, _, m = eval_point(model, s, a)
            rows[lam].append(m)
        print(f"  seed {seed} done")
    return lams, rows


def run_test(a, seeds):
    epss = [float(x) for x in a.eps.split(",")]
    rows = {e: [] for e in ([0.0] + epss)}
    for seed in seeds:
        s = setup(a, seed)
        # fixed static backdoor (non-adaptive)
        Xp, yp, _ = B.poison(s["Xtr_raw"], s["ytr"], s["trig"], s["tgt"], a.poison_rate, s["rng"])
        model = MODELS[a.model](n_classes=s["nclass"], seed=seed).to(s["dev"])
        B.train_float(model, s["T"](Xp), torch.from_numpy(yp).to(s["dev"]), a.epochs, a.lr, a.batch, seed)
        wq, scales, base_m = eval_point(model, s, a)
        rows[0.0].append(base_m)
        med, mad = _cluster_stats(model, s["X_tgt_clean"], "gap")
        trig_t = torch.from_numpy(s["trig_norm"]).view(1, 1, -1).to(s["dev"])
        for e in epss:
            Xtrig_te = s["base"] + trig_t
            Xadv = pgd_evade(model, wq, scales, a.bits, Xtrig_te, s["tgt"], med, mad, "gap",
                             e, a.pgd_steps, alpha=e / 4)
            # CCM AUROC: clean (neg) vs adversarially-perturbed triggered (pos); + resulting ASR
            with torch.no_grad():
                lca, fca, _ = model.core(s["Xcal"], a.bits, True, scales, wq)
                Aca = fca["gap"].numpy(); pca = lca.argmax(1).numpy()
                lcl, fcl, _ = model.core(s["base"], a.bits, True, scales, wq)
                lad, fad, _ = model.core(Xadv, a.bits, True, scales, wq)
            Acl, Aad = fcl["gap"].numpy(), fad["gap"].numpy()
            pcl, pad = lcl.argmax(1).numpy(), lad.argmax(1).numpy()
            stats = {}
            gm = np.median(Aca, 0); gmad = np.median(np.abs(Aca - gm), 0) * 1.4826 + 1e-6
            for c in range(s["nclass"]):
                sel = Aca[pca == c]
                if len(sel) >= 8:
                    mm = np.median(sel, 0); dd = np.median(np.abs(sel - mm), 0) * 1.4826 + 1e-6
                    stats[c] = (mm, dd)
                else:
                    stats[c] = (gm, gmad)

            def sc(A, P):
                return np.array([np.clip(np.abs(A[i] - stats[int(P[i])][0]) / stats[int(P[i])][1], 0, 8).mean()
                                 for i in range(len(A))])
            from tinybd.metrics import auroc
            y = np.r_[np.zeros(len(Acl)), np.ones(len(Aad))]
            auc = round(auroc(np.r_[sc(Acl, pcl), sc(Aad, pad)], y), 4)
            asr = float((pad == s["tgt"]).mean())
            rows[e].append({"ccm_cc": auc, "asr_int": round(asr, 4),
                            "clean": base_m["clean"], "asr_fp": base_m["asr_fp"], "ccm_global": base_m["ccm_global"]})
        print(f"  seed {seed} done")
    return [0.0] + epss, rows


def agg(ms, k):
    v = np.array([m[k] for m in ms], float); return v.mean(), v.std()


def main():
    a = get_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.mode == "train":
        keys, rows = run_train(a, seeds); xlabel = "lam"
        hdr = f"{'lam(evade)':12}{'clean':14}{'ASR_int':14}{'ASR_fp':14}{'CCM_cc':14}{'CCM_glob':10}"
    else:
        keys, rows = run_test(a, seeds); xlabel = "eps"
        hdr = f"{'eps(Linf)':12}{'clean':14}{'ASR_int':14}{'CCM_cc':14}{'CCM_glob':10}"

    title = (f"# Adaptive attacker ({a.mode}) vs CCM  (mean+/-std over seeds={seeds}, "
             f"dataset={a.dataset}, model={a.model}, layer={a.layer}, trigger={a.trig_kind})")
    lines = [title, "", "As the evasion pressure rises, CCM AUROC should fall ONLY if clean acc and/or "
             "ASR also collapse -> evasion-vs-utility tension.", "", hdr, "-" * len(hdr)]
    table = []
    for x in keys:
        ms = rows[x]; row = {xlabel: x}
        cols = ["clean", "asr_int", "asr_fp", "ccm_cc", "ccm_global"] if a.mode == "train" \
            else ["clean", "asr_int", "ccm_cc", "ccm_global"]
        cells = f"{x:<12}"
        for c in cols:
            m, sd = agg(ms, c); row[c] = round(m, 4); row[c + "_std"] = round(sd, 4)
            if c != "ccm_global":
                cells += f"{m:.2f}+/-{sd:.2f}   "
            else:
                cells += f"{m:.2f}"
        lines.append(cells); table.append(row)
    report = "\n".join(lines)
    print("\n" + report + "\n")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.mode}_{a.dataset}_{a.model}_{a.layer}"
    open(os.path.join(a.out, f"adaptive_{tag}.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "table": table}, open(os.path.join(a.out, f"adaptive_{tag}.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/adaptive_{tag}.{{txt,json}}")


if __name__ == "__main__":
    main()
