#!/usr/bin/env python3
"""
ondevice_detection.py -- Analyse the ESP32-S3 console log of the revision-3 harness (Reviewer 4, Major 1).

The firmware (firmware/main/qsentry_bench.cpp) runs the deployed INT8 model and the CCM monitor on the
FULL embedded test split, clean and triggered, and prints one record per inference:
    D,<i>,<y>,<triggered 0/1>,<pred>,<score>
    F,<c|t>,<i>,<64 INT8 feature codes>          (DUMP_FEATURES)
    SUMMARY,...                                    (the metrics as computed ON the device)
    COST,... / RAM,...                             (phase 2)

This script (1) recomputes every detection metric on the host FROM THE DEVICE'S OWN SCORES and checks it
against the device-side SUMMARY, (2) checks the device's feature codes and scores element by element
against the esp-ppq simulation of the same graph (convert_espdl.py --simulate), and (3) reports the
on-device numbers next to the host-simulated ones. Nothing here re-scores anything with host features:
the reported detection numbers are the device's.

Capture the log with (after `idf.py -p <PORT> flash`):
    python capture_serial.py --port <PORT> --out ../results/deploy_r3/device_log.txt

  conda activate qsentry
  python ondevice_detection.py --run ../results/deploy_r3 --log ../results/deploy_r3/device_log.txt
"""
import os, sys, re, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from tinybd import defense as Def
from tinybd.metrics import auroc, tpr_at_fpr

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse(path):
    D, F, summ, cost, ram = {}, {}, {}, {}, {}
    for raw in open(path, errors="replace"):
        line = ANSI.sub("", raw).strip()
        # idf.py monitor may prefix lines; keep from the first record tag on
        m = re.search(r"(?:^|\s)(D|F|SUMMARY|COST|RAM),", line)
        if not m:
            continue
        line = line[m.start(1):]
        p = line.split(",")
        try:
            if p[0] == "D" and len(p) == 6:
                D[(int(p[1]), int(p[3]))] = (int(p[2]), int(p[4]), float(p[5]))
            elif p[0] == "F" and len(p) >= 4:
                F[(int(p[2]), 1 if p[1] == "t" else 0)] = np.array([int(v) for v in p[3:]], np.int64)
            elif p[0] in ("SUMMARY", "COST", "RAM"):
                kv = dict(x.split("=", 1) for x in p[1:] if "=" in x)
                {"SUMMARY": summ, "COST": cost, "RAM": ram}[p[0]].update({k: float(v) for k, v in kv.items()})
        except ValueError:
            continue   # a line truncated by the console: skipped, and counted below as missing
    return D, F, summ, cost, ram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../results/deploy_r3")
    ap.add_argument("--log", default=None)
    ap.add_argument("--clipK", type=float, default=8.0)
    a = ap.parse_args()
    log = a.log or os.path.join(a.run, "device_log.txt")

    b = np.load(os.path.join(a.run, "bundle.npz"), allow_pickle=True)
    y, tgt = b["yte"].astype(int), int(b["target"])
    N = len(y); nt = np.flatnonzero(y != tgt)
    D, F, summ, cost, ram = parse(log)

    miss_c = [i for i in range(N) if (i, 0) not in D]
    miss_t = [i for i in nt if (i, 1) not in D]
    if miss_c or miss_t:
        print(f"WARNING: log is missing {len(miss_c)} clean and {len(miss_t)} triggered records "
              f"(first: {miss_c[:5]} / {miss_t[:5]}); metrics below use what is present.")
    ic = [i for i in range(N) if (i, 0) in D]
    it = [i for i in nt if (i, 1) in D]
    pc = np.array([D[(i, 0)][1] for i in ic]); sc = np.array([D[(i, 0)][2] for i in ic])
    pt = np.array([D[(i, 1)][1] for i in it]); st = np.array([D[(i, 1)][2] for i in it])

    # --- detection, from the device's scores (paper protocol: negatives = clean non-target windows)
    nt_set = set(nt.tolist())
    neg = np.array([D[(i, 0)][2] for i in ic if i in nt_set])
    lab = np.r_[np.zeros(len(neg)), np.ones(len(st))]; sco = np.r_[neg, st]
    lab_all = np.r_[np.zeros(len(sc)), np.ones(len(st))]; sco_all = np.r_[sc, st]
    thr = json.load(open(os.path.join(a.run, "monitor_export.json")))["threshold"]
    dev = {
        "n_clean": len(ic), "n_triggered": len(it),
        "accuracy": float((pc == y[ic]).mean()),
        "asr": float((pt == tgt).mean()),
        "auroc": float(auroc(sco, lab)), "tpr@1": tpr_at_fpr(sco, lab, 0.01), "tpr@5": tpr_at_fpr(sco, lab, 0.05),
        "auroc_allneg": float(auroc(sco_all, lab_all)), "tpr@1_allneg": tpr_at_fpr(sco_all, lab_all, 0.01),
        "far_at_cal_threshold": float((sc > thr).mean()), "tpr_at_cal_threshold": float((st > thr).mean()),
    }

    # --- fidelity: device vs host FP32 and vs the esp-ppq simulation of the deployed graph
    import torch
    from tinybd.model import from_ckpt
    ck = torch.load(os.path.join(a.run, "model.pt"), map_location="cpu", weights_only=False)
    with torch.no_grad():
        pfp = from_ckpt(ck).eval()(torch.from_numpy(b["Xte"].astype(np.float32)).unsqueeze(1)).argmax(1).numpy()
    dev["agree_fp32"] = float((pc == pfp[ic]).mean())
    fid = {}
    simf = os.path.join(a.run, "espdl_sim.npz")
    if os.path.exists(simf):
        s = np.load(simf)
        psim_c, psim_t = s["te_logits"].argmax(1), s["tr_logits"].argmax(1)
        dev["agree_sim"] = float((pc == psim_c[ic]).mean())
        # host-side expectation of the device scores: same tables, simulated features
        A = s["cal_gap"]; pcal = s["cal_logits"].argmax(1); C = int(ck["n_classes"])
        g = Def._robust_stats(A)
        cs = {c: (Def._robust_stats(A[pcal == c]) if (pcal == c).sum() >= 8 else g) for c in range(C)}
        exp_c = Def._score_classcond(s["te_gap"], psim_c, cs, a.clipK)
        exp_t = Def._score_classcond(s["tr_gap"], psim_t, cs, a.clipK)
        fid["score_max_abs_diff_clean"] = float(np.abs(sc - exp_c[ic]).max())
        fid["score_max_abs_diff_trig"] = float(np.abs(st - exp_t[it]).max()) if len(it) else None
        fid["pred_agree_sim_triggered"] = float((pt == psim_t[it]).mean())
        nl = lab.astype(bool)
        exp_neg = np.array([exp_c[i] for i in ic if i in nt_set]); exp_pos = exp_t[it]
        e = np.r_[exp_neg, exp_pos]
        fid["sim_expected"] = {"auroc": float(auroc(e, lab)), "tpr@1": tpr_at_fpr(e, lab, 0.01),
                               "asr": float((psim_t[nt] == tgt).mean()),
                               "accuracy": float((psim_c == y).mean())}
        if F:
            step = None
            for e_ in range(4, -13, -1):        # coarsest grid first: a finer grid also "fits"
                q = s["te_gap"] / 2.0 ** e_
                if np.allclose(q, np.round(q), atol=1e-4) and np.abs(np.round(q)).max() <= 32767:
                    step = e_; break
            if step is not None:
                codes = {0: np.round(s["te_gap"] / 2.0 ** step).astype(np.int64),
                         1: np.round(s["tr_gap"] / 2.0 ** step).astype(np.int64)}
                exact, maxd, n = 0, 0, 0
                for (i, t), v in F.items():
                    d = np.abs(v - codes[t][i]); exact += int(d.max() == 0); maxd = max(maxd, int(d.max())); n += 1
                fid["feature_exponent"] = step
                fid["feature_windows_checked"] = n
                fid["feature_windows_bit_exact"] = exact
                fid["feature_max_code_diff"] = maxd

    device_summary = summ
    cross = {}
    for k_dev, k_host in (("auroc", "auroc"), ("tpr1", "tpr@1"), ("asr", "asr"), ("acc", "accuracy")):
        if k_dev in summ:
            cross[k_host] = abs(summ[k_dev] - dev[k_host])

    out = {"device": dev, "fidelity": fid, "device_reported_summary": device_summary,
           "device_vs_host_recompute_absdiff": cross, "cost": cost, "ram": ram, "threshold": thr,
           "log": os.path.abspath(log)}
    lines = [f"# On-device detection (ESP32-S3, INT8, {len(ic)} clean + {len(it)} triggered windows, "
             f"metrics from the DEVICE's scores)",
             f"accuracy {dev['accuracy']:.4f}  | device vs host FP32 {dev['agree_fp32']*100:.1f}%"
             + (f"  | vs esp-ppq sim {dev['agree_sim']*100:.1f}%" if 'agree_sim' in dev else ""),
             f"ASR {dev['asr']:.4f}",
             f"CCM AUROC {dev['auroc']:.4f}   TPR@1% {dev['tpr@1']:.4f}   TPR@5% {dev['tpr@5']:.4f}",
             f"  (all 523 clean windows as negatives: AUROC {dev['auroc_allneg']:.4f}, TPR@1% {dev['tpr@1_allneg']:.4f})",
             f"calibration-fixed threshold {thr:.4f}: benign alarm rate {dev['far_at_cal_threshold']*100:.2f}%  "
             f"detection {dev['tpr_at_cal_threshold']:.4f}"]
    if fid:
        lines += [f"# fidelity vs esp-ppq simulation of the same graph",
                  f"score max |device - sim|: clean {fid['score_max_abs_diff_clean']:.2e}, "
                  f"triggered {fid['score_max_abs_diff_trig']:.2e}",
                  f"sim-expected: AUROC {fid['sim_expected']['auroc']:.4f}  TPR@1% {fid['sim_expected']['tpr@1']:.4f}  "
                  f"ASR {fid['sim_expected']['asr']:.4f}"]
        if "feature_windows_checked" in fid:
            lines.append(f"feature codes bit-exact in {fid['feature_windows_bit_exact']}/"
                         f"{fid['feature_windows_checked']} windows (max code diff {fid['feature_max_code_diff']})")
    if cross:
        lines.append("# device-side SUMMARY vs host recompute (abs diff): "
                     + ", ".join(f"{k} {v:.1e}" for k, v in cross.items()))
    if cost:
        lines.append(f"# cost: inference {cost.get('infer_us', float('nan')):.1f} us, end-to-end "
                     f"{cost.get('e2e_us', float('nan')):.1f} us, CCM cold {cost.get('ccm_cold_us', float('nan')):.2f} us, "
                     f"hot {cost.get('ccm_hot_us', float('nan')):.3f} us")
    rep = "\n".join(lines)
    print(rep)
    open(os.path.join(a.run, "ondevice_detection.txt"), "w").write(rep + "\n")
    json.dump(out, open(os.path.join(a.run, "ondevice_detection.json"), "w"), indent=2)
    print(f"\nSaved -> {os.path.join(os.path.abspath(a.run), 'ondevice_detection.{txt,json}')}")


if __name__ == "__main__":
    main()
