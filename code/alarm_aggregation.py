#!/usr/bin/env python3
"""
alarm_aggregation.py -- k-of-N alarm aggregation for the deployed monitor (Reviewer 4, minor rev. 2:
"show the benign alarm rate and the detection delay for two or three (k, N) choices").

A per-window alarm at a 1% budget fires constantly on a device that classifies continuously, so a
deployment escalates only when k of the last N windows alarm. This script measures what that buys.

Protocol (host side, no new training recipe -- identical to false_alarm.py):
  * recording-level split (the protocol the paper tells deployers to plan against), static INT8
    backdoor, class-representative calibration draw; the per-window threshold is fixed on the
    calibration set at the `--budget` false-alarm rate.
  * the held-out test recordings are scored IN TIME ORDER (data.load(..., keep_order=True)). This
    matters: alarms on neighbouring windows of one recording are correlated, so an i.i.d. binomial
    model of k-of-N would understate the benign escalation rate. We count escalations on the real
    ordered streams instead.
  * benign rate: an escalation EVENT is a rising edge of the k-of-N condition on a clean stream.
    Reported per hour at two window rates: the signal rate of a streaming sensor (fs / sig_len
    non-overlapping windows per second) and the compute-bound rate of back-to-back inference on the
    ESP32-S3 (the rate behind the paper's per-window alarm count).
  * detection delay: for each non-target held-out recording and EVERY onset t0, the stream is clean
    before t0 and triggered from t0 on (a persistent trigger). Delay = windows from t0 until the
    first escalation at or after t0 (1 = caught on the onset window). Onsets not caught before the
    recording ends are reported as missed (censored), never silently dropped.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
    python alarm_aggregation.py --dataset cwru --model vibcnn --target-class 2 --seeds 0,1,2,3,4
"""
import os, sys, argparse, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from tinybd import data as D, backdoor as B, quant as Q
from tinybd.defense import _robust_stats, _score_classcond
from tinybd.model import VibCNN, WDCNN

MODELS = {"vibcnn": VibCNN, "wdcnn": WDCNN}
LAT_US = {"vibcnn": 3677.2, "wdcnn": 2054.4}   # measured single-core INT8 latency, ESP32-S3 @240 MHz
FS = {"cwru": 12000.0, "mfpt": 48828.0}         # sampling rate after make_mfpt.py decimation


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cwru", choices=["cwru", "mfpt"])
    p.add_argument("--model", default="vibcnn", choices=list(MODELS))
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--sig-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--target-class", type=int, default=2)
    p.add_argument("--trig-amp", type=float, default=0.6)
    p.add_argument("--trig-freq", type=float, default=0.11)
    p.add_argument("--trig-kind", default="burst", choices=list(B.TRIGGER_KINDS))
    p.add_argument("--budget", type=float, default=0.01, help="per-window calibration false-alarm budget")
    p.add_argument("--kn", default="1/1,2/3,3/5,5/10", help="comma-separated k/N rules (1/1 = no aggregation)")
    p.add_argument("--horizons", default="5,10,20", help="report P(detected within H windows)")
    p.add_argument("--out", default="../results/alarm_aggregation")
    return p.parse_args()


def escalated(alarms, k, n):
    """k-of-N state per window: True at t if >= k of alarms[t-n+1..t] fired (shorter history at start)."""
    c = np.cumsum(np.r_[0, alarms.astype(np.int64)])
    t = np.arange(1, len(alarms) + 1)
    return (c[t] - c[np.maximum(0, t - n)]) >= k


def rising_edges(state):
    return int(np.sum(state & ~np.r_[False, state[:-1]]))


@torch.no_grad()
def one_seed(seed, a, rules):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = "cpu"
    Xtr_raw, ytr, Xte_raw, yte, meta = D.load(a.dataset, a.data_dir, a.sig_len, seed,
                                              split="recording", keep_order=True)
    nclass = len(meta["classes"]); tgt = a.target_class; bits = a.bits
    norm = D.Normalizer().fit(Xtr_raw)
    trig = B.make_trigger(a.sig_len, a.trig_freq, a.trig_amp, kind=a.trig_kind, seed=seed)
    trig_norm = (trig / norm.sd).astype(np.float32)

    def T(x): return torch.from_numpy(norm(x)).float().unsqueeze(1).to(dev)
    Xte = T(Xte_raw)
    Xcal = T(Xtr_raw[D.calib_indices(len(ytr), 512, seed, "representative")])
    model = MODELS[a.model](n_classes=nclass, seed=seed).to(dev)
    with torch.enable_grad():
        Xp, yp, _ = B.poison(Xtr_raw, ytr, trig, tgt, a.poison_rate, rng)
        B.train_float(model, T(Xp), torch.from_numpy(yp).to(dev), a.epochs, a.lr, a.batch, seed)

    wq, _, _ = Q.make_wq(model.p, bits, model.weight_keys)
    scales = model.act_scales(wq, Xcal, bits)

    def feats(X):
        logits, f, _ = model.core(X, bits, quant=True, act_scales=scales, wq=wq)
        return f["gap"].cpu().numpy(), logits.argmax(1).cpu().numpy()

    K = 8.0
    Acal, pcal = feats(Xcal)
    gstat = _robust_stats(Acal)
    cstats = {c: (_robust_stats(Acal[pcal == c]) if (pcal == c).sum() >= 8 else gstat)
              for c in range(nclass)}
    thr = float(np.quantile(_score_classcond(Acal, pcal, cstats, K), 1.0 - a.budget))

    Acl, pcl = feats(Xte)
    Atr, ptr = feats(Xte + torch.from_numpy(trig_norm).to(dev).view(1, 1, -1))
    al_cl = _score_classcond(Acl, pcl, cstats, K) > thr
    al_tr = _score_classcond(Atr, ptr, cstats, K) > thr

    rec = meta["te_rec"]
    out = {"seed": seed, "threshold": thr, "n_windows": int(len(rec)),
           "recordings": meta["te_recordings"], "rules": {},
           # per-window benign alarm rate of each held-out recording: shows whether false alarms are
           # spread thinly or concentrated in one recording (an operating condition never seen in training)
           "per_recording_far": {meta["te_recordings"][r]: float(al_cl[rec == r].mean())
                                 for r in np.unique(rec)},
           "per_recording_windows": {meta["te_recordings"][r]: int((rec == r).sum()) for r in np.unique(rec)}}
    for (k, n) in rules:
        events, win_benign, win_esc, delays, censored = 0, 0, 0, [], 0
        for r in np.unique(rec):
            idx = np.where(rec == r)[0]                   # this recording, in time order
            st_b = escalated(al_cl[idx], k, n)
            events += rising_edges(st_b); win_benign += len(idx); win_esc += int(st_b.sum())
            if yte[idx[0]] == tgt:                        # triggers aim to pass a fault off as target
                continue
            for t0 in range(len(idx)):
                st = escalated(np.r_[al_cl[idx[:t0]], al_tr[idx[t0:]]], k, n)[t0:]
                hit = np.flatnonzero(st)
                if len(hit):
                    delays.append(int(hit[0]) + 1)
                else:
                    censored += 1
        out["rules"][f"{k}/{n}"] = {"benign_events": events, "benign_windows": win_benign,
                                    "benign_windows_escalated": win_esc,
                                    "delays": delays, "censored": censored}
    return out


def main():
    a = get_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rules = [tuple(int(v) for v in r.split("/")) for r in a.kn.split(",")]
    horizons = [int(h) for h in a.horizons.split(",")]
    runs = [one_seed(s, a, rules) for s in seeds]
    print(f"[{a.dataset}/{a.model}] done over {len(seeds)} seeds")

    win_s = a.sig_len / FS[a.dataset]                      # seconds of signal per window
    rate_sig = 1.0 / win_s                                 # windows/s, streaming at the signal rate
    rate_cpu = 1e6 / LAT_US[a.model]                       # windows/s, back-to-back inference
    lines = [f"# k-of-N alarm aggregation (dataset={a.dataset}, model={a.model}, static INT8 backdoor, "
             f"recording-level split, per-window budget={a.budget*100:.0f}%, seeds={seeds})",
             f"# benign: escalation events (rising edges) on time-ordered held-out clean recordings; "
             f"rates at the signal rate ({rate_sig:.1f} windows/s, {win_s*1e3:.1f} ms/window) and "
             f"at the compute-bound rate ({rate_cpu:.0f} inferences/s)",
             f"# delay: windows from trigger onset to first escalation, over every onset in every "
             f"non-target held-out recording; 'missed' = not escalated before the recording ended", "",
             f"{'k/N':6}{'events/h (signal)':>19}{'events/h (cpu)':>16}{'median delay':>14}"
             f"{'p90 delay':>11}{'(ms @signal)':>14}" + "".join(f"{'P(<=' + str(h) + ')':>9}" for h in horizons)
             + f"{'missed':>9}{'events':>8}{'95% UB/h':>10}{'%time esc.':>12}",
             "-" * (109 + 9 * len(horizons))]
    table = {}
    for (k, n) in rules:
        key = f"{k}/{n}"
        ev = np.array([r["rules"][key]["benign_events"] for r in runs], float)
        wn = np.array([r["rules"][key]["benign_windows"] for r in runs], float)
        per_win = ev / wn                                   # events per window, per seed
        esc = np.array([r["rules"][key]["benign_windows_escalated"] for r in runs], float).sum() / wn.sum()
        d_all = np.concatenate([np.array(r["rules"][key]["delays"], float) for r in runs] or [[]])
        cens = sum(r["rules"][key]["censored"] for r in runs)
        n_on = len(d_all) + cens
        # detection-within-H counts censored onsets as not detected
        pw = {h: float((d_all <= h).sum() / n_on) if n_on else float("nan") for h in horizons}
        med = float(np.median(d_all)) if len(d_all) else float("nan")
        p90 = float(np.percentile(d_all, 90)) if len(d_all) else float("nan")
        # exact Poisson 95% upper bound on the event rate, pooled exposure (matters when counts are ~0)
        try:
            from scipy.stats import chi2
            ub_events = float(chi2.ppf(0.975, 2 * (ev.sum() + 1)) / 2)
        except Exception:
            ub_events = float(ev.sum() + 3.0 * np.sqrt(ev.sum() + 1) + 3)
        ub_sig = ub_events / (wn.sum() / rate_sig) * 3600
        row = {"events_per_hour_signal_ub95": ub_sig,
               "events_per_hour_signal_pooled": float(ev.sum() / (wn.sum() / rate_sig) * 3600),
               "events_per_hour_signal": float(per_win.mean() * rate_sig * 3600),
               "events_per_hour_signal_std": float(per_win.std() * rate_sig * 3600),
               "events_per_hour_cpu": float(per_win.mean() * rate_cpu * 3600),
               "events_per_window": float(per_win.mean()),
               "benign_events_total": int(ev.sum()), "benign_windows_total": int(wn.sum()),
               "benign_exposure_s": float(wn.sum() * win_s), "frac_time_escalated": float(esc),
               "median_delay_windows": med, "p90_delay_windows": p90,
               "median_delay_ms_signal": med * win_s * 1e3, "p90_delay_ms_signal": p90 * win_s * 1e3,
               "p_detect_within": pw, "missed_frac": cens / n_on if n_on else float("nan"),
               "n_onsets": int(n_on)}
        table[key] = row
        lines.append(f"{key:6}{row['events_per_hour_signal']:>12.0f}+/-{row['events_per_hour_signal_std']:<5.0f}"
                     f"{row['events_per_hour_cpu']:>15.0f}{med:>13.1f}{p90:>11.1f}"
                     f"{row['median_delay_ms_signal']:>8.0f}/{row['p90_delay_ms_signal']:<5.0f}"
                     + "".join(f"{pw[h]:>9.2f}" for h in horizons) + f"{row['missed_frac']:>9.2f}"
                     + f"{int(ev.sum()):>8d}{ub_sig:>10.0f}{esc*100:>11.2f}%")
    lines.append("# per-window benign alarm rate by held-out recording (mean over the seeds that held it out):")
    acc = {}
    for r in runs:
        for name, f in r["per_recording_far"].items():
            acc.setdefault(name, []).append(f)
    for name in sorted(acc):
        lines.append(f"#   {name:36s} {np.mean(acc[name])*100:6.2f}%  (held out in {len(acc[name])} seed(s))")
    w_tot = sum(r["rules"][f"{rules[0][0]}/{rules[0][1]}"]["benign_windows"] for r in runs)
    lines.append(f"# benign exposure: {w_tot} clean windows = {w_tot * win_s:.0f} s of signal over "
                 f"{len(seeds)} seeds; 'events' is the raw count behind each per-hour rate")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.model}"
    open(os.path.join(a.out, f"agg_{tag}.txt"), "w").write(report + "\n")
    json.dump({"config": vars(a), "window_s": win_s, "rate_signal": rate_sig, "rate_cpu": rate_cpu,
               "table": table, "per_seed": runs}, open(os.path.join(a.out, f"agg_{tag}.json"), "w"), indent=2)
    print(f"Saved -> {os.path.abspath(a.out)}/agg_{tag}.{{txt,json}}")


if __name__ == "__main__":
    main()
