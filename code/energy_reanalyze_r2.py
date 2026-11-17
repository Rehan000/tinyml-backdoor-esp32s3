#!/usr/bin/env python3
"""
energy_reanalyze_r2.py -- Cold-path monitor energy from the EXISTING revision-2 PPK2 trace.

The R2 firmware (results/deploy_wp7/firmware_snapshot/qsentry_bench.wp7.cpp) already ran three regions
per round, in this order, each followed by an idle gap:
    A   : inference                              x 200
    E2E : inference + 1x CCM (the deployed path)   x 200
    B   : inference + 1024x CCM (amplified)         x 200
The R2 analysis split humps only by duration into "short" and "long", so A and E2E were pooled and the
cold-path difference E2E - A was never computed; the reported 3.0 uJ came from B (hot cache). This
script pairs each round's humps by TIME ORDER (short, short, long) and reports, per round:
    cold-path CCM energy  = (E_E2E - E_A) / 200
    cold-path CCM time    = (T_E2E - T_A) / 200     (cross-check against the on-chip timer, 39.6 us)
    implied power of the extra work = (E_E2E - E_A) / (T_E2E - T_A)
The last line tests the assumption behind estimating energy as latency x active power.

  python energy_reanalyze_r2.py --csv ../results/energy/ppk2.csv
"""
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/energy/ppk2.csv")
    ap.add_argument("--voltage", type=float, default=3.3)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--smooth-ms", type=float, default=20.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = np.loadtxt(a.csv, delimiter=",", skiprows=1, usecols=(0, 1))
    t_ms, I = d[:, 0], d[:, 1]
    dt = float(np.median(np.diff(t_ms))) / 1e3
    w = max(1, int(a.smooth_ms / 1e3 / dt))
    Is = np.convolve(I, np.ones(w) / w, mode="same")
    idle, peak = np.percentile(Is, 10), np.percentile(Is, 95)
    thr = idle + 0.4 * (peak - idle)
    idx = np.flatnonzero(Is > thr)
    parts = np.split(idx, np.flatnonzero(np.diff(idx) > w) + 1)
    hs = [(p[0], p[-1] + 1) for p in parts if (p[-1] - p[0]) * dt > 0.4]
    dur = np.array([(e - s) * dt for s, e in hs])
    E = np.array([a.voltage * I[s:e].sum() * 1e-6 * dt for s, e in hs])
    long_ = dur > 0.5 * (dur.min() + dur.max())

    rounds = []
    for k in range(2, len(hs)):
        if long_[k] and not long_[k - 1] and not long_[k - 2]:
            iA, iE = k - 2, k - 1
            rounds.append({"EA": E[iA], "EE": E[iE], "TA": dur[iA], "TE": dur[iE]})
    if not rounds:
        raise SystemExit("no (short, short, long) round pattern found")
    EA = np.array([r["EA"] for r in rounds]); EE = np.array([r["EE"] for r in rounds])
    TA = np.array([r["TA"] for r in rounds]); TE = np.array([r["TE"] for r in rounds])
    e_inf = EA / a.iters
    e_cold = (EE - EA) / a.iters
    t_cold = (TE - TA) / a.iters
    p_extra = (EE - EA) / (TE - TA)

    def ms(x):
        return float(np.mean(x)), float(np.std(x, ddof=1)), float(np.median(x))
    res = {"rounds": len(rounds), "idle_mW": a.voltage * idle * 1e-3,
           "inference_uJ": ms(e_inf * 1e6), "ccm_cold_uJ": ms(e_cold * 1e6),
           "ccm_cold_us_from_trace": ms(t_cold * 1e6), "extra_work_power_mW": ms(p_extra * 1e3),
           "ccm_cold_pct_of_inference": 100 * float(np.median(e_cold) / np.median(e_inf))}
    print(f"rounds paired: {len(rounds)}   (humps: {len(hs)}, {int((~long_).sum())} short / {int(long_.sum())} long)")
    print(f"inference         : {res['inference_uJ'][2]:8.1f} uJ/inf (median; mean {res['inference_uJ'][0]:.1f} +/- {res['inference_uJ'][1]:.1f})")
    print(f"CCM cold (E2E-A)  : {res['ccm_cold_uJ'][2]:8.2f} uJ/inf (median; mean {res['ccm_cold_uJ'][0]:.2f} +/- {res['ccm_cold_uJ'][1]:.2f})"
          f"  = {res['ccm_cold_pct_of_inference']:.2f}% of inference")
    print(f"CCM cold time     : {res['ccm_cold_us_from_trace'][2]:8.1f} us/inf from hump widths "
          f"(mean {res['ccm_cold_us_from_trace'][0]:.1f} +/- {res['ccm_cold_us_from_trace'][1]:.1f}); on-chip timer said 39.6 us")
    print(f"power of the extra work: {res['extra_work_power_mW'][2]:.0f} mW (median; mean {res['extra_work_power_mW'][0]:.0f} "
          f"+/- {res['extra_work_power_mW'][1]:.0f}); inference runs at ~463 mW, idle {res['idle_mW']:.0f} mW")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
        print(f"saved -> {a.json}")


if __name__ == "__main__":
    main()
