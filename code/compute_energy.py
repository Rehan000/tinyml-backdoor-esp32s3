#!/usr/bin/env python3
"""
compute_energy.py -- Per-inference and COLD-PATH monitor energy from a Nordic PPK2 CSV export.

Revision 3 layout (firmware/main/qsentry_bench.cpp, phase 2). Each of the N_ROUNDS rounds runs three
single-core bursts, separated by idle gaps, in this order:
    A   : inference                                  x ITERS_CE    (PPK2 logic D0 high)
    E2E : inference + feature read + 1x CCM           x ITERS_CE    (D2 high)  <- the deployed path
    B   : inference + REPEAT CCM calls (hot cache)    x ITERS_B     (D1 high)

    inference energy / inf      = E_A / ITERS_CE
    CCM energy, COLD (deployed) = (E_E2E - E_A) / ITERS_CE          <- what the paper reports now
    CCM energy, hot (lower bd.) = (E_B - E_A * ITERS_B/ITERS_CE) / (ITERS_B * REPEAT)

The previous revision reported only the hot-cache figure; Reviewer 4 pointed out that it contradicts
our own "measure cold" lesson. Both are reported here, per round, as median and spread.

Segmentation: the trace is split into humps of high current; the energy rounds are the LAST
3*N_ROUNDS humps (the detection phase before them is separated by a 5 s idle) and are taken in
order A, E2E, B. When the logic markers are present and toggling, the marker-gated integrals are
computed as well and printed side by side -- they do not depend on hump-edge detection.
The previous (two-region) script is archived in results/deploy_wp7/firmware_snapshot/.

Usage:
    conda activate qsentry
    python compute_energy.py --csv results/energy/ppk2_r3.csv --voltage 3.3
"""
import argparse, sys, json
import numpy as np


def load(path):
    t, c, codes = [], [], []
    with open(path) as f:
        header = f.readline().strip().split(",")
        low = [h.lower() for h in header]
        ti = next(i for i, h in enumerate(low) if "time" in h)
        ci = next(i for i, h in enumerate(low) if "current" in h)
        di = next((i for i, h in enumerate(low) if "-d7" in h or h in ("d0-d7", "logic")), None)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) <= ci:
                continue
            t.append(p[ti]); c.append(p[ci]); codes.append(p[di].strip() if di is not None else "")
    t = np.array(t, np.float64); c = np.array(c, np.float64)
    bits = None
    if codes and codes[0]:
        w = len(codes[0])
        ca = np.array(codes, dtype=f"<U{w}").view("<U1").reshape(len(codes), w)
        bits = (ca == "1")          # column j = logic channel D{j} (PPK2 exports D0 first)
    return t, c, bits


def humps(I_smooth, dt, idle, peak, bridge, min_s):
    thr = idle + 0.4 * (peak - idle)
    idx = np.flatnonzero(I_smooth > thr)
    if len(idx) == 0:
        return []
    parts = np.split(idx, np.flatnonzero(np.diff(idx) > bridge) + 1)
    return [(p[0], p[-1] + 1) for p in parts if (p[-1] - p[0]) * dt > min_s]


def runs(b):
    """(start, end) index pairs where boolean array b is True."""
    d = np.diff(np.r_[0, b.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def per_round(EA, EE, EB, a):
    e_inf = EA / a.iters_ce
    e_cold = (EE - EA) / a.iters_ce
    e_hot = (EB - EA * a.iters_b / a.iters_ce) / (a.iters_b * a.repeat)
    return e_inf, e_cold, e_hot


def ms(x):
    return float(np.median(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--voltage", type=float, default=3.3)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--iters-ce", type=int, default=1000)
    ap.add_argument("--iters-b", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1024)
    ap.add_argument("--smooth-ms", type=float, default=20.0)
    ap.add_argument("--json", default=None, help="optional path to write the results as JSON")
    a = ap.parse_args()
    V = a.voltage

    t_ms, cur_uA, bits = load(a.csv)
    dt = float(np.median(np.diff(t_ms))) / 1000.0
    w = max(1, int(a.smooth_ms / 1000.0 / dt))
    Is = np.convolve(cur_uA, np.ones(w) / w, mode="same")
    idle = float(np.percentile(Is, 10)); peak = float(np.percentile(Is, 95))
    E = lambda s, e: V * float(np.sum(cur_uA[s:e])) * 1e-6 * dt      # joules over [s, e)
    idle_P = V * idle * 1e-6

    hs = humps(Is, dt, idle, peak, bridge=w, min_s=0.4)
    need = 3 * a.rounds
    if len(hs) < need:
        sys.exit(f"found {len(hs)} humps, need >= {need} (3 per round). Capture longer, or check wiring.")
    hs = hs[-need:]
    trip = np.array(hs).reshape(a.rounds, 3, 2)
    durs = (trip[:, :, 1] - trip[:, :, 0]) * dt
    ens = np.array([[E(s, e) for s, e in r] for r in trip])
    # sanity: A and E2E must be near-equal in length, B distinct
    if np.any(np.abs(durs[:, 1] / durs[:, 0] - 1) > 0.05):
        sys.exit(f"hump pattern is not (A, E2E, B): durations per round =\n{np.round(durs, 3)}")
    e_inf, e_cold, e_hot = per_round(ens[:, 0], ens[:, 1], ens[:, 2], a)
    t_cold = (durs[:, 1] - durs[:, 0]) / a.iters_ce
    P_inf = ens[:, 0] / durs[:, 0]

    res = {"source": "hump", "idle_mW": idle_P * 1e3, "inference_mW": float(np.median(P_inf)) * 1e3,
           "inference_uJ": ms(e_inf * 1e6), "ccm_cold_uJ": ms(e_cold * 1e6), "ccm_hot_uJ": ms(e_hot * 1e6),
           "ccm_cold_us_from_trace": ms(t_cold * 1e6), "inference_ms_from_trace": float(np.median(durs[:, 0])) / a.iters_ce * 1e3}

    print(f"file: {a.csv}   V={V} V  dt={dt*1e6:.1f} us  duration={t_ms[-1]/1e3:.1f} s  "
          f"rounds={a.rounds}  A/E2E={a.iters_ce} iters  B={a.iters_b}x{a.repeat}")
    print(f"idle {idle_P*1e3:.1f} mW | inference {res['inference_mW']:.1f} mW | "
          f"latency from trace {res['inference_ms_from_trace']:.3f} ms/inf")
    print(f"[humps]   inference {res['inference_uJ'][0]:8.1f} +/- {res['inference_uJ'][1]:.1f} uJ/inf")
    print(f"[humps]   CCM cold  {res['ccm_cold_uJ'][0]:8.2f} +/- {res['ccm_cold_uJ'][1]:.2f} uJ/inf "
          f"({100*res['ccm_cold_uJ'][0]/res['inference_uJ'][0]:.2f}% of inference)   "
          f"[time from trace {res['ccm_cold_us_from_trace'][0]:.1f} us]")
    print(f"[humps]   CCM hot   {res['ccm_hot_uJ'][0]:8.3f} +/- {res['ccm_hot_uJ'][1]:.3f} uJ/call (lower bound)")

    if bits is not None and bits.shape[1] >= 3:
        rA, rB, rE = runs(bits[:, 0]), runs(bits[:, 1]), runs(bits[:, 2])
        if min(len(rA), len(rB), len(rE)) >= a.rounds:
            rA, rB, rE = rA[-a.rounds:], rB[-a.rounds:], rE[-a.rounds:]
            mA = np.array([E(s, e) for s, e in rA]); mE = np.array([E(s, e) for s, e in rE])
            mB = np.array([E(s, e) for s, e in rB])
            mi, mc, mh = per_round(mA, mE, mB, a)
            tA = np.array([(e - s) * dt for s, e in rA]); tE = np.array([(e - s) * dt for s, e in rE])
            res["markers"] = {"inference_uJ": ms(mi * 1e6), "ccm_cold_uJ": ms(mc * 1e6),
                              "ccm_hot_uJ": ms(mh * 1e6), "ccm_cold_us": ms((tE - tA) / a.iters_ce * 1e6)}
            m = res["markers"]
            print(f"[markers] inference {m['inference_uJ'][0]:8.1f} +/- {m['inference_uJ'][1]:.1f} uJ/inf")
            print(f"[markers] CCM cold  {m['ccm_cold_uJ'][0]:8.2f} +/- {m['ccm_cold_uJ'][1]:.2f} uJ/inf "
                  f"({100*m['ccm_cold_uJ'][0]/m['inference_uJ'][0]:.2f}%)   [gated time {m['ccm_cold_us'][0]:.1f} us]")
            print(f"[markers] CCM hot   {m['ccm_hot_uJ'][0]:8.3f} +/- {m['ccm_hot_uJ'][1]:.3f} uJ/call")
        else:
            print(f"logic markers: found {len(rA)}/{len(rE)}/{len(rB)} runs on D0/D2/D1, need {a.rounds} each "
                  f"-- using hump segmentation only")
    else:
        print("logic markers: not present in the export -- using hump segmentation only")

    print(f"\nPaper-ready: idle {res['idle_mW']:.0f} mW; inference {res['inference_mW']:.0f} mW, "
          f"{res['inference_uJ'][0]:.0f} uJ/inf; CCM cold +{res['ccm_cold_uJ'][0]:.1f} uJ/inf "
          f"({100*res['ccm_cold_uJ'][0]/res['inference_uJ'][0]:.2f}%); hot {res['ccm_hot_uJ'][0]:.2f} uJ/call.")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
        print(f"saved -> {a.json}")


if __name__ == "__main__":
    main()
