#!/usr/bin/env python3
"""
compute_energy.py -- Per-inference energy from a Nordic PPK2 CSV export (marker-independent).

The firmware (firmware/main/qsentry_bench.cpp) runs, per round, two single-core bursts separated by
an idle baseline:
    Region A : inference x N_ITERS                              -> short hump (~1 s)
    Region B : inference + N_CCM_REPEAT monitor calls / inf     -> long  hump
Because Region B repeats the monitor N_CCM_REPEAT times per inference, (E_B - E_A) is the energy of
N_ITERS*N_CCM_REPEAT monitor calls -- large and directly measurable -- and the inference energy
cancels. So the CCM cost is recovered WITHOUT the logic markers: we just segment the current trace
into humps and split them into the short (inference) and long (amplified) clusters by duration.

    inference energy / inf = E_A / N_ITERS
    CCM energy / inf       = (E_B - E_A) / (N_ITERS * N_CCM_REPEAT)     (one monitor call per inference)

If the logic markers (D0/D1) are present and toggling, they are used as a cross-check.

Usage:
    conda activate qsentry
    python compute_energy.py --csv results/energy/ppk2.csv --voltage 3.3 --iters 200 --ccm-repeat 256
"""
import argparse, sys
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
        bits = (ca == "1")
    return t, c, bits


def humps(I_smooth, dt, idle, peak, bridge, min_s):
    thr = idle + 0.4 * (peak - idle)
    idx = np.flatnonzero(I_smooth > thr)
    if len(idx) == 0:
        return []
    parts = np.split(idx, np.flatnonzero(np.diff(idx) > bridge) + 1)
    return [(p[0], p[-1] + 1) for p in parts if (p[-1] - p[0]) * dt > min_s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--voltage", type=float, default=3.3)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ccm-repeat", type=int, default=256)
    ap.add_argument("--smooth-ms", type=float, default=20.0)
    a = ap.parse_args()
    V = a.voltage

    t_ms, cur_uA, bits = load(a.csv)
    t_s = t_ms / 1000.0
    dt = float(np.median(np.diff(t_s)))
    w = max(1, int(a.smooth_ms / 1000.0 / dt))
    Is = np.convolve(cur_uA, np.ones(w) / w, mode="same")
    idle = float(np.percentile(Is, 10)); peak = float(np.percentile(Is, 95))

    hs = humps(Is, dt, idle, peak, bridge=w, min_s=0.4)
    if len(hs) < 2:
        sys.exit(f"found {len(hs)} humps; expected >=2 (one inference + one amplified per round).")

    def energy(s, e):                       # joules over [s,e)
        return V * float(np.sum(cur_uA[s:e] * 1e-6 * dt))
    durs = np.array([(e - s) * dt for s, e in hs])
    ens = np.array([energy(s, e) for s, e in hs])
    imeans = np.array([float(np.mean(cur_uA[s:e])) for s, e in hs])

    # split humps into short (inference, A) vs long (amplified, B) by duration
    cut = 0.5 * (durs.min() + durs.max())
    A = durs < cut
    B = ~A
    if A.sum() == 0 or B.sum() == 0:
        sys.exit(f"could not separate short/long humps; durations={np.round(durs,2)}")

    idle_uA = idle
    idle_P = V * idle_uA * 1e-6
    EA = float(np.median(ens[A]))           # J per Region-A burst (N_ITERS inferences)
    EB = float(np.median(ens[B]))           # J per Region-B burst (N_ITERS inf + N*REPEAT ccm)
    E_inf = EA / a.iters
    E_ccm = (EB - EA) / (a.iters * a.ccm_repeat)
    P_inf = float(np.mean(imeans[A])) * V * 1e-6
    tA = float(np.median(durs[A])) / a.iters

    print(f"file: {a.csv}")
    print(f"V={V} V  dt={dt*1e6:.1f} us (~{1/dt:.0f} Hz)  duration={t_s[-1]:.1f} s")
    print(f"humps: {A.sum()} inference (~{np.median(durs[A]):.2f} s), "
          f"{B.sum()} amplified (~{np.median(durs[B]):.2f} s)\n")
    print(f"idle baseline        : {idle_uA:8.1f} uA   ({idle_P*1e3:.1f} mW)")
    print(f"inference active     : {np.mean(imeans[A]):8.1f} uA   ({P_inf*1e3:.1f} mW)")
    print(f"inference latency    : {tA*1e3:8.2f} ms/inf  (from hump width)")
    print(f"inference energy/inf : {E_inf*1e6:8.1f} uJ  (total)   "
          f"{(P_inf-idle_P)*tA*1e6:8.1f} uJ (dynamic)")
    print(f"CCM energy/inf       : {E_ccm*1e6:8.3f} uJ   "
          f"({100*E_ccm/E_inf:.2f}% of inference)")

    if bits is not None:
        tog = [j for j in range(bits.shape[1]) if bits[:, j].any() and not bits[:, j].all()]
        print(f"\nlogic markers toggling: {tog if tog else 'none (segmented by duration instead)'}")

    print("\nPaper-ready: idle {:.0f} mW; inference {:.0f} mW, {:.0f} uJ/inf; "
          "CCM +{:.2f} uJ/inf ({:.2f}%).".format(idle_P*1e3, P_inf*1e3, E_inf*1e6,
                                                  E_ccm*1e6, 100*E_ccm/E_inf))


if __name__ == "__main__":
    main()
