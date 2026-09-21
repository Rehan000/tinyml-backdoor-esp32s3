#!/usr/bin/env python3
"""
capture_ppk2.py -- Drive a Nordic PPK2 in source-meter mode and log a current trace to CSV.

Companion to compute_energy.py. The PPK2 powers the ESP32-S3 (VOUT->3V3, GND->GND) at a fixed
voltage and samples current at 100 ksps. The firmware (qsentry_bench.cpp) runs its 12-round
measurement loop ONCE at boot and then idles, so we power-cycle the DUT (toggle_DUT_power OFF->ON)
at the start of the capture to guarantee the inference / amplified-monitor humps land inside the
window. The resulting CSV (Timestamp(ms),Current(uA)) is consumed by compute_energy.py, which
segments the trace into inference vs amplified humps by duration (logic markers not required).

Usage:
    conda activate qsentry
    python code/capture_ppk2.py --out results/energy/ppk2.csv --voltage 3300 --seconds 95
"""
import argparse, time, sys
import numpy as np
from ppk2_api.ppk2_api import PPK2_API


def pick_port():
    ports = PPK2_API.list_devices()
    for p in ports:
        try:
            dev = PPK2_API(p)
            if dev.get_modifiers():          # only the data interface returns calibration modifiers
                return p, dev
        except Exception:
            continue
    sys.exit(f"no PPK2 data interface responded (saw {ports})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/energy/ppk2.csv")
    ap.add_argument("--voltage", type=int, default=3300, help="source voltage in mV")
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument("--no-cycle", action="store_true", help="do not power-cycle the DUT at start")
    a = ap.parse_args()

    port, ppk2 = pick_port()
    print(f"PPK2 data port: {port}")
    ppk2.use_source_meter()
    ppk2.set_source_voltage(a.voltage)
    print(f"source-meter mode @ {a.voltage} mV")

    if not a.no_cycle:
        ppk2.toggle_DUT_power("OFF")
        time.sleep(1.5)                       # let the board fully power down
    ppk2.start_measuring()
    ppk2.toggle_DUT_power("ON")               # t~=0: board boots and runs the measurement loop
    print(f"DUT powered on; capturing {a.seconds:.0f} s ...")

    samples = []
    t0 = time.time()
    last = t0
    while time.time() - t0 < a.seconds:
        buf = ppk2.get_data()
        if buf:
            s, _ = ppk2.get_samples(buf)
            samples.extend(s)
        else:
            time.sleep(0.001)
        now = time.time()
        if now - last >= 5.0 and samples:
            recent = np.array(samples[-20000:], dtype=np.float64)
            print(f"  t={now-t0:5.1f}s  n={len(samples):>9,}  "
                  f"I(recent) min={recent.min():7.0f} max={recent.max():7.0f} uA")
            last = now

    ppk2.stop_measuring()
    try:
        ppk2.toggle_DUT_power("OFF")
    except Exception:
        pass

    cur = np.array(samples, dtype=np.float64)
    n = len(cur)
    if n == 0:
        sys.exit("captured 0 samples -- check wiring / that the DUT is on VOUT")
    dt_ms = 0.01                              # PPK2 ADC = 100 ksps -> 10 us/sample
    print(f"captured {n:,} samples (~{n*dt_ms/1000:.1f} s), "
          f"overall I: min={cur.min():.0f} max={cur.max():.0f} mean={cur.mean():.0f} uA")

    t = np.arange(n, dtype=np.float64) * dt_ms
    with open(a.out, "w") as f:
        f.write("Timestamp(ms),Current(uA)\n")
        CH = 1_000_000
        for i in range(0, n, CH):
            j = min(i + CH, n)
            tt, cc = t[i:j], cur[i:j]
            f.write("\n".join(f"{tt[k]:.2f},{cc[k]:.3f}" for k in range(len(tt))))
            f.write("\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
