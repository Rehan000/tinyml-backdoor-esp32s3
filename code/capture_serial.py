#!/usr/bin/env python3
"""
capture_serial.py -- Log the ESP32-S3 harness console to a file until it prints DONE.

`idf.py monitor` needs a TTY and cannot be piped reliably, so this opens the port with pyserial, resets
the board (RTS pulse on EN; works for the USB-UART bridge and the native USB-Serial-JTAG port), and
writes every line to --out. Progress is echoed every few hundred records.

  python capture_serial.py --port /dev/cu.usbmodem1101 --out ../results/deploy_r3/device_log.txt
"""
import argparse, sys, time

try:
    import serial
except ImportError:
    sys.exit("needs pyserial: `pip install pyserial` (or run from the ESP-IDF python env)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="../results/deploy_r3/device_log.txt")
    ap.add_argument("--timeout", type=float, default=600.0, help="give up after this many seconds")
    ap.add_argument("--no-reset", action="store_true")
    a = ap.parse_args()

    s = serial.Serial(a.port, a.baud, timeout=1.0)
    if not a.no_reset:
        s.dtr = False; s.rts = True; time.sleep(0.2); s.rts = False   # pulse EN low -> reboot
    t0 = time.time(); n = 0; nD = 0
    with open(a.out, "w") as f:
        while time.time() - t0 < a.timeout:
            raw = s.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            f.write(line + "\n"); n += 1
            if line.startswith("D,"):
                nD += 1
                if nD % 200 == 0:
                    print(f"  {time.time()-t0:6.1f}s  {nD} detection records", flush=True)
            elif not line.startswith("F,"):
                print(line, flush=True)
            if line.strip() == "DONE":
                break
        else:
            print(f"timeout after {a.timeout:.0f}s without DONE ({n} lines saved)")
    print(f"wrote {n} lines -> {a.out}")


if __name__ == "__main__":
    main()
