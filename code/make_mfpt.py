#!/usr/bin/env python3
"""
make_mfpt.py -- Prepare the MFPT bearing dataset into the project's class-folder layout.

Reads the MathWorks MFPT mirror (data/mfpt_raw/{train_data,test_data}/*.mat), extracts the 1-D
acceleration signal `bearing.gs` from each file, maps the file name to one of three classes
(baseline->normal, InnerRaceFault->inner, OuterRaceFault->outer), and writes a flat .mat with a
top-level 1-D `signal` array into data/mfpt/<class>/, which tinybd.data._load_folders reads directly.

The baseline files are sampled at 97656 Hz vs 48828 Hz for the fault files; we decimate the baseline
by 2 so all windows share a comparable time scale.

  conda activate qsentry
  KMP_DUPLICATE_LIB_OK=TRUE python make_mfpt.py
"""
import os, glob
import numpy as np
from scipy.io import loadmat, savemat

SRC = os.path.join(os.path.dirname(__file__), "..", "data", "mfpt_raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "mfpt")
FAULT_SR = 48828   # Hz; baseline is ~2x this and is decimated to match


def cls_of(name):
    n = name.lower()
    if n.startswith("baseline"):
        return "normal"
    if "innerrace" in n:
        return "inner"
    if "outerrace" in n:
        return "outer"
    return None


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*", "*.mat")))
    if not files:
        raise SystemExit(f"no MFPT .mat files under {SRC} (clone mathworks/RollingElementBearingFaultDiagnosis-Data)")
    counts = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        cls = cls_of(name)
        if cls is None:
            continue
        m = loadmat(f)
        b = m["bearing"]
        gs = np.asarray(b["gs"][0, 0]).squeeze().astype(np.float32)
        sr = int(np.asarray(b["sr"][0, 0]).squeeze())
        if sr > 1.5 * FAULT_SR:          # decimate the high-rate baseline to match the fault files
            gs = gs[::2]
        d = os.path.join(OUT, cls)
        os.makedirs(d, exist_ok=True)
        savemat(os.path.join(d, name + ".mat"), {"signal": gs})
        counts[cls] = counts.get(cls, 0) + len(gs)
    print("MFPT prepared ->", os.path.abspath(OUT))
    for c, n in sorted(counts.items()):
        print(f"  {c:7s}: {n} samples (~{n // 1024} windows of 1024)")


if __name__ == "__main__":
    main()
