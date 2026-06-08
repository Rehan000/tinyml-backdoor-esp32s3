# Datasets

The pipeline runs **today** on a built-in **synthetic** bearing-vibration generator (no download).
For the paper, drop in a real dataset using the simple class-folder convention below; the loader
(`tinybd/data.py`) reads the first 1-D vibration array per `.mat`, windows it, and splits.

## Convention: one sub-folder per class
```
data/
  cwru/          # Case Western Reserve University bearing data
    normal/   *.mat
    inner/    *.mat
    outer/    *.mat
    ball/     *.mat
  paderborn/     # Paderborn University bearing data
    healthy/  *.mat
    inner/    *.mat
    outer/    *.mat
```
You organize the downloaded files into these class folders (sidesteps the datasets' messy native
naming). The class is the folder name; ordering is alphabetical → integer labels.

## DOWNLOADED: CWRU subset (already in data/cwru/)
12 kHz drive-end, 0.007" faults, multiple motor loads (file numbers from CWRU):
- `normal/`  97, 98, 100        (baseline; 99 was a corrupt download, omitted)
- `inner/`   105, 106, 107, 108 (inner-race fault)
- `ball/`    118, 119, 120, 121 (ball fault)
- `outer/`   130, 131, 132, 133 (outer-race fault @6:00)
Loader picks the drive-end channel (`*_DE_time`). Classes are ALPHABETICAL ->
ball=0, inner=1, **normal=2**, outer=3. For the safety framing (hide a fault as normal) use
**`--target-class 2`**. Source base URL: https://engineering.case.edu/sites/default/files/<n>.mat

## Getting the data (other datasets)
- **CWRU**: https://engineering.case.edu/bearingdatacenter/download-data-file — download the
  drive-end (DE) `.mat` files and sort by fault type into the folders above.
- **Paderborn**: https://mb.uni-paderborn.de/kat/datensaetze — KAt bearing dataset.
- **MFPT**: clone the MathWorks mirror and run the prep script (handles the nested `bearing.gs` struct
  and decimates the high-rate baseline):
  ```
  git clone https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data data/mfpt_raw
  python code/make_mfpt.py        # -> data/mfpt/{normal,inner,outer}/
  ```
  MFPT classes are ALPHABETICAL -> inner=0, **normal=1**, outer=2; use **`--target-class 1`**.

## Requirements for real data
Real `.mat` loading needs SciPy (not required for synthetic):
```
conda install -n qsentry scipy
```

## Use it
```
python train.py    --dataset cwru --data-dir ../data --attack static
python evaluate.py  --run ../results/phase0
```
If the loader can't find class folders or usable signals, it raises a clear error pointing here.
Tune `--sig-len` (window length) to your sampling rate; default 1024.
