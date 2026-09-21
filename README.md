# On-Device Backdoor Attacks and Defense for Quantized TinyML on the ESP32-S3

Code, conversion pipeline, and firmware harness for the paper:

> **On-Device Backdoor Attacks and Defense for Quantized TinyML on Microcontrollers: A Measured Study on the ESP32-S3**
> Muhammad Rehan, Muhammad Ali Munir, Haider Ali. *(Journal of Systems Architecture, under review;
> minor revision.)*

This repository reproduces, end to end, a **measured, on-device** study of backdoor attacks and a
resource-aware defense for INT8-quantized TinyML on a commodity microcontroller (Espressif ESP32-S3,
ESP-DL pipeline), on the CWRU and MFPT bearing-fault benchmarks.

## What's here

- **Quantization-conditioned backdoors (QCBs).** Two faithful constructions — a scale-pinned
  bin-projection and a joint post-training-quantization optimization with a straight-through estimator —
  used to characterize a **bit-width threshold**: conditioning is reproducible at INT4, while neither
  construction achieves it at INT8 on shallow TinyML CNNs in any of the sixteen settings tested. That
  is an empirical finding under symmetric post-training quantization, not a proof of impossibility.
- **CCM, a class-conditional INT8 activation monitor** that runs on-device over already-resident
  features to flag triggered inputs.
- **On-device measurement harness** (latency, peak RAM, energy) on real silicon, including the
  defense's per-inference overhead.
- **Generalization** across two architectures (`VibCNN`, `WDCNN`), three datasets (CWRU, MFPT, and
  low-resolution MNIST), and four trigger designs (localized burst, full-window tone, frequency-domain
  spectral spike, broadband distributed).
- **Detector baselines with on-device cost**: CCM against diagonal Mahalanobis, class-centroid (L2 and
  cosine), a thresholded activation norm, and a one-class kNN, each reported with the floats it must
  store and the multiply-adds it costs per inference.
- **Adaptive attackers**: a train-time evasion-vs-utility frontier on the monitored layer, the same
  attack against an unmonitored layer (which does evade the monitor, at an accuracy cost), and a
  bounded test-time PGD attack (which does not move it).
- **Calibration protocol matters.** The monitor and the quantizer share a small clean calibration set.
  Drawing it as the first N training windows gives a class-ordered sample covering only one or two
  classes, which both weakens the monitor and flatters the "global" ablation it is compared against.
  `code/calib_draw.py` measures the effect; reported results use a class-representative draw
  (`tinybd.data.calib_indices`).

## Repository layout

```
code/                       host-side Python (training, attacks, defense, evaluation, export, figures)
  tinybd/                   package: data, model (VibCNN/WDCNN), quant, backdoor, defense, metrics, export
  run_experiments.py        full attack x bit-width x seed matrix (CCM vs global vs RQP)
  train.py                  train one model under one attack; saves a deployable checkpoint
  convert_espdl.py          quantize a checkpoint to an on-device .espdl model (esp-ppq)
  export_testset.py         embed a test set + host predictions for the on-device device==host check
  export_monitor.py         dump CCM parameters as C arrays for the firmware
  make_figures.py           regenerate the paper figures (matplotlib)
  make_mfpt.py              prepare the MFPT dataset into the class-folder layout
  compute_energy.py         per-inference energy from a Nordic PPK2 CSV export
  capture_ppk2.py           drive a Nordic PPK2 over USB and log the current trace
  eval_baselines.py         CCM vs the lightweight MCU-feasible detectors, with their cost
  adaptive_attack.py        train-time (monitored / unmonitored layer) and test-time PGD attackers
  image_experiments.py      the bit-width study on a second domain (low-res MNIST)
  poisoned_calib.py         detection under a partially poisoned calibration set
  calib_draw.py             how the calibration draw changes detection
  false_alarm.py            benign alarm rate at a fixed false-alarm budget
  stats.py                  paired bootstrap CIs and Wilcoxon tests over the result grid
  make_tables.py            regenerate the paper tables from results/
firmware/                   ESP-IDF project (qsentry_bench.cpp = on-device measurement harness)
data/                       dataset README + class-folder layout (datasets downloaded separately)
```

## Environment

Two conda environments and (for the firmware) ESP-IDF v5.x:

```bash
# host-side experiments / figures / energy analysis
conda create -n qsentry python=3.11 numpy scipy matplotlib pytorch -c pytorch
# torch needs the OpenMP workaround on some systems:
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

# model -> .espdl conversion (Espressif quantization toolkit)
conda create -n espdl python=3.10 && conda activate espdl && pip install esp-ppq

# firmware: install ESP-IDF v5.x  (https://docs.espressif.com/projects/esp-idf)
```

## Datasets

See [`data/README.md`](data/README.md). In brief:
- **CWRU** — download the drive-end `.mat` files and sort into `data/cwru/{normal,inner,outer,ball}/`.
- **MFPT** — `git clone https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data data/mfpt_raw`
  then `python code/make_mfpt.py` (writes `data/mfpt/{normal,inner,outer}/`).

## Reproduce the host-side results

```bash
conda activate qsentry
cd code
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1

# attack x bit-width x detection matrix (3 seeds). normal = class 2 on CWRU, class 1 on MFPT.
python run_experiments.py --dataset cwru --model vibcnn --target-class 2 --seeds 0,1,2
python run_experiments.py --dataset cwru --model wdcnn  --target-class 2 --seeds 0,1,2
python run_experiments.py --dataset mfpt --model vibcnn --target-class 1 --seeds 0,1,2
python run_experiments.py --dataset mfpt --model wdcnn  --target-class 1 --seeds 0,1,2

# helper data for two figures, then the figures themselves -> ../paper/figs/
python make_trigger_example.py
python make_mechanism_data.py --dataset cwru --seeds 0,1,2
python make_figures.py
```

## Reproduce the on-device measurements (ESP32-S3)

```bash
# 1) train a deployable model, 2) emit the device==host vectors + CCM params, 3) quantize to .espdl
conda activate qsentry
python train.py --dataset cwru --model vibcnn --attack clean --out ../results/deploy
python export_testset.py --run ../results/deploy -k 512   # full test set, not a 64-sample subset
python export_monitor.py --run ../results/deploy
conda activate espdl
python convert_espdl.py --run ../results/deploy --target esp32s3

# 4) build & flash the measurement firmware
cd ../firmware && idf.py build && idf.py -p <PORT> flash monitor
```

The firmware reports inference latency (single/dual core), peak RAM, and the device-vs-host agreement,
and raises GPIO markers around each region so an external **Nordic PPK2** can segment energy. Convert a
PPK2 CSV export to per-inference energy with:

```bash
python compute_energy.py --csv results/energy/ppk2.csv --voltage 3.3 --iters 200 --ccm-repeat 1024
```

## Key measured results (ESP32-S3, INT8, 240 MHz, single core)

| | VibCNN | WDCNN |
|---|---|---|
| Inference latency | 3677.2 ± 0.2 µs | 2054.4 ± 1.7 µs |
| Dual-core latency | 3694 µs (no speedup) | 2077 µs (no speedup) |
| Model RAM (internal / PSRAM) | 6.6 / 41.8 KB | 7.3 / 37.8 KB |
| End-to-end (inference + input quant + 1× CCM) | 3716.8 ± 0.2 µs | 2090.0 ± 1.9 µs |
| CCM latency overhead (cold, single-shot) | 39.6 µs (1.08%) | 35.7 µs (1.74%) |
| CCM per-call cost (hot cache, amplified) | 7.57 µs | 7.57 µs (64-d, identical) |
| CCM energy overhead | 3.03 µJ (0.18% of 1.72 mJ/inference) | (identical, 64-d) |
| Device == host | 99.0% (N=512, full set) | 100% (N=128) |

Board power at 240 MHz: 132.1 mW idle, 463.1 mW during inference (Nordic PPK2, 3.3 V).

Neither construction achieves INT8 conditioning for either architecture on any of the three datasets;
the CCM is a near-free on-device monitor, and its detection advantage over a global statistic depends
on the setting — and, as the revision found, on how the calibration set is drawn (see below).

## Citation

```bibtex
@article{rehan2026tinymlbackdoor,
  title   = {On-Device Backdoor Attacks and Defense for Quantized TinyML on Microcontrollers:
             A Measured Study on the ESP32-S3},
  author  = {Rehan, Muhammad and Munir, Muhammad Ali and Ali, Haider},
  journal = {Journal of Systems Architecture (under review)},
  year    = {2026}
}
```

## License

[MIT](LICENSE).
