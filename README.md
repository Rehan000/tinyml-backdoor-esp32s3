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

The full procedure, with wiring, is in [`firmware/README.md`](firmware/README.md). In brief:

```bash
cd code
conda activate qsentry
python train.py --dataset cwru --model vibcnn --attack static --target-class 2 --seed 0 --out ../results/deploy_r3
conda activate espdl                                   # two-output graph (logits + features) + host simulation
python convert_espdl.py --run ../results/deploy_r3 --target esp32s3 --with-features --simulate
conda activate qsentry
python export_monitor.py --run ../results/deploy_r3 --from-sim   # tables fitted on the deployed graph
python export_testset.py --run ../results/deploy_r3              # all 523 test windows + the trigger
cd ../firmware && idf.py build && idf.py -p <PORT> flash
python ../code/capture_serial.py --port <PORT> --out ../results/deploy_r3/device_log.txt
cd ../code && python ondevice_detection.py --run ../results/deploy_r3
```

The firmware runs the monitor **on the device** over the full test split, clean and triggered, and
reports detection (AUROC, TPR at 1% false alarms, attack success), device-vs-host agreement, and the
INT8 feature codes for a bit-exact check against the host simulation; it then measures latency and
raises GPIO markers (GPIO4/6/5 -> PPK2 D0/D2/D1) around the inference-only, deployed (cold) and
amplified regions. With a Nordic PPK2: `capture_ppk2.py` then `compute_energy.py`.

## Key measured results (ESP32-S3, INT8, 240 MHz, single core)

Deployed models: static INT8 backdoor, class-representative calibration, seed 0 (`results/deploy_r3*`).

| | VibCNN | WDCNN |
|---|---|---|
| On-device CCM AUROC / TPR@1% (523 clean + 286 triggered) | 0.988 / 0.934 | 0.997 / 0.913 |
| Device == host (FP32 and ESP-PPQ simulation) | 100% (523/523) | 100% (523/523) |
| Feature codes bit-exact with the host simulation | 809/809 | 809/809 |
| Inference latency (single / dual core) | 3669.8 / 3652 µs | 1969.2 / 1974 µs |
| End-to-end, cold (inference + feature read + 1x CCM) | 3728.8 µs | 2043.5 µs |
| CCM overhead, cold (the deployed cost) | 58.9 µs (1.61%) | 74.2 µs (3.77%) |
| CCM per call, hot cache (lower bound only) | 7.75 µs | 7.83 µs |
| Model RAM (internal / PSRAM) | 6.6 / 41.9 KB | 7.3 / 37.8 KB |
| CCM energy, cold (estimate: latency x 463 mW) | ~27 µJ (1.6%) | ~34 µJ (3.8%) |

Board power at 240 MHz: 132.1 mW idle, 463.1 mW during inference (Nordic PPK2, 3.3 V). On the previous
image the PPK2 trace gives a *measured* cold-path monitor energy of 16.9 µJ (0.98%) per inference
(`code/energy_reanalyze_r2.py`); the 3.03 µJ reported in an earlier version was the hot-cache figure.

Neither construction achieves INT8 conditioning for either architecture on any of the three datasets;
the CCM costs a few percent of an inference on the deployed (cold) path, and its detection advantage
over a global statistic depends on the setting and on how the calibration set is drawn (see above).
At a 1% false-alarm budget under a recording-level split it detects 0.19-0.60 of triggered inputs, and
it alarms on operating conditions absent from its calibration data as readily as on a trigger.

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
