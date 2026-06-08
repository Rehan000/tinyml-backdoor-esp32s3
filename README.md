# On-Device Backdoor Attacks and Defense for Quantized TinyML on the ESP32-S3

Code, conversion pipeline, and firmware harness for the paper:

> **On-Device Backdoor Attacks and Defense for Quantized TinyML on Microcontrollers: A Measured Study on the ESP32-S3**
> Muhammad Rehan, Muhammad Ali Munir, Haider Ali. *(Under review, Journal of Systems Architecture.)*

This repository reproduces, end to end, a **measured, on-device** study of backdoor attacks and a
resource-aware defense for INT8-quantized TinyML on a commodity microcontroller (Espressif ESP32-S3,
ESP-DL pipeline), on the CWRU and MFPT bearing-fault benchmarks.

## What's here

- **Quantization-conditioned backdoors (QCBs).** Two faithful constructions — a scale-pinned
  bin-projection and a joint post-training-quantization optimization with a straight-through estimator —
  used to characterize a **bit-width threshold**: conditioning is reproducible at INT4 but **not**
  achievable at INT8 on shallow TinyML CNNs.
- **CCM, a class-conditional INT8 activation monitor** that runs on-device over already-resident
  features to flag triggered inputs.
- **On-device measurement harness** (latency, peak RAM, energy) on real silicon, including the
  defense's per-inference overhead.
- **Generalization** across two architectures (`VibCNN`, `WDCNN`) and two datasets (CWRU, MFPT).

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
python export_testset.py --run ../results/deploy -k 64
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

## Key measured results (ESP32-S3, INT8)

| | VibCNN | WDCNN |
|---|---|---|
| Inference latency | ~5.1 ms | ~2.8 ms |
| Peak RAM | ~49 KB | ~45 KB |
| CCM latency overhead | 11.5 µs (0.22%) | 11.5 µs (0.41%) |
| CCM energy overhead | 2.15 µJ (0.19%) | (identical, 64-d) |
| Device == host (N=64) | 100% | 100% |

INT8 conditioning is not achievable for either architecture on either dataset; the CCM is a near-free
on-device monitor whose detection advantage over a global statistic is setting-dependent.

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
