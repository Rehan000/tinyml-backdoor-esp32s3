# Firmware — On-Device Measurement Harness (ESP32-S3)

ESP-IDF app that measures the paper's on-device cost (contribution C1): **latency, peak RAM, and
energy** of (1) baseline INT8 inference and (2) inference + the CCM activation monitor. The delta is
the defense's measured overhead.

> Status: **complete and measured on hardware.** `qsentry_bench.cpp` loads the embedded `.espdl`
> model through `dl::Model`, runs the full measurement protocol (12 rounds x 200 iterations after 20
> warm-up), checks device-vs-host agreement over the embedded test set, and raises GPIO markers for
> the PPK2. Results are in `results/wp7_measurements.md`.

## Measured results (ESP32-S3 N16R8, 240 MHz, single core)

| | VibCNN | WDCNN |
|---|---|---|
| Inference latency | 3677.2 +/- 0.2 us | 2054.4 +/- 1.7 us |
| End-to-end (+ input quant + 1x CCM) | 3716.8 +/- 0.2 us | 2090.0 +/- 1.9 us |
| CCM overhead (cold, single-shot) | 39.6 us (1.08%) | 35.7 us (1.74%) |
| CCM per-call (hot cache, amplified) | 7.57 us | 7.57 us |
| Model RAM (internal / PSRAM) | 6.6 / 41.8 KB | 7.3 / 37.8 KB |
| Device == host | 99.0% (N=512) | 100% (N=128) |

Energy (Nordic PPK2, 3.3 V, 100 ksps): 132.1 mW idle, 463.1 mW during inference, 1.72 mJ per
inference, CCM +3.03 uJ (0.18%).

Two measurement lessons worth repeating: report the **cold** single-shot overhead, because the
amplified hot-cache microbenchmark understates the real per-inference cost about fivefold, and check
fidelity on the **full** test set, because a 64-sample check reported a misleading 100%.

## VERIFIED (host side, no board needed)
- ESP-IDF v5.3.5 installed; `esp-ppq` (in conda env `espdl`) installed.
- `code/convert_espdl.py` converts our trained VibCNN -> **`firmware/main/model/model.espdl`** (28 KB)
  at INT8 for esp32s3 with **<0.04% per-layer quantization error**. Conv1d converts fine — no
  Conv2d reformulation needed.
- `code/export_monitor.py` -> `firmware/main/monitor_params.c` (C3 detector params as C arrays).

## SIMPLIFIED cost-measurement plan (avoids esp-dl intermediate-output extraction)
The paper's on-device contribution is **cost** (latency/energy/RAM), not on-device detection accuracy
(that's a host result, data-dependent, already validated). So we do NOT need esp-dl to expose the
penultimate "gap" features:
- **Inference cost**: load `model.espdl` with `dl::Model`, run it, measure latency/RAM/energy.
- **Monitor (C3) cost**: benchmark `tier1_score()` over a 64-float buffer (the same O(d) compute it
  does in deployment) — its cost is independent of where the features come from.
- **Defended cost = inference + monitor.** Both measured on real silicon; detection AUROC stays the
  host-side number.
This keeps the firmware to: `dl::Model` inference + the standalone monitor micro-benchmark.

## What you need
- ESP-IDF ≥ 5.x, target `esp32s3`.
- `esp-dl` + `esp-ppq` (Espressif). Pin the exact versions here once it builds: `esp-dl __`, `esp-ppq __`.
- Hardware: ESP32-S3 **N16R8**, **Nordic PPK2** (energy), USB-C. Wire `MARK_GPIO` (GPIO4) to a
  PPK2 logic/marker input so energy can be integrated between high/low edges.

## Steps
1. **Export & quantize the model** (host): `python train.py --attack <clean|static|qcb> --onnx`
   then follow `tinybd.export.esp_dl_instructions()` to convert `model.onnx` → `.espdl` with
   `esp-ppq`, using `calib.npy` as the PTQ calibration set (so device scales match the host sim).
   Place the model under `main/model/`.
2. **Generate `monitor_params.c`**: `python code/export_monitor.py --run <run>` dumps
   `g_median[C][D]` and `g_inv_mad[C][D]` (= 1/MAD) as C arrays. Note that these tables depend on the
   calibration draw; use a class-representative one (see the repository README).
3. **Embed the test vectors**: `python code/export_testset.py --run <run> -k 512` writes
   `main/testset.{c,h}` (inputs, labels, and host FP32 predictions) for the device-vs-host check.
4. **Build & flash:** `idf.py set-target esp32s3 && idf.py build flash monitor`.
5. **Read results:** latency + RAM print over serial; **energy** = PPK2 integral between
   `MARK_GPIO` high/low edges, per region. Report avg & worst case.

## Validate before trusting cost numbers
Device top-1 accuracy and ASR must match the host **simulated-INT8** numbers (from `evaluate.py`).
If they diverge, the device quantization scheme ≠ the host simulation — fix before measuring.

## Measurement protocol (for the paper)
- ≥200 inferences after ≥20 warm-up; report mean ± std latency, mean energy/inference (mJ),
  peak RAM (internal + PSRAM).
- Report baseline, defended, and the **delta** (C3 overhead). Sweep gating/layer if applicable.
- Pin: ESP-IDF version, esp-dl/esp-ppq versions, CPU freq (240 MHz), single vs dual core.
