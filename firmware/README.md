# Firmware — On-Device Measurement Harness (ESP32-S3)

Skeleton ESP-IDF app that measures the paper's on-device cost (contribution C1): **latency, peak
RAM, and energy** of (1) baseline INT8 inference and (2) inference + the C3 activation monitor.
The delta is the defense's measured overhead.

> Status: **skeleton**. Timing / RAM / GPIO-energy-marker logic is complete; the ESP-DL
> model-load and forward calls are stubbed (`model_init`, `model_infer`). `monitor_params.c` is
> generated. Model conversion is **verified working** (see below).

## VERIFIED so far (host side, no board needed)
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
2. **Generate `monitor_params.h`** from the host detector: dump `g_median[C][D]` and
   `g_inv_mad[C][D]` (= 1/MAD) from `tinybd.defense.calibrate(...)` as C arrays. (A small exporter
   is a TODO — `np.savetxt`-style.)
3. **Implement `model_init` / `model_infer`** in `qsentry_bench.c` against your esp-dl version;
   `model_infer` must return argmax class and write the 64-d penultimate (gap) features.
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
