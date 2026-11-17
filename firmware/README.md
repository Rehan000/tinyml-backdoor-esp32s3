# Firmware — On-Device Measurement Harness (ESP32-S3)

ESP-IDF app that measures, on real silicon, both **detection** and **cost** of the CCM monitor on the
deployed INT8 model. This directory holds exactly the image that the paper's on-device numbers were
measured on: `main/model/model.espdl`, `main/monitor_params.{c,h}`, `main/testset.{c,h}`,
`main/testset_x.bin`, all generated from `results/deploy_r3` by the scripts below.

Pinned versions: **ESP-IDF v5.3.5**, **esp-dl 3.3.5** (`main/idf_component.yml`, `dependencies.lock`),
**esp-ppq 1.3.4** (conda env `espdl`). Board: ESP32-S3 **N16R8** at 240 MHz, single core.

## What the harness does (`main/qsentry_bench.cpp`)

**Phase 1 — on-device detection and fidelity, full test split.** The model is exported with two
outputs, the logits and the 64-d pooled features (`convert_espdl.py --with-features`), so the monitor
reads the real features of every inference, dequantized from INT8, exactly as in deployment. For each
of the 523 CWRU test windows the device runs the clean input and, for non-target windows, the triggered
input (`x + g_trig`, formed on the device). It computes on the device: accuracy, agreement with host FP32
and with the esp-ppq simulation of the same graph, attack success rate, AUROC, TPR at 1% / 5% false
alarms, and the benign alarm / detection rate at the calibration-fixed threshold; and it prints one
record per inference (plus the INT8 feature codes) for `code/ondevice_detection.py`.

**Phase 2 — cost on the same image.** Per round (12 rounds), three regions separated by idle gaps:

| Region | GPIO → PPK2 | Work | Iterations |
|---|---|---|---|
| A | GPIO4 → D0 | inference | 1000 |
| E2E | GPIO6 → D2 | inference + feature read + 1× CCM (**deployed path**) | 1000 |
| B | GPIO5 → D1 | inference + 1024× CCM (hot cache, lower bound) | 200 |

`(E2E − A)/1000` is the **cold-path** monitor cost per inference, in both time and energy — the figure
the paper now reports. Region B's amplified hot-cache figure is kept only as a lower bound.

## Model and monitor provenance (`results/deploy_r3`)

VibCNN on CWRU, **static INT8 backdoor** (target = normal, 15% poisoning, burst trigger), seed 0,
**class-representative calibration draw** (512 windows; predicted-class counts 76/95/240/101). The monitor
tables are fitted on the calibration features of the deployed graph as simulated by esp-ppq
(`export_monitor.py --from-sim`), i.e. on the same power-of-two INT8 arithmetic the device runs.
The previous revision's image (a clean model quantized with the class-ordered first-512 draw) is archived
in `results/deploy_wp7/firmware_snapshot/`.

## Reproduce

```bash
# 1) host: train, convert (two outputs) + simulate, export monitor tables and the full test split
cd code
conda activate qsentry
python train.py --dataset cwru --model vibcnn --attack static --target-class 2 --seed 0 --out ../results/deploy_r3
conda activate espdl
python convert_espdl.py --run ../results/deploy_r3 --target esp32s3 --with-features --simulate
conda activate qsentry
python export_monitor.py --run ../results/deploy_r3 --from-sim
python export_testset.py --run ../results/deploy_r3          # all 523 windows

# 2) build + flash (USB power), then capture the console log until the harness prints DONE (~4 min)
cd ../firmware && . ~/esp/esp-idf/export.sh
idf.py build && idf.py -p <PORT> flash
python ../code/capture_serial.py --port <PORT> --out ../results/deploy_r3/device_log.txt

# 3) on-device detection + fidelity report (metrics from the DEVICE's scores)
cd ../code && conda activate qsentry
python ondevice_detection.py --run ../results/deploy_r3

# 4) energy: power the board from the PPK2 (VOUT -> 3V3, GND -> GND; USB data only or unplugged),
#    wire GPIO4 -> D0, GPIO5 -> D1, GPIO6 -> D2 (+ GND to the logic GND), then
python capture_ppk2.py --out ../results/energy/ppk2_r3.csv --voltage 3300 --seconds 240
python compute_energy.py --csv ../results/energy/ppk2_r3.csv --voltage 3.3 --json ../results/energy/energy_r3.json
```

`ondevice_detection.py` also checks the device against the esp-ppq simulation element by element: the
feature codes should be bit-exact and the scores equal to float rounding. If they are not, the device
arithmetic differs from the simulation and the discrepancy is reported, not hidden.

## Measurement lessons (kept from the previous revision)

Report the **cold** single-shot overhead, because an amplified hot-cache microbenchmark understates the
real per-inference cost about fivefold; and check fidelity on the **full** test set, because a 64-sample
check reported a misleading 100%.
