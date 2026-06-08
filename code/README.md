# `code/` — Phase-0 research package

Refactored, runnable implementation of the JSA paper's pipeline: train a deployable 1D-CNN,
inject backdoors (static + quantization-conditioned), run the on-device detector (C3), and export
for ESP-DL. Runs end-to-end **today** on synthetic data; drop in CWRU/Paderborn when ready.

## Environment
```
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate qsentry
# torch needs the OpenMP workaround on this Mac:
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
# env already has: numpy, torch (CPU), onnx (for export)
# real .mat data only: conda install -n qsentry scipy
```

## Run the pipeline
```
python train.py --attack qcb --bits 4        # conditioned backdoor (faithful at INT4)
python train.py --attack static               # conventional backdoor (works at INT8)
python train.py --attack clean                # benign baseline
python evaluate.py --run ../results/phase0     # C3 detector + negative-result diagnostic
python train.py --attack static --onnx         # also export model.onnx for ESP-DL
python train.py --attack pq --bits 8           # faithful INT8 PQ-backdoor attempt (joint STE)

# paper artifacts:
python run_experiments.py --seeds 0,1,2        # results matrix (attacks x bits x seeds), mean+/-std
python export_monitor.py --run ../results/phase0  # -> firmware/main/monitor_params.c (C3 params as C)
```
Swap `--dataset cwru` / `--dataset paderborn` (with `--data-dir ../data`) once real data is in place;
the same commands produce the real-data results.

## Package layout (`tinybd/`)
| Module | Role |
|---|---|
| `data.py` | synthetic generator + CWRU/Paderborn class-folder loaders, `Normalizer` |
| `model.py` | `VibCNN` 1D-CNN with a functional core (FP32 + simulated-INT8 paths) |
| `quant.py` | per-channel weight / per-tensor activation fake-quant, calibration, `make_wq` |
| `backdoor.py` | trigger, poisoning, training, **bin-projection QCB** (scale-pinned, PTQ-faithful), ASR/clean metrics |
| `defense.py` | **C3** class-conditional INT8 activation monitor + detection eval; `tier2_diagnostic` (dropped, negative result) |
| `metrics.py` | AUROC, TPR@FPR |
| `export.py` | ONNX export + ESP-DL conversion procedure |

`train.py` / `evaluate.py` are the drivers; artifacts land in `../results/phase0/`.

## Key correctness notes (validated)
- **Faithful INT8 evaluation**: the deployed model is obtained by *fresh PTQ* of the released FP32
  weights (what ESP-DL does), not by reusing implant-time integers. This honest evaluation revealed
  that the simple bin-projection QCB **does not survive INT8** (narrow bins ⇒ FP32 ≈ INT8) but
  **is faithfully conditioned at INT4** (`ASR_INT8≈0.88 / FP32≈0.00`). A true INT8 QCB needs a
  proper PQ-backdoor construction — a Phase-1 task. See `../docs/validation-findings.md`.
- **C3** detects static and (INT4) conditioned backdoors at AUROC ≈0.94–0.98; the **class-conditional**
  form (R1) clearly beats the global ablation (~0.71) — confirming R1.
- **Tier-2** stays ≈0.5–0.59 → correctly demoted to a measured negative result.

## Next (Phase 1)
Real CWRU/Paderborn data → ESP-DL deployment (`firmware/`) → measured latency/energy/RAM (PPK2);
implement a proper PQ-backdoor for an INT8 QCB; second-MCU cross-check.
