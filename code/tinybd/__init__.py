"""
tinybd -- Backdoor attacks & on-device defense for quantized TinyML on the ESP32-S3.

Phase-0 research package (host side). Refactored from the validated de-risk scripts
(validate_qsentry_cnn.py / _adaptive.py). Modules:

  data      -- dataset loaders: synthetic (works now) + CWRU/Paderborn (drop-in real data)
  model     -- deployable 1D-CNN with a functional core (float + simulated-INT8 paths)
  quant     -- per-channel weight / per-tensor activation fake-quant + calibration
  backdoor  -- static trigger, poisoning, and the bin-projection quantization-conditioned backdoor
  defense   -- class-conditional INT8 activation monitor (C3) + detection evaluation
  metrics   -- AUROC, TPR@FPR
  export    -- ONNX export + ESP-DL conversion procedure

Drivers (in ../code): train.py, evaluate.py.
"""
__all__ = ["data", "model", "quant", "backdoor", "defense", "metrics", "export"]
