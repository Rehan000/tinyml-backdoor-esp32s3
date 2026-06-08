"""Export a trained VibCNN to ONNX and document the ESP-DL (ESP32-S3) deployment path.

ONNX export is real and runs here. The ONNX -> ESP-DL step runs in the ESP-IDF toolchain
(esp-ppq / esp-dl) on your machine, not in this package -- see esp_dl_instructions().
"""
import torch


def to_onnx(model, sig_len, path, opset=13):
    """Export the FP32 model to ONNX. Input shape (1,1,sig_len)."""
    model.eval()
    dummy = torch.zeros(1, 1, sig_len)
    # Use the legacy TorchScript exporter (dynamo=False) to avoid the onnxscript dependency
    # that torch>=2.x's default dynamo exporter requires.
    try:
        torch.onnx.export(
            model, dummy, path,
            input_names=["signal"], output_names=["logits"],
            dynamic_axes={"signal": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )
    except TypeError:  # very old torch without the `dynamo` kwarg
        torch.onnx.export(
            model, dummy, path,
            input_names=["signal"], output_names=["logits"],
            dynamic_axes={"signal": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )
    return path


def esp_dl_instructions():
    return """\
ESP-DL deployment (run in your ESP-IDF environment, not here):

1. Install ESP-IDF (>=5.x) and clone esp-dl + esp-ppq (Espressif's PTQ toolkit).
2. Quantize the ONNX to ESP-DL INT8 with esp-ppq, using your PTQ CALIBRATION SET
   (the same clean subset this package saved as calib.npy) so deployed scales match
   the host-side simulation:
       quantize the ONNX -> .espdl model, target = esp32s3, per-channel weights / per-tensor acts.
3. Copy the .espdl model into the firmware project (firmware/main/model/), flash, and run the
   measurement harness (firmware/main/qsentry_bench.c).
4. KEEP host and device aligned: same architecture, same INT8 scheme, same calibration set.
   Validate device top-1 accuracy == host simulated-INT8 accuracy before trusting cost numbers.

Note: exact esp-ppq API names vary by version -- pin versions in firmware/README.md once working.
"""
