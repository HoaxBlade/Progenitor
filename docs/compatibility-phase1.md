# Phase 1: Progenitor compatibility spec (ML)

Progenitor enhances only **compatible** model + target pairs. This document defines what is supported in Phase 1.

---

## Model format

| Format   | Extension | Status   | Notes                          |
|----------|-----------|----------|--------------------------------|
| **ONNX** | `.onnx`   | Supported | Primary format for Phase 1.   |

- **ONNX opset:** 11–17 (tested). Newer opsets may work but are not guaranteed.
- **Inputs:** Models must have clearly defined input names and shapes (static or dynamic).
- **Unsupported for now:** Sparse tensors, custom ops without ONNX Runtime implementation, training graphs.

---

## Hardware targets

| Target   | ID    | Status   | Notes                                      |
|----------|-------|----------|--------------------------------------------|
| **CPU**  | `cpu` | Supported | x86_64 and ARM64. Default execution provider. |
| **CUDA** | `cuda`| Planned  | Phase 1.1 after CPU is stable.             |

- **CPU:** Uses `CPUExecutionProvider`. No GPU required.
- **CUDA:** Will use `CUDAExecutionProvider` when added; requires `onnxruntime-gpu`.

---

## Optimization passes (Phase 1)

| Pass               | Applied when     | Effect                          |
|--------------------|------------------|---------------------------------|
| Graph optimization | Always           | Constant folding, redundant node removal, layout optimizations (via ONNX Runtime). |
| Session options    | Always           | Optimized execution provider options for the target. |
| Quantization       | Opt-in (`--quantize`) | Dynamic INT8 (CPU) for supported ops; may reduce accuracy. |

- Passes are applied in order. Later phases may add pruning, fusion, and custom kernels.
- **Quantization:** Only applied if explicitly requested; user is responsible for accuracy validation.

---

## Runtimes

- **Inference:** ONNX Runtime only. Other runtimes (TensorRT, OpenVINO) may be added as optional backends later.
- **Execution providers (CPU):** `CPUExecutionProvider` (default).

---

## Compatibility checklist

A model is **Progenitor-compatible** for Phase 1 if:

1. It is a valid ONNX file (`.onnx`).
2. It loads without error in ONNX Runtime with the chosen target (e.g. CPU).
3. It uses only ops supported by ONNX Runtime for that target.
4. Input shapes are known (fixed or symbolic with known rank).

If any of the above fails, Progenitor will report **incompatible** and will not produce an enhanced artifact.

---

## Version

- **Spec version:** 1.0  
- **Phase:** 1 (ML algorithms)  
- **Last updated:** 2026-02
