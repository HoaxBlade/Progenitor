# Phase 1: Progenitor compatibility spec (ML)

Progenitor enhances only **compatible** model + target pairs. This document defines what is supported in Phase 1 and lists all optimization passes.

---

## Model format

| Format   | Extension | Status    | Notes                          |
|----------|-----------|-----------|--------------------------------|
| **ONNX** | `.onnx`   | Supported | Primary format for Phase 1.   |

- **ONNX opset:** 11–17 (tested). Newer opsets may work but are not guaranteed.
- **Inputs:** Models must have clearly defined input names and shapes (static or dynamic).
- **Unsupported for now:** Sparse tensors as input format, custom ops without ONNX Runtime implementation, training graphs.

---

## Hardware targets

| Target   | ID    | Status   | Notes                                      |
|----------|-------|----------|--------------------------------------------|
| **CPU**  | `cpu` | Supported | x86_64 and ARM64. Default execution provider. |
| **CUDA** | `cuda`| Supported | GPU inference. Requires `onnxruntime-gpu`.   |

- **CPU:** Uses `CPUExecutionProvider`. No GPU required.
- **CUDA:** Uses `CUDAExecutionProvider` with CPU fallback. Install `onnxruntime-gpu`; same enhance pipeline, benchmark and validation run on GPU when `--target cuda`.

---

## Optimization passes (Phase 1)

Passes are applied in a fixed order. Architecture is **auto-detected** and only beneficial passes run per model type:

- **CNN:** Conv-heavy (more Conv than MatMul/Gemm, ≥3 convs). ResNet-style.
- **Diffusion:** Conv-heavy + attention (Softmax/LayerNorm) + ≥4 linear ops; e.g. U-Net + cross-attn.
- **Transformer:** Not conv-heavy, has Softmax/LayerNorm, ≥6 linear ops.
- **RNN:** At least one LSTM or GRU op.
- **GNN:** At least two Gather/Scatter ops and ≥4 linear ops (message-passing style).
- **MLP:** None of the above; large MLP if param count >100k.

### Always-on

| Pass               | Effect                                                                 |
|--------------------|------------------------------------------------------------------------|
| **Shape inference**| Infer and propagate shapes for downstream passes.                     |
| **ONNX simplifier**| Redundant node removal, constant folding where applicable.            |
| **Graph optimization** | When no other passes run: ORT offline optimization (fusion, layout). |

### Opt-in (explicit flags or `--max-speed`)

| Pass | Flag / trigger | Effect | Model types |
|------|----------------|--------|-------------|
| **Conv channel pruning** | `--conv-prune RATIO` or CNN `--max-speed` | Remove fraction of Conv bottleneck channels; physically shrinks Conv weights. | CNN (ResNet, VGG, etc.) |
| **Block removal** | CNN `--max-speed` only (validation-guided) | Remove least-important residual blocks (identity branches); replaces block with skip. | CNN (ResNet-style) |
| **Structured pruning** | `--struct-prune RATIO` or MLP/transformer `--max-speed` | Remove entire neurons / filters; shrinks MatMul/Gemm/Conv dimensions. | MLP, transformer, CNN (struct path) |
| **Low-rank (SVD)** | `--lowrank RATIO` or MLP/transformer `--max-speed` | Replace weight matrices with low-rank factor; keeps top singular values. | MLP, transformer, CNN |
| **Unstructured (magnitude) pruning** | `--prune SPARSITY` or MLP/RNN/GNN `--max-speed` | Zero out fraction of weights by magnitude; same graph, sparse weights. Supports Conv, MatMul, Gemm, **LSTM, GRU**. Use sparse backend for 5–15×. | MLP, transformer, RNN, GNN, CNN (with conv) |
| **Output calibration** | `--calibrate` or default for CNN/transformer `--max-speed` | Fit per-output scale/bias so pruned/optimized output matches original; preserves cosine. | Any (except large MLP 123× path) |
| **Dynamic INT8 quantization** | `--quantize` | Weights and activations quantized to INT8 at runtime; 2–4× on CPU typical. | Any |
| **Static INT8 quantization** | `--static-quantize` or CNN `--max-speed` | Calibrated INT8; often faster than dynamic on CPU. Chained after conv prune for CNN. | Any |

### Max-speed behavior (architecture-specific)

| Architecture | What `--max-speed` does | Typical result |
|--------------|-------------------------|----------------|
| **CNN** | Validation-guided conv channel prune (max ratio that preserves cosine) → validation-guided block removal → output calibration → static INT8. | ~6–9× speedup, cosine preserved (~0.99). |
| **Large MLP** (>100k params) | Structured prune 0.75 + low-rank 0.1 + unstructured prune 0.99 (magnitude only). No block size, no per-layer tune. | ~100–125× with sparse backend; cosine not calibrated. |
| **Small MLP** | Per-layer tune or single sparsity sweep for cosine ≥ 0.9; optional block sparse (4,4). With `--max-speed-aggressive`: struct 0.5 + lowrank 0.2 + prune 0.9 + calibrate. | ~3–7× (dense run) or 5–15× with sparse backend. |
| **Transformer** | Structured prune 0.25 + low-rank 0.4; optional prune 0.9 if aggressive. Calibration on. | ~2×+ and improved cosine. |
| **RNN** (LSTM/GRU) | Structured prune 0.2 + low-rank 0.35 + magnitude prune 0.85 on linear and LSTM/GRU weights. Calibration on. | *Tested (small LSTM):* ~2.2× speedup, cosine 0.99. Run with `--validate` on your model. |
| **GNN** | No struct (graph has Gather/Scatter); low-rank 0.35 + magnitude prune 0.85. Calibration on. | *Tested (examples/gnn_like.onnx):* ~2.3× speedup; cosine can drop (e.g. 0.72). Run with `--validate`. |
| **Diffusion** (conv + attention) | Low-rank 0.5 + magnitude prune 0.75 + output calibration. No INT8 (static QDQ/dynamic quant issues). Saves FP32 optimized. | *Tested:* ~3× speedup. **Known bug:** reported cosine stays ~0.098 in validation; likely bug in calibration/validation path for diffusion — to be fixed. |

### High cosine / advanced (optional flags)

| Option | Effect |
|--------|--------|
| `--per-layer-tune` | Tune sparsity per layer (small MLP) to keep cosine high at same speedup. |
| `--block-size H,W` | Block-sparse pruning (e.g. 4,4) for 2D weights. |
| `--calibrate` | Post-prune output calibration to recover cosine. |
| `--progressive-steps 0.5,0.7,0.9` | Progressive pruning then calibrate once at end. |
| `--sparse-pattern 2:4` | 2:4 sparsity (2 non-zeros per 4) for hardware-friendly layout. |
| `--max-speed-aggressive` | Small MLP / transformer: stronger struct+lowrank+prune then calibrate. |

---

## Accuracy and validation

- **Quantization and pruning** can change outputs. User is responsible for validating on real data for production.
- **Benchmark:** Use `--validate` to compare baseline vs enhanced model (cosine similarity, MSE, top-1 match) on random feeds:
  ```bash
  python benchmarks/run.py path/to/model.onnx --max-speed --validate
  ```
- **CNN max-speed:** Validation-guided conv and block removal keep intermediate cosine above a threshold; output calibration restores final cosine so it is not compromised.

### Known issues (to fix)

- **Diffusion cosine:** For diffusion-style models, the reported cosine similarity in `--validate` stays fixed at ~0.098 and does not reflect calibration. This is a bug (either in validation, in how the calibrated model is saved/loaded, or in the calibration path for diffusion). Speedup (~3×) is correct; cosine reporting must be fixed separately.

---

## Runtimes and sparse backends

- **Inference:** ONNX Runtime only. Other runtimes (TensorRT, OpenVINO) may be added as optional backends later.
- **Execution providers:** `CPUExecutionProvider` (default), `CUDAExecutionProvider` (use `--target cuda` with `onnxruntime-gpu`).
- **Sparse inference:** Unstructured pruned models get full speedup (5–15× or 100×+ for large MLP) only with a sparse-capable backend (e.g. native sparse, MKL, or INT8-sparse where available). The benchmark script uses sparse backends when present; otherwise it runs pruned weights as dense (smaller file, accuracy check).

---

## Compatibility checklist

A model is **Progenitor-compatible** for Phase 1 if:

1. It is a valid ONNX file (`.onnx`).
2. It loads without error in ONNX Runtime with the chosen target (e.g. CPU).
3. It uses only ops supported by ONNX Runtime for that target.
4. Input shapes are known (fixed or symbolic with known rank).

If any of the above fails, Progenitor reports **incompatible** and does not produce an enhanced artifact.

---

## Version

- **Spec version:** 1.1  
- **Phase:** 1 (ML algorithms)  
- **Last updated:** 2026-02  
