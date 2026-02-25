# Progenitor examples (Phase 1: ML)

## Quick start

1. **Enhance an ONNX model** (CLI):

   ```bash
   progenitor enhance path/to/model.onnx --target cpu --output path/to/model_enhanced.onnx
   ```

2. **In Python:**

   ```python
   from progenitor import enhance

   result = enhance("model.onnx", "cpu", output_path="model_enhanced.onnx")
   if result.compatible:
       print("Enhanced:", result.output_path)
   ```

3. **Benchmark before/after:**

   ```bash
   python benchmarks/run.py path/to/model.onnx --target cpu --repeat 100
   ```

   To see **real-time** updates as each run completes, use `--live` (or `-l`). To confirm numbers are not hardcoded: use `--verbose` for raw timings, or `--live` to stream every run.

## Getting a small ONNX model

If you don't have an ONNX model yet:

- **PyTorch:** `torch.onnx.export(model, dummy_input, "model.onnx")`
- **Hugging Face:** Many models have ONNX weights or you can export with `optimum` (e.g. `optimum-cli export onnx --model ...`).
- **Samples:** You can create a minimal ONNX for testing with the script below (run from repo root):

```python
# examples/create_tiny_onnx.py
import numpy as np
from onnx import helper, TensorProto, save
from onnx import numpy_helper

# Minimal add/matmul graph
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
node = helper.make_node("Add", ["x", "y"], ["z"])
graph = helper.make_graph([node], "tiny", [X, Y], [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
save(model, "tiny.onnx")
```

Then: `progenitor enhance tiny.onnx -o tiny_enhanced.onnx` and `python benchmarks/run.py tiny.onnx`.

### Testing on a larger model (small MLP)

To test Progenitor on a model with more than one op (so optimization can actually help):

```bash
python examples/export_small_model.py
progenitor enhance examples/small_mlp.onnx -o examples/small_mlp_enhanced.onnx
python benchmarks/run.py examples/small_mlp.onnx --target cpu --repeat 50
```

See the main [README](../README.md) for the full command list.

### Large model (ResNet-50)

To test on a **large** ONNX model (~97 MB, real conv/BatchNorm graph):

```bash
python examples/download_large_model.py
progenitor enhance examples/resnet50.onnx -o examples/resnet50_enhanced.onnx
python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20
```

**Why does tiny.onnx sometimes show *slower* "After"?**  
`tiny.onnx` is a single Add op. Full graph optimization (fusion, constant folding, etc.) adds a small fixed overhead. For one op that overhead can outweigh any benefit, so "After" may be slower. Progenitor is meant for **larger** models (real networks with many layers), where those optimizations typically give a clear speedup.
