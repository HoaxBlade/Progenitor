"""ONNX graph-level optimization passes."""

from onnx import ModelProto, shape_inference
import onnxruntime as ort
from pathlib import Path

from progenitor.config import Target


def apply_shape_inference(model: ModelProto) -> ModelProto:
    """Run ONNX shape inference. Helps runtimes and later passes."""
    return shape_inference.infer_shapes(model)

def apply_onnx_simplifier(model: ModelProto) -> ModelProto:
    """Run onnx-simplifier to fold constants and remove redundant ops."""
    try:
        from onnxsim import simplify
        model_simp, check = simplify(model)
        if check:
            return model_simp
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: onnx-simplifier failed ({e}). Proceeding without simplification.")
    return model

def apply_ort_offline_optimization(model: ModelProto, output_path: str | Path, target: Target) -> None:
    """
    Load the model through ONNX Runtime with ORT_ENABLE_ALL and save the fused graph.
    This bakes in optimizations like Conv+BN fusion into the .onnx file itself.
    """
    import tempfile
    from progenitor.loader import save_onnx
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        save_onnx(model, tmp_path)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.optimized_model_filepath = str(output_path)
        
        # Just creating the session with optimized_model_filepath will save the optimized graph
        _ = ort.InferenceSession(
            tmp_path, 
            sess_options=sess_opts,
            providers=list(target.execution_providers)
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
