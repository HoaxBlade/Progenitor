"""Validation tool to ensure enhanced models yield correct outputs."""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from progenitor.runner import create_random_feed

def validate_accuracy(original_model_path: str | Path, enhanced_model_path: str | Path) -> float:
    """
    Compare outputs of original and enhanced models to measure accuracy degradation.
    Returns Mean Squared Error (MSE). Lower is better.
    """
    sess_orig = ort.InferenceSession(str(original_model_path), providers=["CPUExecutionProvider"])
    sess_enh = ort.InferenceSession(str(enhanced_model_path), providers=["CPUExecutionProvider"])
    
    feed = create_random_feed(sess_orig)
    
    # Run original model
    out_orig = sess_orig.run(None, feed)
    # Run enhanced model
    out_enh = sess_enh.run(None, feed)
    
    # Calculate MSE across all outputs
    total_mse = 0.0
    for orig_arr, enh_arr in zip(out_orig, out_enh):
        mse = np.mean((orig_arr - enh_arr) ** 2)
        total_mse += mse
        
    avg_mse = total_mse / len(out_orig) if out_orig else 0.0
    return float(avg_mse)
