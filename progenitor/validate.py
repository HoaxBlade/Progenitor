"""Validation tool to ensure enhanced models yield correct outputs."""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from progenitor.runner import create_random_feed

def validate_accuracy(
    original_model_path: str | Path,
    enhanced_model_path: str | Path,
    *,
    seed: int = 42,
    execution_providers: tuple[str, ...] = ("CPUExecutionProvider",),
) -> dict:
    """
    Compare outputs of original and enhanced models to measure accuracy degradation.
    Returns dict with 'mse', 'cosine_similarity', and 'top1_match'.
    Use execution_providers to run on the same target as the benchmark (e.g. CUDA).
    """
    if seed is not None:
        np.random.seed(seed)
    providers = list(execution_providers)
    sess_orig = ort.InferenceSession(str(original_model_path), providers=providers)
    sess_enh = ort.InferenceSession(str(enhanced_model_path), providers=providers)
    
    feed = create_random_feed(sess_orig)
    
    out_orig = sess_orig.run(None, feed)
    out_enh = sess_enh.run(None, feed)
    
    total_mse = 0.0
    total_cosine = 0.0
    top1_match = True
    n = len(out_orig)
    
    for orig_arr, enh_arr in zip(out_orig, out_enh):
        o_flat = orig_arr.flatten().astype(np.float64)
        e_flat = enh_arr.flatten().astype(np.float64)
        
        # Raw MSE
        total_mse += float(np.mean((o_flat - e_flat) ** 2))
        
        # Cosine similarity (1.0 = identical direction, 0.0 = orthogonal)
        norm_o = np.linalg.norm(o_flat)
        norm_e = np.linalg.norm(e_flat)
        if norm_o > 0 and norm_e > 0:
            total_cosine += float(np.dot(o_flat, e_flat) / (norm_o * norm_e))
        
        # Top-1 match (do both models agree on the highest-scoring class?)
        if o_flat.size > 1:
            if np.argmax(o_flat) != np.argmax(e_flat):
                top1_match = False
    
    return {
        'mse': total_mse / n if n else 0.0,
        'cosine_similarity': total_cosine / n if n else 0.0,
        'top1_match': top1_match,
    }
