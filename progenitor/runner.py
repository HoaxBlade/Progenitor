"""Run inference and measure latency, throughput, memory."""

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

@dataclass
class InferenceMetrics:
    """Metrics from a single inference run."""

    latency_ms: float
    throughput_per_sec: float
    peak_memory_mb: float
    warmup_runs: int
    timed_runs: int


def run_metrics(
    model_path: str | Path,
    input_feed: dict[str, np.ndarray],
    execution_providers: tuple[str, ...] = ("CPUExecutionProvider",),
    graph_optimization_level: int = ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    warmup: int = 10,
    repeat: int = 100,
    return_raw_times: bool = False,
    on_sample: Callable[[int, float], None] | None = None,
) -> InferenceMetrics | tuple[InferenceMetrics, list[float]]:
    """
    Run inference and return latency (median) and throughput.
    Use graph_optimization_level=ORT_DISABLE_ALL for baseline, ORT_ENABLE_ALL for enhanced.
    If on_sample is set, it is called as on_sample(run_index, time_ms) after each timed run (real-time streaming).
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = graph_optimization_level

    tracemalloc.start()

    session = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=list(execution_providers),
    )

    # Warmup
    for _ in range(warmup):
        session.run(None, input_feed)

    # Timed runs
    times: list[float] = []
    for i in range(repeat):
        start = time.perf_counter()
        session.run(None, input_feed)
        end = time.perf_counter()
        t_ms = (end - start) * 1000.0
        times.append(t_ms)
        if on_sample is not None:
            on_sample(i + 1, t_ms)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / (1024 * 1024)

    latency_ms = float(np.median(times))
    throughput_per_sec = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    metrics = InferenceMetrics(
        latency_ms=latency_ms,
        throughput_per_sec=throughput_per_sec,
        peak_memory_mb=peak_memory_mb,
        warmup_runs=warmup,
        timed_runs=repeat,
    )
    if return_raw_times:
        return metrics, times
    return metrics


def create_random_feed(session: ort.InferenceSession, dynamic_dim_default: int = 1) -> dict[str, np.ndarray]:
    """Create a strictly typed random input feed matching the model's exact input specs.
    Handles different data types (float, int, bool) and dynamic dimensions."""
    feed = {}
    
    # Map ONNX tensor types to numpy dtypes
    # ORT uses strings like 'tensor(float)' in input.type
    type_map = {
        'tensor(float)': np.float32,
        'tensor(float16)': np.float16,
        'tensor(double)': np.float64,
        'tensor(int8)': np.int8,
        'tensor(uint8)': np.uint8,
        'tensor(int16)': np.int16,
        'tensor(uint16)': np.uint16,
        'tensor(int32)': np.int32,
        'tensor(uint32)': np.uint32,
        'tensor(int64)': np.int64,
        'tensor(uint64)': np.uint64,
        'tensor(bool)': np.bool_,
    }
    
    # Vision models often use dynamic (batch, channels, H, W). 
    # If a model specifically uses dynamic 4D inputs, default to 224x224 RGB images.
    nchw_defaults = (1, 3, 224, 224)
    
    for inp in session.get_inputs():
        # Resolve shapes: replace any None/string dynamic dims
        shape = list(inp.shape)
        if len(shape) == 4 and any(not isinstance(s, int) or s <= 0 for s in shape):
             shape = [
                 s if isinstance(s, int) and s > 0 else nchw_defaults[i]
                 for i, s in enumerate(shape)
             ]
        else:
             shape = [s if isinstance(s, int) and s > 0 else dynamic_dim_default for s in shape]
             
        dtype = type_map.get(inp.type, np.float32)

        if dtype in (np.float32, np.float16, np.float64):
            data = np.random.randn(*shape).astype(dtype)
        elif dtype in (np.int8, np.int16, np.int32, np.int64):
            # For integers (often indices for NLP embeddings like Transformers), 
            # pick small recognizable positive ints to avoid index-out-of-bounds in embeddings
            data = np.random.randint(1, 100, size=shape, dtype=dtype)
        elif dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            data = np.random.randint(0, 100, size=shape, dtype=dtype)
        elif dtype == np.bool_:
            data = np.random.choice([True, False], size=shape)
        else:
            # Fallback
            data = np.random.randn(*shape).astype(np.float32)
            
        feed[inp.name] = data
        
    return feed
