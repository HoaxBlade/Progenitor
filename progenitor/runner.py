"""Run inference and measure latency, throughput, memory."""

import time
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

    latency_ms = float(np.median(times))
    throughput_per_sec = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    metrics = InferenceMetrics(
        latency_ms=latency_ms,
        throughput_per_sec=throughput_per_sec,
        warmup_runs=warmup,
        timed_runs=repeat,
    )
    if return_raw_times:
        return metrics, times
    return metrics


def create_random_feed(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    """Create a random input feed matching the model's input specs.
    For 4D inputs (NCHW image), unknown dims default to (1, 3, 224, 224) so models like ResNet work."""
    feed = {}
    # Defaults for 4D NCHW (batch, channels, height, width) when dim is dynamic/unknown
    nchw_defaults = (1, 3, 224, 224)
    for inp in session.get_inputs():
        shape = list(inp.shape)
        if len(shape) == 4:
            shape = [
                s if isinstance(s, int) and s > 0 else nchw_defaults[i]
                for i, s in enumerate(shape)
            ]
        else:
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed
