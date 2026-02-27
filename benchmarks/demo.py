#!/usr/bin/env python3
"""
Progenitor Demo — Auto-benchmark all models × optimization combos.

Produces:
  1. Formatted console comparison table
  2. visualizations (saved as PNG):
     - Speedup bar chart (grouped by model)
     - Accuracy vs Speed Pareto frontier
     - Optimization breakdown heatmap
  3. Machine-readable results JSON
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from progenitor.loader import load_onnx
import numpy as np
import onnxruntime as ort

from progenitor.api import enhance
from progenitor.runner import create_random_feed, run_metrics
from progenitor.validate import validate_accuracy
from progenitor.backends.accelerate_sparse_native import native_sparse_available, NativeSparseSession


# ── Configuration ──────────────────────────────────────────────────

MODELS = {
    "small_mlp": "examples/small_mlp.onnx",
    "resnet50": "examples/resnet50.onnx",
}

COMBOS = {
    "baseline":         {},
    "conv-prune 30%":   {"conv_prune": 0.3},
    "struct-prune 50%": {"struct_prune": 0.5},
    "static INT8":      {"static_quantize": True},
    "max-speed":        {"max_speed": True},
}

WARMUP = 10
REPEAT = 20
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_benchmark(model_path: str, combo_kwargs: dict) -> dict:
    """Run a single benchmark: enhance + measure."""
    model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    # Baseline latency
    try:
        sess_base = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        feed = create_random_feed(sess_base)
        base_result = run_metrics(model_path, feed, warmup=WARMUP, repeat=REPEAT)
        baseline_ms = base_result.latency_ms
    except Exception as e:
        return {"error": f"Baseline failed: {e}"}

    if not combo_kwargs:
        return {
            "latency_ms": baseline_ms,
            "speedup": 1.0,
            "cosine": 1.0,
            "top1_match": True,
            "mse": 0.0,
        }

    # Enhanced
    try:
        result = enhance(str(model_path), "cpu", **combo_kwargs)
        if not result.compatible:
            return {"error": result.message}

        # Check for native sparse backend for MLP unstructured pruning
        use_native_sparse = False
        if combo_kwargs.get("prune") is not None or combo_kwargs.get("max_speed"):
            # sparse natively supports basic MLP ops
            model_tmp = ort.InferenceSession(str(result.output_path), providers=["CPUExecutionProvider"])
            op_types = {n.op_type for n in getattr(load_onnx(result.output_path).graph, 'node', [])}
            if native_sparse_available() and not any(t in op_types for t in ["Conv", "LSTM"]):
                use_native_sparse = True
            
        if use_native_sparse:
            engine = NativeSparseSession(str(result.output_path))
            feed_vals = {name: val for name, val in feed.items()}
            # wrapper to match run_metrics signature expect
            class DummySess:
                def run(self, _, f):
                    return engine.run(f)
            sess_enh = DummySess()
        else:
            sess_enh = ort.InferenceSession(
                str(result.output_path), providers=["CPUExecutionProvider"]
            )
            
        enh_result = run_metrics(result.output_path if not use_native_sparse else sess_enh, feed, warmup=WARMUP, repeat=REPEAT)
        enh_ms = enh_result.latency_ms

        metrics = validate_accuracy(model_path, result.output_path)

        return {
            "latency_ms": enh_ms,
            "speedup": baseline_ms / enh_ms if enh_ms > 0 else 0,
            "cosine": metrics["cosine_similarity"],
            "top1_match": metrics["top1_match"],
            "mse": metrics["mse"],
            "message": result.message,
        }
    except Exception as e:
        return {"error": str(e)}


def print_table(results: dict):
    """Print a formatted comparison table."""
    print()
    print("=" * 90)
    print("PROGENITOR BENCHMARK RESULTS")
    print("=" * 90)

    for model_name, combos in results.items():
        print(f"\n{'─' * 90}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 90}")
        print(f"  {'Optimization':<22} {'Latency':>10} {'Speedup':>8} {'Cosine':>8} {'Top-1':>6}")
        print(f"  {'─' * 60}")

        for combo_name, data in combos.items():
            if "error" in data:
                print(f"  {combo_name:<22} {'ERROR':>10}   {data['error'][:40]}")
                continue
            lat = f"{data['latency_ms']:.3f} ms"
            spd = f"{data['speedup']:.2f}x"
            cos = f"{data['cosine']:.4f}"
            top1 = "Yes" if data.get("top1_match") else "No"
            print(f"  {combo_name:<22} {lat:>10} {spd:>8} {cos:>8} {top1:>6}")

    print(f"\n{'=' * 90}")


def create_visualizations(results: dict, output_dir: Path):
    """Generate research-grade visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Note: matplotlib not installed. Skipping visualizations.")
        print("  Install with: pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette (professional, colorblind-friendly)
    colors = {
        "baseline": "#4A90D9",
        "conv-prune 30%": "#E8913A",
        "struct-prune 50%": "#67B279",
        "static INT8": "#D94A6B",
        "max-speed": "#9B59B6",
    }

    # ── Figure 1: Speedup Comparison Bar Chart ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Progenitor Optimization Speedup Comparison",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, (model_name, combos) in enumerate(results.items()):
        ax = axes[idx]
        names = []
        speedups = []
        bar_colors = []

        for combo_name, data in combos.items():
            if "error" not in data:
                names.append(combo_name.replace(" ", "\n"))
                speedups.append(data["speedup"])
                bar_colors.append(colors.get(combo_name, "#888"))

        bars = ax.bar(names, speedups, color=bar_colors, edgecolor="white",
                      linewidth=0.5, width=0.6)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Speedup (×)" if idx == 0 else "")
        ax.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(0, max(speedups) * 1.2 if speedups else 2)

        # Value labels on bars
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}×", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "speedup_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'speedup_comparison.png'}")

    # ── Figure 2: Accuracy vs Speed Pareto Frontier ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Accuracy–Speed Trade-off (Pareto Frontier)",
                 fontsize=13, fontweight="bold")

    markers = {"small_mlp": "o", "resnet50": "s"}
    for model_name, combos in results.items():
        for combo_name, data in combos.items():
            if "error" in data:
                continue
            ax.scatter(data["speedup"], data["cosine"],
                       c=colors.get(combo_name, "#888"),
                       marker=markers.get(model_name, "o"),
                       s=120, edgecolors="white", linewidth=0.8, zorder=3)
            ax.annotate(f"{combo_name}\n({model_name})",
                        (data["speedup"], data["cosine"]),
                        fontsize=7, ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

    ax.set_xlabel("Speedup (×)", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.axhline(y=0.9, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.6,
               label="Quality threshold (cos > 0.9)")
    ax.legend(fontsize=9, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    fig.savefig(output_dir / "pareto_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'pareto_frontier.png'}")

    # ── Figure 3: Heatmap (Speedup × Model × Combo) ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Optimization Heatmap — Speedup by Model × Method",
                 fontsize=13, fontweight="bold")

    model_names = list(results.keys())
    combo_names = list(next(iter(results.values())).keys())
    heatmap_data = []

    for model_name in model_names:
        row = []
        for combo_name in combo_names:
            data = results[model_name].get(combo_name, {})
            row.append(data.get("speedup", 0))
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(combo_names)))
    ax.set_xticklabels(combo_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(combo_names)):
            val = heatmap_data[i, j]
            color = "white" if val > heatmap_data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}×", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, label="Speedup (×)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_dir / "optimization_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'optimization_heatmap.png'}")


def main():
    print("Progenitor Demo — Running all benchmarks...")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Combos: {list(COMBOS.keys())}")
    print(f"Warmup: {WARMUP}  Repeat: {REPEAT}")
    print()

    results = {}
    total = len(MODELS) * len(COMBOS)
    done = 0

    for model_name, model_path in MODELS.items():
        results[model_name] = {}
        for combo_name, combo_kwargs in COMBOS.items():
            done += 1
            print(f"  [{done}/{total}] {model_name} × {combo_name}...", end=" ", flush=True)
            t0 = time.time()
            data = run_benchmark(model_path, combo_kwargs)
            elapsed = time.time() - t0
            if "error" in data:
                print(f"ERROR ({elapsed:.1f}s): {data['error'][:50]}")
            else:
                print(f"{data['speedup']:.2f}x  cos={data['cosine']:.4f}  ({elapsed:.1f}s)")
            results[model_name][combo_name] = data

    # Print table
    print_table(results)

    # Save JSON
    out_dir = PROJECT_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, out_dir)
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
