#!/usr/bin/env python3
"""
Download a large ONNX model (ResNet-50, ~100MB) for testing Progenitor on a real large graph.

Uses only the standard library (urllib). Run from repo root. Creates examples/resnet50.onnx.
"""

import sys
import urllib.request
from pathlib import Path


# ResNet-50 ONNX from Hugging Face (public, ~97MB)
RESNET50_URL = "https://huggingface.co/onnx-community/resnet-50-ONNX/resolve/main/onnx/model.onnx"


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "resnet50.onnx"

    if out_path.exists():
        print(f"{out_path} already exists. Delete it to re-download.")
        print("Run: progenitor enhance examples/resnet50.onnx -o examples/resnet50_enhanced.onnx")
        print("Then: python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20")
        return

    print("Downloading ResNet-50 ONNX (~97 MB). This may take a minute...")
    req = urllib.request.Request(RESNET50_URL, headers={"User-Agent": "Progenitor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            data = resp.read()
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Install huggingface_hub and run: python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('onnx-community/resnet-50-ONNX', 'onnx/model.onnx', local_dir='examples', local_dir_use_symlinks=False)\"", file=sys.stderr)
        sys.exit(1)

    out_path.write_bytes(data)
    print(f"Created {out_path}")

    print()
    print("Next commands (large model — use fewer repeats for quicker benchmark):")
    print("  progenitor enhance examples/resnet50.onnx --target cpu -o examples/resnet50_enhanced.onnx")
    print("  python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20")
    print("  python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20 --live")


if __name__ == "__main__":
    main()
