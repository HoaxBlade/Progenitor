#!/usr/bin/env python3
"""Export a small LSTM model to ONNX for testing Progenitor RNN path.

Run from repo root. Creates examples/rnn_lstm.onnx.
Requires: pip install torch
"""

import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch required: pip install torch", file=sys.stderr)
    sys.exit(1)


class SmallLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)


def main():
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "rnn_lstm.onnx"

    model = SmallLSTM(input_size=32, hidden_size=64, num_layers=2, num_classes=10)
    model.eval()

    batch, seq_len, input_size = 1, 16, 32
    dummy = torch.randn(batch, seq_len, input_size)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=14,
    )
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print("Run: progenitor enhance examples/rnn_lstm.onnx --target cpu --max-speed")
    print("Then: python benchmarks/run.py examples/rnn_lstm.onnx --target cpu --max-speed --validate --repeat 20")


if __name__ == "__main__":
    main()
