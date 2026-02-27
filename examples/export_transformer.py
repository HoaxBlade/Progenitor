#!/usr/bin/env python3
"""Export a transformer encoder model to ONNX for benchmarking.

Creates a 6-layer transformer encoder (BERT-style) with:
  - hidden_size=768, num_heads=12, intermediate_size=3072
  - sequence_length=128, batch_size=1
  - ~44M parameters (similar to DistilBERT)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, num_layers=6,
                 intermediate_size=3072, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, 1000)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        encoded = self.encoder(x)
        # Pool: take the first token
        pooled = encoded[:, 0, :]
        return self.classifier(pooled)


def main():
    out_dir = Path(__file__).parent
    out_path = out_dir / "transformer.onnx"

    model = TransformerEncoder()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer encoder: {total_params:,} parameters")

    dummy = torch.randn(1, 128, 768)

    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
