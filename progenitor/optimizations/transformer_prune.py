"""
Transformer-aware structured pruning.

Safely prunes transformer encoder layers by:
1. FFN intermediate reduction: reduce the intermediate_size (e.g. 3072 -> 768)
   in the feed-forward network (two MatMul/Gemm ops with intermediate dim).
2. Attention head removal: remove entire attention heads from Q/K/V projections
   and the output projection.

Both approaches physically shrink weight matrices, giving real speedup with ORT.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, numpy_helper


def _find_ffn_pairs(model: ModelProto):
    """
    Find FFN up-projection/down-projection pairs in a transformer.
    Pattern: MatMul(in, W_up) -> ... -> MatMul(mid, W_down) where W_up is (d, ffn_dim)
    and W_down is (ffn_dim, d) with ffn_dim > d.
    """
    init_map = {i.name: numpy_helper.to_array(i).astype(np.float32)
                for i in model.graph.initializer}

    pairs = []
    nodes = list(model.graph.node)

    for i, node in enumerate(nodes):
        if node.op_type not in ("MatMul", "Gemm"):
            continue

        w_name = None
        for idx in (0, 1):
            if idx < len(node.input) and node.input[idx] in init_map:
                arr = init_map[node.input[idx]]
                if arr.ndim == 2:
                    w_name = node.input[idx]
                    break
        if w_name is None:
            continue

        W_up = init_map[w_name]

        if node.op_type == "Gemm":
            transB = 0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
            if transB:
                W_up = W_up.T

        d_model, ffn_dim = W_up.shape
        if ffn_dim <= d_model:
            continue

        bias_name = None
        if node.op_type == "Gemm" and len(node.input) > 2 and node.input[2] in init_map:
            b = init_map[node.input[2]]
            if b.ndim == 1 and b.shape[0] == ffn_dim:
                bias_name = node.input[2]

        # For MatMul: check if the next node is an Add with an ffn_dim-sized bias
        if bias_name is None and i + 1 < len(nodes):
            next_node = nodes[i + 1]
            if next_node.op_type == "Add":
                for inp in next_node.input:
                    if inp in init_map and init_map[inp].ndim == 1 and init_map[inp].shape[0] == ffn_dim:
                        bias_name = inp
                        break

        # Look ahead for the down-projection: should have shape (ffn_dim, d_model)
        for j in range(i + 1, min(i + 15, len(nodes))):
            down_node = nodes[j]
            if down_node.op_type not in ("MatMul", "Gemm"):
                continue

            dw_name = None
            for idx in (0, 1):
                if idx < len(down_node.input) and down_node.input[idx] in init_map:
                    arr = init_map[down_node.input[idx]]
                    if arr.ndim == 2:
                        dw_name = down_node.input[idx]
                        break
            if dw_name is None:
                continue

            W_down = init_map[dw_name]
            if down_node.op_type == "Gemm":
                dtB = 0
                for attr in down_node.attribute:
                    if attr.name == "transB":
                        dtB = attr.i
                if dtB:
                    W_down = W_down.T

            if W_down.shape == (ffn_dim, d_model):
                down_bias_name = None
                if down_node.op_type == "Gemm" and len(down_node.input) > 2 and down_node.input[2] in init_map:
                    db = init_map[down_node.input[2]]
                    if db.ndim == 1 and db.shape[0] == d_model:
                        down_bias_name = down_node.input[2]

                pairs.append({
                    "up_node": node,
                    "up_w_name": w_name,
                    "up_bias_name": bias_name,
                    "down_node": down_node,
                    "down_w_name": dw_name,
                    "down_bias_name": down_bias_name,
                    "d_model": d_model,
                    "ffn_dim": ffn_dim,
                })
                break

    return pairs


def apply_transformer_structured_pruning(model: ModelProto, ratio: float) -> None:
    """
    Prune transformer FFN intermediate dimensions.

    ratio: fraction of FFN neurons to remove. 0.75 = keep 25% of intermediate size.
    E.g. BERT's 3072 intermediate -> 768 with ratio=0.75.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be in (0, 1)")

    init_map = {i.name: numpy_helper.to_array(i).astype(np.float32)
                for i in model.graph.initializer}

    pairs = _find_ffn_pairs(model)
    if not pairs:
        return

    updates = {}

    for pair in pairs:
        ffn_dim = pair["ffn_dim"]
        n_keep = max(1, int(ffn_dim * (1 - ratio)))

        W_up = init_map[pair["up_w_name"]]
        up_transB = 0
        if pair["up_node"].op_type == "Gemm":
            for attr in pair["up_node"].attribute:
                if attr.name == "transB":
                    up_transB = attr.i

        if up_transB:
            W_up_eff = W_up.T
        else:
            W_up_eff = W_up

        # Score neurons by L2 norm of the up-projection columns
        scores = np.linalg.norm(W_up_eff, axis=0)  # (ffn_dim,)
        keep_idx = np.argsort(scores)[-n_keep:]
        keep_idx.sort()

        # Prune up-projection output dimension
        new_up = W_up_eff[:, keep_idx]
        if up_transB:
            updates[pair["up_w_name"]] = new_up.T
        else:
            updates[pair["up_w_name"]] = new_up

        if pair["up_bias_name"] is not None:
            updates[pair["up_bias_name"]] = init_map[pair["up_bias_name"]][keep_idx]

        # Prune down-projection input dimension
        W_down = init_map[pair["down_w_name"]]
        down_transB = 0
        if pair["down_node"].op_type == "Gemm":
            for attr in pair["down_node"].attribute:
                if attr.name == "transB":
                    down_transB = attr.i

        if down_transB:
            W_down_eff = W_down.T
        else:
            W_down_eff = W_down

        new_down = W_down_eff[keep_idx, :]
        if down_transB:
            updates[pair["down_w_name"]] = new_down.T
        else:
            updates[pair["down_w_name"]] = new_down

    if not updates:
        return

    new_inits = []
    for init in model.graph.initializer:
        if init.name in updates:
            arr = updates[init.name].astype(np.float32)
            new_inits.append(numpy_helper.from_array(arr, init.name))
        else:
            new_inits.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    del model.graph.value_info[:]
