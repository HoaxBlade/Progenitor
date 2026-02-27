"""
Low-rank SVD decomposition: replace large weight matrices W (m x n) with
factored pairs U (m x r) and V (r x n) where r << min(m, n).

For a rank ratio of 0.25 on a 2048x2048 matrix:
  Original: 2048*2048 = 4M multiplies per matmul
  After:    2048*512 + 512*2048 = 2M multiplies -> ~2x fewer FLOPs

Works on any runtime (dense), no sparse backend needed. Stacks multiplicatively
with structured pruning and unstructured pruning.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper


def apply_lowrank_decomposition(model: ModelProto, rank_ratio: float = 0.25) -> None:
    """
    Replace each large weight matrix in MatMul/Gemm nodes with two smaller
    factored matrices via truncated SVD, in-place.

    rank_ratio: fraction of min(m, n) to keep. 0.25 means keep top 25% singular
    values -> ~4x fewer FLOPs per matmul -> ~2-3x overall speedup.
    """
    if not 0 < rank_ratio < 1:
        raise ValueError("rank_ratio must be in (0, 1)")

    init_map = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}
    init_names = set(init_map.keys())

    nodes_to_remove = []
    nodes_to_add = []
    inits_to_add = []
    inits_to_remove = set()

    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue

        w_name = None
        w_input_idx = None
        for idx in (0, 1):
            if idx < len(node.input) and node.input[idx] in init_names:
                arr = init_map[node.input[idx]]
                if arr.ndim == 2 and min(arr.shape) >= 8:
                    w_name = node.input[idx]
                    w_input_idx = idx
                    break

        if w_name is None:
            continue

        W = init_map[w_name]
        
        transB = 0
        if node.op_type == "Gemm":
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
                    break
        
        if transB:
            W = W.T
            
        m, n = W.shape
        r = max(1, int(min(m, n) * rank_ratio))

        if r >= min(m, n):
            continue

        U_full, S, Vt_full = np.linalg.svd(W, full_matrices=False)
        U = (U_full[:, :r] * S[:r]).astype(np.float32)  # (m, r) — absorb S into U
        V = Vt_full[:r, :].astype(np.float32)            # (r, n)

        u_name = f"{w_name}_U"
        v_name = f"{w_name}_V"
        
        # If the original Gemm had transB=1, it expects a weight matrix of shape (n, m).
        # We factored W (m, n) into U (m, r) and V (r, n).
        # We need the node replacement to execute X @ W_orig.
        # X @ (U @ V)
        # So MatMul1 = X @ U, Gemm2 = MatMul1 @ V.
        # But if transB=1, Gemm2 will transpose its second argument natively.
        # So we must provide V.T to Gemm2, and we must *remove* the transB flag from Gemm2 entirely,
        # OR we can keep transB=1 on Gemm2 and provide V. 
        # Actually, simpler: just remove transB from the Gemm2 attributes and always pass V (r, n).
        
        inits_to_add.append(numpy_helper.from_array(np.ascontiguousarray(U), u_name))
        inits_to_add.append(numpy_helper.from_array(np.ascontiguousarray(V), v_name))
        inits_to_remove.add(w_name)

        intermediate = f"{node.output[0]}_lowrank_mid"

        if node.op_type == "Gemm":
            # Gemm: Y = alpha * X @ W + beta * C
            # Replace with: mid = MatMul(X, U), then Y = Gemm(mid, V, C)
            x_input = node.input[0] if w_input_idx == 1 else node.input[1]

            matmul_node = helper.make_node(
                "MatMul",
                inputs=[x_input, u_name],
                outputs=[intermediate],
                name=f"{node.name}_U" if node.name else "",
            )

            gemm_inputs = [intermediate, v_name]
            if len(node.input) > 2:
                gemm_inputs.append(node.input[2])

            gemm_node = helper.make_node(
                "Gemm",
                inputs=gemm_inputs,
                outputs=list(node.output),
                name=f"{node.name}_V" if node.name else "",
            )
            for attr in node.attribute:
                # We do not carry over transB because V is stored in correct (r, n) shape
                if attr.name in ("alpha", "beta", "transA"):
                    gemm_node.attribute.append(attr)

            nodes_to_remove.append(node)
            nodes_to_add.append((node, [matmul_node, gemm_node]))
        else:
            # MatMul: Y = A @ B
            if w_input_idx == 1:
                a_input = node.input[0]
                matmul1 = helper.make_node("MatMul", [a_input, u_name], [intermediate], name=f"{node.name}_U" if node.name else "")
                matmul2 = helper.make_node("MatMul", [intermediate, v_name], list(node.output), name=f"{node.name}_V" if node.name else "")
            else:
                b_input = node.input[1]
                matmul1 = helper.make_node("MatMul", [u_name, b_input], [intermediate], name=f"{node.name}_U" if node.name else "")
                matmul2 = helper.make_node("MatMul", [intermediate, v_name], list(node.output), name=f"{node.name}_V" if node.name else "")

            nodes_to_remove.append(node)
            nodes_to_add.append((node, [matmul1, matmul2]))

    if not nodes_to_remove:
        return

    # Replace nodes in graph
    new_nodes = []
    remove_set = set(id(n) for n in nodes_to_remove)
    add_map = {id(orig): replacements for orig, replacements in nodes_to_add}
    for node in model.graph.node:
        if id(node) in remove_set:
            new_nodes.extend(add_map[id(node)])
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Update initializers
    new_inits = [i for i in model.graph.initializer if i.name not in inits_to_remove]
    new_inits.extend(inits_to_add)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)

    # Clear stale value_info — new intermediate tensors added, shapes changed.
    del model.graph.value_info[:]
