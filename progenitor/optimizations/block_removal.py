"""
Residual block removal for CNNs: replace low-importance blocks with identity (skip connection).

In ResNet, each residual block computes:
    output = F(x) + x   (or output = F(x) + shortcut(x) for projection blocks)

If F(x) contributes little to the output (low importance), we can remove the block's
convolutions entirely and replace the block with an identity: output = x.

This removes entire blocks worth of compute, producing dramatic speedups.

Block importance is scored by the average L2 norm of the block's internal weights
relative to the input magnitude.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper


def _find_residual_blocks(model: ModelProto):
    """
    Find residual blocks: sequences of ops ending in an Add node where both inputs
    come from the same source (one through convolutions, one direct/shortcut).
    
    Returns list of dicts with: add_node, branch_nodes (to remove), type ('identity'|'projection')
    """
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    
    producer = {}
    for n in model.graph.node:
        for out in n.output:
            producer[out] = n
    
    consumers = {}
    for n in model.graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    
    blocks = []
    
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        if len(node.input) != 2:
            continue
        
        inp_a, inp_b = node.input[0], node.input[1]
        
        # Trace back from each input to find branches
        def trace_branch(tensor_name, depth=0):
            """Trace back through Conv/Relu/BatchNorm nodes, return (source_tensor, [nodes])."""
            if depth > 20:
                return tensor_name, []
            if tensor_name not in producer:
                return tensor_name, []
            p = producer[tensor_name]
            if p.op_type in ("Conv", "Relu", "BatchNormalization"):
                source, nodes = trace_branch(p.input[0], depth + 1)
                return source, [p] + nodes
            return tensor_name, []
        
        source_a, nodes_a = trace_branch(inp_a)
        source_b, nodes_b = trace_branch(inp_b)
        
        # Both branches should originate from the same tensor (or via a Relu from the same Add)
        if source_a != source_b:
            continue
        
        # Identify which is the main branch (more conv nodes) and which is skip
        conv_count_a = sum(1 for n in nodes_a if n.op_type == "Conv")
        conv_count_b = sum(1 for n in nodes_b if n.op_type == "Conv")
        
        if conv_count_a >= 2 and conv_count_b <= 1:
            main_branch = nodes_a
            skip_branch = nodes_b
            main_input = inp_a
            skip_input = inp_b
        elif conv_count_b >= 2 and conv_count_a <= 1:
            main_branch = nodes_b
            skip_branch = nodes_a
            main_input = inp_b
            skip_input = inp_a
        else:
            continue
        
        has_shortcut_conv = any(n.op_type == "Conv" for n in skip_branch)
        
        # Score block importance: average L2 norm of conv weights in main branch
        importance = 0.0
        n_conv = 0
        for n in main_branch:
            if n.op_type == "Conv" and n.input[1] in init_map:
                w = init_map[n.input[1]]
                importance += np.linalg.norm(w)
                n_conv += 1
        if n_conv > 0:
            importance /= n_conv
        
        blocks.append({
            "add_node": node,
            "main_branch": main_branch,
            "skip_branch": skip_branch,
            "main_input": main_input,
            "skip_input": skip_input,
            "source": source_a,
            "has_shortcut_conv": has_shortcut_conv,
            "importance": importance,
        })
    
    return blocks


def apply_block_removal(model: ModelProto, removal_ratio: float) -> None:
    """
    Remove the least important residual blocks from a CNN.
    
    removal_ratio: fraction of removable blocks to eliminate (e.g. 0.5 = remove half).
    Only identity blocks (no projection shortcut) are candidates for removal,
    since projection blocks change tensor dimensions.
    """
    if not 0 < removal_ratio < 1:
        raise ValueError("removal_ratio must be in (0, 1)")
    
    blocks = _find_residual_blocks(model)
    
    # Only remove identity blocks (no shortcut conv) — projection blocks change dimensions
    removable = [b for b in blocks if not b["has_shortcut_conv"]]
    
    if not removable:
        return
    
    # Sort by importance (ascending) — remove least important first
    removable.sort(key=lambda b: b["importance"])
    
    n_remove = max(1, int(len(removable) * removal_ratio))
    to_remove = removable[:n_remove]
    
    if not to_remove:
        return
    
    # Collect all nodes that belong to removed blocks' main branches
    nodes_to_delete = set()
    # Map: output tensor of Add node -> skip input tensor (to rewire)
    rewire = {}
    
    for block in to_remove:
        # Only delete nodes exclusive to the main branch (not shared with skip)
        skip_ids = set(id(n) for n in block["skip_branch"])
        for n in block["main_branch"]:
            if id(n) not in skip_ids:
                nodes_to_delete.add(id(n))
        rewire[block["add_node"].output[0]] = block["skip_input"]
        nodes_to_delete.add(id(block["add_node"]))
    
    # Rebuild the node list, replacing removed nodes and rewiring inputs
    new_nodes = []
    for node in model.graph.node:
        if id(node) in nodes_to_delete:
            continue
        # Rewire inputs: if any input was an Add output we removed, point to the skip
        new_inputs = []
        for inp in node.input:
            new_inputs.append(rewire.get(inp, inp))
        
        if list(node.input) != new_inputs:
            del node.input[:]
            node.input.extend(new_inputs)
        
        new_nodes.append(node)
    
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    
    # Remove orphaned initializers (weights of deleted conv nodes)
    used_inputs = set()
    for n in model.graph.node:
        used_inputs.update(n.input)
    for out in model.graph.output:
        used_inputs.add(out.name)
    
    new_inits = [i for i in model.graph.initializer if i.name in used_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    
    del model.graph.value_info[:]
