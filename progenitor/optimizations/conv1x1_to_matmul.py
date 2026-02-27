"""
Convert 1x1 Conv layers to equivalent MatMul + Reshape operations.

A Conv with kernel_size=1, stride=1, no padding, group=1 on input (N, C_in, H, W) with
weight (C_out, C_in, 1, 1) is equivalent to:
    1. Reshape input: (N, C_in, H, W) -> (N*H*W, C_in)  [via Transpose + Reshape]
    2. MatMul: (N*H*W, C_in) @ (C_in, C_out) -> (N*H*W, C_out)
    3. Add bias
    4. Reshape back: (N*H*W, C_out) -> (N, C_out, H, W)

This enables the use of sparse backends on what were previously Conv operations.
Only converts 1x1, stride=1, no-dilation, group=1 convolutions.
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper


def _is_eligible_conv(node, init_map):
    """Check if a Conv node is eligible for conversion to MatMul."""
    if node.op_type != "Conv":
        return False
    if node.input[1] not in init_map:
        return False
    
    w = init_map[node.input[1]]
    if w.ndim != 4:
        return False
    
    _, _, kH, kW = w.shape
    if kH != 1 or kW != 1:
        return False
    
    attrs = {a.name: a for a in node.attribute}
    
    strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
    if strides != [1, 1]:
        return False
    
    pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
    if any(p != 0 for p in pads):
        return False
    
    group = attrs["group"].i if "group" in attrs else 1
    if group != 1:
        return False
    
    dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
    if dilations != [1, 1]:
        return False
    
    return True


def apply_conv1x1_to_matmul(model: ModelProto) -> int:
    """
    Convert eligible 1x1 Conv nodes to MatMul equivalents in-place.
    Returns the number of conversions made.
    
    The conversion uses Transpose + Reshape + MatMul + (Add) + Reshape + Transpose
    to maintain NCHW tensor format compatibility.
    """
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    
    nodes_to_remove = []
    nodes_to_add = []
    inits_to_add = []
    inits_to_remove = set()
    
    for node in model.graph.node:
        if not _is_eligible_conv(node, init_map):
            continue
        
        w_name = node.input[1]
        w = init_map[w_name]  # (C_out, C_in, 1, 1)
        c_out, c_in = w.shape[0], w.shape[1]
        
        has_bias = len(node.input) > 2 and node.input[2] in init_map
        bias_name = node.input[2] if has_bias else None
        
        # Reshape weight: (C_out, C_in, 1, 1) -> (C_in, C_out) for MatMul
        w_2d = w.reshape(c_out, c_in).T.astype(np.float32)
        w_2d_name = f"{w_name}_2d"
        inits_to_add.append(numpy_helper.from_array(np.ascontiguousarray(w_2d), w_2d_name))
        inits_to_remove.add(w_name)
        
        conv_input = node.input[0]
        conv_output = node.output[0]
        prefix = conv_output
        
        # Step 1: Transpose input from (N, C_in, H, W) -> (N, H, W, C_in)
        trans_out = f"{prefix}_nhwc"
        transpose_node = helper.make_node(
            "Transpose",
            inputs=[conv_input],
            outputs=[trans_out],
            perm=[0, 2, 3, 1],
        )
        
        # Step 2: Reshape from (N, H, W, C_in) -> (-1, C_in)
        reshape1_out = f"{prefix}_flat"
        reshape1_shape_name = f"{prefix}_flat_shape"
        inits_to_add.append(numpy_helper.from_array(
            np.array([-1, c_in], dtype=np.int64), reshape1_shape_name))
        reshape1_node = helper.make_node(
            "Reshape",
            inputs=[trans_out, reshape1_shape_name],
            outputs=[reshape1_out],
        )
        
        # Step 3: MatMul (-1, C_in) @ (C_in, C_out) -> (-1, C_out)
        matmul_out = f"{prefix}_mm"
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[reshape1_out, w_2d_name],
            outputs=[matmul_out],
        )
        
        # Step 4: Add bias if present
        if has_bias:
            add_out = f"{prefix}_biased"
            add_node = helper.make_node(
                "Add",
                inputs=[matmul_out, bias_name],
                outputs=[add_out],
            )
            last_out = add_out
        else:
            add_node = None
            last_out = matmul_out
        
        # Step 5: Reshape from (-1, C_out) -> (N, H, W, C_out) using original spatial dims
        # We use a shape-inference trick: reshape to (0, -1, C_out) then let ORT figure it out
        # Actually, safer: reshape to (1, -1, C_out) since batch=1 at inference,
        # then use Reshape + Transpose
        
        # For simplicity with dynamic shapes: (N*H*W, C_out) -> (-1, C_out) -> need H,W info
        # Use Shape + Gather to extract H, W from the original input dynamically
        
        # Get shape of original input
        shape_out = f"{prefix}_orig_shape"
        shape_node = helper.make_node("Shape", inputs=[conv_input], outputs=[shape_out])
        
        # Gather indices for N, H, W (indices 0, 2, 3)
        nhw_indices_name = f"{prefix}_nhw_idx"
        inits_to_add.append(numpy_helper.from_array(
            np.array([0, 2, 3], dtype=np.int64), nhw_indices_name))
        nhw_out = f"{prefix}_nhw"
        gather_nhw = helper.make_node("Gather", inputs=[shape_out, nhw_indices_name], outputs=[nhw_out], axis=0)
        
        # Build target shape: [N, H, W, C_out]
        cout_const_name = f"{prefix}_cout"
        inits_to_add.append(numpy_helper.from_array(
            np.array([c_out], dtype=np.int64), cout_const_name))
        reshape2_shape = f"{prefix}_nhwc_shape"
        concat_node = helper.make_node("Concat", inputs=[nhw_out, cout_const_name], outputs=[reshape2_shape], axis=0)
        
        reshape2_out = f"{prefix}_nhwc_out"
        reshape2_node = helper.make_node("Reshape", inputs=[last_out, reshape2_shape], outputs=[reshape2_out])
        
        # Step 6: Transpose back: (N, H, W, C_out) -> (N, C_out, H, W)
        transpose_back = helper.make_node(
            "Transpose",
            inputs=[reshape2_out],
            outputs=[conv_output],
            perm=[0, 3, 1, 2],
        )
        
        replacement_nodes = [transpose_node, reshape1_node, matmul_node]
        if add_node:
            replacement_nodes.append(add_node)
        replacement_nodes.extend([shape_node, gather_nhw, concat_node, reshape2_node, transpose_back])
        
        nodes_to_remove.append(node)
        nodes_to_add.append((node, replacement_nodes))
    
    if not nodes_to_remove:
        return 0
    
    # Replace nodes
    remove_set = set(id(n) for n in nodes_to_remove)
    add_map = {id(orig): replacements for orig, replacements in nodes_to_add}
    new_nodes = []
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
    
    del model.graph.value_info[:]
    
    return len(nodes_to_remove)
