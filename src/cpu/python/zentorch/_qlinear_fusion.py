# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import functools
import torch
from torch._inductor import config
from torch._inductor.pattern_matcher import (
    PatternMatcherPass,
    register_graph_pattern,
    CallFunction,
    Arg,
    Match,
    stable_topological_sort,
)

matcher_pass = PatternMatcherPass(pass_name="qlinear_fusion_pass")
aten = torch.ops.aten
zentorch = torch.ops.zentorch
torch_decomp = torch.ops.quantized_decomposed

# QLinear->Q-DQ->Mul->Add pattern fusion


# Replacement implementation
def _qlinear_q_dq_mul_add_replacement_impl(
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    mul_input,
    add_input,
):
    output = zentorch.zentorch_qlinear_mul_add(
        input,
        weight,
        bias,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        mul_input,
        add_input,
        output_dtype=input.dtype,
        output_scales=None,
        output_zero_points=None,
    )
    return (output,)


# Pattern
@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        CallFunction(  # Mul
            aten.mul.Tensor,
            Arg(),  # Mul input
            CallFunction(  # Dequantize input
                torch_decomp.dequantize_per_tensor.default,
                CallFunction(  # Quantize input
                    torch_decomp.quantize_per_tensor.default,
                    CallFunction(  # QLinear
                        zentorch.zentorch_qlinear.default,
                        Arg(),  # Input
                        Arg(),  # Weight
                        Arg(),  # Bias
                        Arg(),  # Input scales
                        Arg(),  # Input zero points
                        Arg(),  # Weight scales
                        Arg(),  # Weight zero points
                    ),
                    Arg(),
                    Arg(),
                    0,
                    255,
                    torch.uint8,
                ),
                Arg(),
                Arg(),
                0,
                255,
                torch.uint8,
            ),
        ),
        Arg(),  # Add input
    ),
    pass_dict=matcher_pass,
)
# We are not using the q_scale, q_zero_point, dq_scale, dq_zero_point arguments,
# as we are not utilizing the Quantized Quantize and Dequantize ops introduced before the mul op.
def qlinear_q_dq_mul_add_replacement_decorated(
    match: Match,
    mul_input,
    input,
    weight,
    bias,
    input_scales,
    input_zero_points,
    weight_scales,
    weight_zero_points,
    q_scale,
    q_zero_point,
    dq_scale,
    dq_zero_point,
    add_input,
):
    match.replace_by_example(
        _qlinear_q_dq_mul_add_replacement_impl,
        [
            input,
            weight,
            bias,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            mul_input,
            add_input,
        ],
    )


# TODO : Add isometric pattern fusion for QLinear->Q-DQ->Mul->Add pattern
# Current pattern is applicable only to the quantized MLPerf DLRMv2 model.


def qlinear_fusion_pass(graph):
    if config.pattern_matcher:
        GraphTransformObserver = functools.partial(
            torch.fx.passes.graph_transform_observer.GraphTransformObserver,
            subsystem="qlinear_fusion_pass",
        )
        assert graph.owning_module is not None, "Graph has no owning module"
        replacements = GraphTransformObserver(
            graph.owning_module, "qlinear_fusion_pass"
        ).apply_graph_pass(matcher_pass.apply)
        if replacements is not None:
            stable_topological_sort(graph)
            graph.lint()

    return graph
