#!/usr/bin/env python3

import argparse
import copy
from typing import Iterable

import onnx
from onnx import TensorProto, checker, helper, shape_inference


PAD_VALUES = {
    "/pre_conv/Pad": [0, 0, 2, 0, 0, 0],
    "/upsample.0.1/dwconv/Pad": [0, 0, 6, 0, 0, 0],
    "/upsample.1.1/dwconv/Pad": [0, 0, 6, 0, 0, 0],
    "/decoder.0/Pad": [0, 0, 6, 0, 0, 0],
    "/decoder.1/block.2/conv1/Pad": [0, 0, 6, 0, 0, 0],
    "/decoder.1/block.2/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.1/block.3/conv1/Pad": [0, 0, 18, 0, 0, 0],
    "/decoder.1/block.3/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.1/block.4/conv1/Pad": [0, 0, 54, 0, 0, 0],
    "/decoder.1/block.4/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.2/block.2/conv1/Pad": [0, 0, 6, 0, 0, 0],
    "/decoder.2/block.2/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.2/block.3/conv1/Pad": [0, 0, 18, 0, 0, 0],
    "/decoder.2/block.3/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.2/block.4/conv1/Pad": [0, 0, 54, 0, 0, 0],
    "/decoder.2/block.4/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.3/block.2/conv1/Pad": [0, 0, 6, 0, 0, 0],
    "/decoder.3/block.2/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.3/block.3/conv1/Pad": [0, 0, 18, 0, 0, 0],
    "/decoder.3/block.3/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.3/block.4/conv1/Pad": [0, 0, 54, 0, 0, 0],
    "/decoder.3/block.4/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.4/block.2/conv1/Pad": [0, 0, 6, 0, 0, 0],
    "/decoder.4/block.2/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.4/block.3/conv1/Pad": [0, 0, 18, 0, 0, 0],
    "/decoder.4/block.3/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.4/block.4/conv1/Pad": [0, 0, 54, 0, 0, 0],
    "/decoder.4/block.4/conv2/Pad": [0, 0, 0, 0, 0, 0],
    "/decoder.6/Pad": [0, 0, 6, 0, 0, 0],
}


def constant_i64_node(node_name: str, output_name: str, tensor_name: str, values: Iterable[int]):
    values = list(values)
    tensor = helper.make_tensor(
        name=tensor_name,
        data_type=TensorProto.INT64,
        dims=[len(values)],
        vals=values,
    )
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=node_name,
        value=tensor,
    )


def rewrite_split_node(node):
    if node.name == "/Split":
        return [
            helper.make_node(
                "Identity",
                inputs=[node.input[0]],
                outputs=[node.output[0]],
                name="/Split_Identity",
            )
        ]

    if node.name != "/Split_1":
        return [copy.deepcopy(node)]

    source = node.input[0]
    output_nodes = []
    for index, output_name in enumerate(node.output):
        start_output = f"/split1_const_start_{index}"
        end_output = f"/split1_const_end_{index}"
        axes_output = f"/split1_const_axes_{index}"
        steps_output = f"/split1_const_steps_{index}"

        output_nodes.append(
            constant_i64_node(
                f"/Split_1/ConstStart_{index}",
                start_output,
                f"split1_start_{index}",
                [index],
            )
        )
        output_nodes.append(
            constant_i64_node(
                f"/Split_1/ConstEnd_{index}",
                end_output,
                f"split1_end_{index}",
                [index + 1],
            )
        )
        output_nodes.append(
            constant_i64_node(
                f"/Split_1/ConstAxes_{index}",
                axes_output,
                f"split1_axes_{index}",
                [0],
            )
        )
        output_nodes.append(
            constant_i64_node(
                f"/Split_1/ConstSteps_{index}",
                steps_output,
                f"split1_steps_{index}",
                [1],
            )
        )
        output_nodes.append(
            helper.make_node(
                "Slice",
                inputs=[source, start_output, end_output, axes_output, steps_output],
                outputs=[output_name],
                name=f"/Split_1/Slice_{index}",
            )
        )

    return output_nodes


def rewrite_pad_node(node):
    pads = PAD_VALUES.get(node.name)
    if pads is None:
        return [copy.deepcopy(node)]

    if len(node.input) < 2:
        raise ValueError(f"Pad node {node.name} does not have a pads input")

    pad_const_name = f"{node.name}/StaticPads"
    pad_const_output = f"{node.name}/StaticPads_output_0"
    pad_tensor_name = f"{node.name}_pads"

    rewritten = copy.deepcopy(node)
    del rewritten.input[:]
    rewritten.input.extend([node.input[0], pad_const_output, node.input[2]])

    return [
        constant_i64_node(pad_const_name, pad_const_output, pad_tensor_name, pads),
        rewritten,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the Qwen3-TTS decoder ONNX graph for Burn/WGPU codegen."
    )
    parser.add_argument("input", help="Path to qwen3_tts_decoder.onnx")
    parser.add_argument("output", help="Path to the normalized ONNX file to write")
    args = parser.parse_args()

    model = onnx.load(args.input)
    model = shape_inference.infer_shapes(model)

    rewritten_nodes = []
    for node in model.graph.node:
        for split_node in rewrite_split_node(node):
            rewritten_nodes.extend(rewrite_pad_node(split_node))

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
