#!/usr/bin/env python3
"""Export a PyTorch model to ONNX format so it can be used in the 
browser application.

PYTHONPATH=. wild_sight/train/export_to_onnx.py \
    --timestamp 
"""

import argparse

import torch

from wild_sight.core import detector


@torch.no_grad()
def convert_model(timestamp: str) -> None:

    model = detector.Detector(timestamp=timestamp)
    model.eval()
    # Input to the model
    x = torch.randn(1, 3, model.img_height, model.img_width, requires_grad=True)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        "/tmp/detector.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable lenght axes
            "output": {0: "clf", 1: "reg"},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    convert_model(args.timestamp)
