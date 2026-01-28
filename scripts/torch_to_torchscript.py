"""
Script used for transforming the weights after training of a YOLO model (.pt)
into a TorchScript (.torchscript) model in the weights/ folder.
"""

import os
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser(
        description="Torch model to TorchScript model conversor"
    )

    parser.add_argument(
        '--data',
        required=True,
        type=str,
        help="The route to the .pt model"
    )

    return parser.parse_args()


def main():

    args = get_args()
    path = args.data
    parts = path.split(os.sep)
    
    parts[-1] = "best.torchscript"

    model = YOLO(path)
    model.export(format="torchscript")

    # check that effectively the model exists
    torchscript_model = Path(*filter(None, parts))
    try:
        model = torch.jit.load(torchscript_model)
        model.eval()
    except Exception as e:
        raise e

if __name__ == "__main__":
    main()