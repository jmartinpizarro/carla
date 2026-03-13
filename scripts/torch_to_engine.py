"""
Convert a YOLO .pt model to a TensorRT .engine using:
PT → ONNX → TensorRT (trtexec)
"""

import argparse
import subprocess
from pathlib import Path
from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(
        description='Torch model to TensorRT engine converter'
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='Path to the .pt model',
    )
    parser.add_argument(
        '--tiled',
        action='store_true',
        help='Whether the model uses tiling (changes batch size)',
    )
    return parser.parse_args()


def main():
    args = get_args()
    pt_path = Path(args.model)

    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    # batch inference on the jetson is going to be done 2-2-2 (6 tiles)
    batch = 2 if args.tiled else 1

    # bcs of how the YOLO wrapper works, it is fucking broken. First, it is needed to create
    # a .onnx file and then transform it into the .engine
    print('Exporting ONNX...')
    model = YOLO(str(pt_path))
    model.export(
        format='onnx',
        imgsz=640,
        half=True,
        dynamic=False,
        batch=batch,
        device=0,
    )

    onnx_path = pt_path.with_suffix('.onnx')
    engine_path = onnx_path.with_suffix('.engine')

    print('Building TensorRT engine...')
    cmd = [
        'trtexec',
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        '--fp16',
        '--memPoolSize=workspace:4096MiB',
    ]

    subprocess.run(cmd, check=True)
    print(f'Engine created at: {engine_path}')


if __name__ == '__main__':
    main()
