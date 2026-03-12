"""
Script used for transforming the weights after training of a YOLO model (.pt)
into an Engine (.engine) model in the weights/ folder.
"""

import argparse

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(
        description='Torch model to .engine model conversor'
    )

    parser.add_argument(
        '--model', required=True, type=str, help='The route to the .pt model'
    )

    parser.add_argument(
        '--tiled',
        action='store_true',
        help='The model uses tiling approaches or not',
    )

    return parser.parse_args()


def main():
    args = get_args()
    path = args.model

    if args.tiled is True:
        BATCH = 6
    else:
        BATCH = 1

    model = YOLO(path)
    model.export(format='engine', half=True, workspace=4, batch=BATCH, device=0)


if __name__ == '__main__':
    main()
