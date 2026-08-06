"""Basic inference script. It should be used for prototyping or quick testing"""

import argparse
import matplotlib

matplotlib.use('Agg')  # otherwise memory goes fucked
from src.yolo.yolo_model import YoloModel


def get_args():
    parser = argparse.ArgumentParser('PoC Inference')

    parser.add_argument('--model', required=True, help='The route to the .pt model')

    parser.add_argument(
        '--tiled',
        action='store_true',
        required=False,
        help='The model that uses the inference uses tiling mechanisms or not',
    )

    parser.add_argument(
        '--input-data',
        required=True,
        help='The route to the file (image or video) to which inference is going to be made',
    )

    parser.add_argument(
        '--log-files',
        required=False,
        default='output.pred',
        help='Depuration or extra information. Creates a file and anotates there coverage, boxes location and more.',
    )

    return parser.parse_args()


def main():
    args = get_args()
    model = YoloModel(
        model=args.model,
        tiled=args.tiled,
        input_data=args.input_data,
        log_files=args.log_files,
    )

    # PART 1: THE INFERENCE
    _ = model.inference()

    return


if __name__ == '__main__':
    main()
