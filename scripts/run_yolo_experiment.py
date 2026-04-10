"""
This file contains the script for launching a full YOLO experiment
using YoloExperiment for training and testing.
"""

import argparse

from src.yolo.yolo_experiment import YoloExperiment


def get_args():
    parser = argparse.ArgumentParser('YoloExperiment Runner')

    parser.add_argument(
        '--config',
        required=True,
        help='Route to the .yaml file that contains the experiment config',
    )

    parser.add_argument(
        '--data',
        required=True,
        help='Route where the dataset is located. It must not point to data.yaml',
    )

    return parser.parse_args()


def main():
    args = get_args()

    experiment = YoloExperiment(config=args.config)
    experiment.start_experiment(data=args.data)

    return


if __name__ == '__main__':
    main()
