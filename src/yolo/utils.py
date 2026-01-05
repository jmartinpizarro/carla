import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="CARLA's YOLO Model Training Launcher"
    )
    parser.add_argument(
        '--data',
        type=str,
        help='The route for the .yaml file that contains the YOLO Dataset information',
        default='data/lowFlightRGB_v7/data.yaml',
    )
    parser.add_argument('--outdir', type=str, default='results_yolo')
    return parser.parse_args()
