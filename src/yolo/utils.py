import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="CARLA's YOLO Model Training Launcher"
    )
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='results_yolo')
    return parser.parse_args()
