"""
This file contains the script for training a YOLO model
"""

import os
import argparse
import time

from ultralytics import YOLO

def get_args():
    parser = argparse.ArgumentParser(description="CARLA's YOLO Model Training Script")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results_yolo")
    return parser.parse_args()


training_space = {
    "model": [
        "yolov5su.pt",
        "yolov5lu.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov11s.pt",
        "yolov11l.pt", # last is an "l" not a one :)
    ],
    "epochs": [50, 75, 100],
    "batch": [8, 16, 32],
    "imgsz": [640, 1280],
    "optimizer": ['auto', 'AdamW'],
    "seed": [42],
    "box": [5.0, 7.5, 10.0],
}

def grid_search(space):
    """Generate all possible combinations for the training space"""
    import itertools

    keys = list(space.keys())
    values = (space[k] if isinstance(space[k], list) else [space[k]] for k in keys)

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


if __name__ == "__main__":
    args = get_args()

    start = time.time()
    print("[train_yolo.py] :: The training of YOLO models has started\n\n")

    for config in grid_search(training_space):

        model_path = config["model"]
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # unique name per experiment
        exp_name = (
            f"{model_name}_e{config['epochs']}_b{config['batch']}_"
            f"s{config['seed']}_img{config['imgsz']}_box{config['box']}"
        )

        print(f"\n\t[train_yolo.py] :: Running experiment: {exp_name}")

        model = YOLO(model_path)

        model.train(
            data=args.data,
            epochs=config["epochs"],
            batch=config["batch"],
            imgsz=config["imgsz"],
            optimizer=config["optimizer"],
            seed=config["seed"],
            box=config["box"],
            project=args.outdir,
            name=exp_name,   
            cache=True,
            exist_ok=True
        )

    end = time.time()

    print(f"\n[train_yolo.py] :: The training of YOLO models has finished.\n",
          f"It has lasted a total elapsed time of {(end - start):2f} seconds\n")


