"""
This file contains the script for launching multiple YOLO training runs
safely using subprocess, avoiding memory leaks in VRAM and RAM.
"""

import os
import time
import uuid
import argparse
import subprocess
import itertools

def get_args():
    parser = argparse.ArgumentParser(description="CARLA's YOLO Model Training Launcher")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results_yolo")
    return parser.parse_args()

training_space = {
    "model": [
        "yolov8s.pt",
        "yolov8l.pt",
        "yolo11s.pt",
        "yolo11l.pt",
    ],
    "epochs": [60, 100],
    "batch": [16, 32],
    "seed": [42],
    "box": [7.5, 10.0],
}

def grid_search(space):
    """Generate all possible combinations for the training space"""
    keys = list(space.keys())
    values = (space[k] if isinstance(space[k], list) else [space[k]] for k in keys)
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def run_training(config, data, outdir):
    exp_id = uuid.uuid4().hex[:5]

    model_path = config["model"]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    exp_name = (
        f"{model_name}_e{config['epochs']}_b{config['batch']}_"
        f"s{config['seed']}_box{config['box']}_{exp_id}"
    )

    print(f"\n[train_yolo.py] :: Starting experiment: {exp_name}")

    # Build the YOLO training command
    cmd = [
        "yolo", "detect", "train",
        f"model={model_path}",
        f"data={data}",
        f"epochs={config['epochs']}",
        f"batch={config['batch']}",
        f"imgsz=640",
        f"optimizer=auto",
        f"seed={config['seed']}",
        f"box={config['box']}",
        f"project={outdir}",
        f"name={exp_name}",
        "cache=disk",
        "plots=True"
    ]

    # Run training in a SEPARATE PROCESS
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[train_yolo.py] :: WARNING: Experiment {exp_name} crashed with code {result.returncode}")
    else:
        print(f"[train_yolo.py] :: Finished experiment: {exp_name}")


def main():
    args = get_args()
    start = time.time()

    print("[train_yolo.py] :: Grid Search Training Started\n")

    for config in grid_search(training_space):
        run_training(config, args.data, args.outdir)

    end = time.time()

    print("\n[train_yolo.py] :: All experiments finished.")
    print(f"Total elapsed time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
