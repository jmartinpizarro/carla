"""
This file contains the script for launching multiple YOLO training runs
safely using subprocess, avoiding memory leaks in VRAM and RAM.
"""

import os
import time
import uuid
import argparse
import itertools

from ultralytics import YOLO

training_space = {
    'model': ['yolov8n.pt', 'yolo11n.pt', 'yolo26n.pt'],
    'epochs': [200],
    'batch': [16],
    'seed': [42, 43, 44, 45, 46],
    'box': [5.0],
}


def get_args():
    parser = argparse.ArgumentParser('Trainer')

    parser.add_argument('--data', required=True, help='Route to .yaml')

    parser.add_argument('--outdir', required=True, help='Output folder')

    return parser.parse_args()


def grid_search(space):
    """Generate all possible combinations for the training space"""
    keys = list(space.keys())
    values = (
        space[k] if isinstance(space[k], list) else [space[k]] for k in keys
    )
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_training(config, data, outdir):
    exp_id = uuid.uuid4().hex[:5]

    model_path = config['model']
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    exp_name = (
        f'{model_name}_e{config["epochs"]}_b{config["batch"]}_'
        f's{config["seed"]}_box{config["box"]}_{exp_id}'
    )

    print(f'\n[train_yolo.py] :: Starting experiment: {exp_name}')

    model = YOLO(model_path)
    # This parameters have been obtained through experimentations
    model.train(
        data=data,
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=640,
        optimizer='auto',
        seed=config['seed'],
        box=config['box'],
        project=outdir,
        name=exp_name,
        cache='disk',
        plots=True,
    )


def main():
    # TODO: not necessary to refactor this
    args = get_args()
    start = time.time()

    print('[train_yolo.py] :: Grid Search Training Started\n')

    for config in grid_search(training_space):
        try:
            run_training(config, args.data, args.outdir)
        except Exception as e:
            print(
                f'[train_yolo] :: An exception has ocurred when training the model {config}\n\n\t{e}'
            )

    end = time.time()

    print('\n[train_yolo.py] :: All experiments finished.')
    print(f'Total elapsed time: {end - start:.2f} seconds')


if __name__ == '__main__':
    main()
