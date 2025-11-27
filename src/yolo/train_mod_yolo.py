"""
This file contains the script for the YOLOv11L with different modifications in order to check
the performance of it in the Cardilla Problem.

After researching, it is seen that some of the best configuration for this problem is:
    - Epochs: 100
    - Batch: 16 (as 32 won't run in our machine)
    - BoxLoss: 7.5

Now, an exploration on different modifications to the YOLO Arquitecture will be applied, in order
check if it is possible to achieve better results doing these experiments
"""

import os
import time
import subprocess


from src.yolo.utils import *


YOLO_MODEL = 'yolo11t.pt'
EPOCHS = '100'
BATCH = '16'
BOX_LOSS = '7.5'  # although this is default
SEED = '42'

# this .yaml file contains - each one- a slightly modification of the original arquitecture
configs_file = []


def main():
    args = get_args()

    start = time.time()

    for _f in configs_file:
        cmd = [
            'yolo',
            'detect',
            'train',
            f'model={YOLO_MODEL}',
            f'data={_f}',
            f'epochs={EPOCHS}',
            f'batch={BATCH}',
            'imgsz=640',
            'optimizer=auto',
            f'seed={SEED}',
            f'box={BOX_LOSS}',
            f'project=runs_mod',
            f'name={_f[0:-5]}',
            'cache=disk',
            'plots=True',
        ]

        # Run training in a SEPARATE PROCESS
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(
                f'[train_mod_yolo.py] :: WARNING: Experiment {_f} crashed with code {result.returncode}'
            )
        else:
            print(f'[train_mod_yolo.py] :: Finished experiment: {_f}')

    end = time.time()

    print(
        f'\n[train_mod_yolo.py] :: The training has finished! It lapsed a total time of {(end - start):.2f}\n'
    )


if __name__ == '__main__':
    main()
