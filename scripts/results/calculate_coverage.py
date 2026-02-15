"""
Script used for calculating the coverage between GT-Preds
"""

import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser('Arguments for calculating the coverage')

    parser.add_argument(
        '--data',
        required=True,
        help='The data folder where the .yaml file is included',
    )

    parser.add_argument(
        '--models-route',
        required=True,
        help='The data folder were all the models are inside',
    )

    return parser.parse_args()


def gt_coverage_percent(image_path: str, label_path: str):
    img = Image.open(image_path)
    w, h = img.size
    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        with open(label_path) as f:
            for line in f:
                _, xc, yc, bw, bh = map(float, line.split())
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                mask[y1:y2, x1:x2] = 1
    except FileNotFoundError:
        # the image does not contain any prediction
        return 0.0

    return 100 * mask.sum() / (w * h)


def pred_coverage_percent(image_path, model, conf=0.3):
    img = Image.open(image_path)
    w, h = img.size
    mask = np.zeros((h, w), dtype=np.uint8)

    results = model(image_path, conf=conf, iou=0.4, verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            mask[y1:y2, x1:x2] = 1

    return 100 * mask.sum() / (w * h)


def main():
    # Basically, for each element in the test/ folder, it is necessary
    # to calculate the total area occupied by boxes by the GTs and the
    # predictions of the model. After that, it is just necessary to
    # calculate the mean for the entire split of the dataset.

    args = get_args()
    data = args.data
    models_route = args.models_route

    models = os.listdir(models_route)

    models_coverage = {}
    results = []

    for model in models:
        route = os.path.join(models_route, model)
        models_coverage[model] = {'gt': [], 'pred': []}

        # if the model could not be trained, skip it
        if model == 'detect' or not os.listdir(os.path.join(route, 'weights')):
            print('\tNo model founded for the testing evaluation\n\n')
            continue

        yolo_model = YOLO(f'{route}/weights/best.pt')

        # for each image of the dataset, process it and add it to the dict
        for image in os.listdir(f'{data}/test/images'):
            image_route = f'{data}/test/images/{image}'
            label_route = f'{data}/test/labels/{image[:-4]}.txt'

            models_coverage[model]['gt'].append(
                gt_coverage_percent(image_route, label_route)
            )
            models_coverage[model]['pred'].append(
                pred_coverage_percent(image_route, yolo_model)
            )

        mean_pred = np.mean(models_coverage[model]['pred'])
        mean_gt = np.mean(models_coverage[model]['gt'])
        abs_diff = abs(mean_pred - mean_gt)

        results.append(
            {
                'model': model,
                'mean_gt_coverage': mean_gt,
                'mean_pred_coverage': mean_pred,
                'abs_diff': abs_diff,
            }
        )

        print(mean_pred)
        print(mean_gt)
        print(abs_diff)

        df = pd.DataFrame(results)
        df.to_csv('coverage_summary.csv', index=False)

    return 0


if __name__ == '__main__':
    main()
