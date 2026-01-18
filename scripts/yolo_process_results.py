"""
This script contains the code for visualizing and computing accuracy metrics for the YOLO predictions (and its modifications)
"""

import os
import argparse

import pandas as pd
from ultralytics import YOLO
import numpy as np
import torch
from ultralytics.utils.metrics import box_iou
import cv2

TESTING_EXPORT_DIR = 'model_testing_results'
TILE_SIZE = 640
STRIDE = 160


def get_args():
    parser = argparse.ArgumentParser(
        description='YOLO and YOLO-mods Results Renderer'
    )

    parser.add_argument(
        '--data',
        required=True,
        type=str,
        help='Route where the .yaml definition of the dataset for YOLO can be found.',
    )

    parser.add_argument(
        '--results-folder',
        default='runs',
        type=str,
        help="Route where the output model and its results are generated. By default is 'runs/'",
    )

    parser.add_argument(
        '--tiled',
        required=True,
        type=bool,
        default=False,
        help='If the model has been trained using a tiled-input approach, activate this parameter for '
        "modifying the input of the model. By default is 'False'",
    )

    return parser.parse_args()


def load_yolo_gt(txt_path, img_shape):
    h, w = img_shape[:2]
    boxes = []

    if not os.path.exists(txt_path):
        return torch.empty((0, 4))

    with open(txt_path) as f:
        for line in f:
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])

    return torch.tensor(boxes)


# For computing the RMSE metric. Not TP based
def compute_rmse_from_model(model, img_dir, gt_dir, tiled=False, iou_thr=0.5):
    rmse_sum = 0.0
    rmse_n = 0

    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_boxes = load_yolo_gt(gt_path, img.shape)
        if len(gt_boxes) == 0:
            continue

        if tiled:
            pred_boxes = tiled_inference(model, img)
        else:
            results = model.predict(img_path, verbose=False)
            if len(results[0].boxes) == 0:
                continue
            pred_boxes = results[0].boxes.xyxy.cpu()

        if len(pred_boxes) == 0:
            continue

        iou = box_iou(gt_boxes, pred_boxes)
        best_iou, best_pred = iou.max(dim=1)
        valid = best_iou > iou_thr

        if valid.sum() == 0:
            continue

        gt_idx = torch.arange(len(gt_boxes))[valid]
        pred_idx = best_pred[valid]

        unique = {}
        for g, p, i in zip(
            gt_idx.tolist(), pred_idx.tolist(), best_iou[valid].tolist()
        ):
            if p not in unique or i > unique[p][1]:
                unique[p] = (g, i)

        pred_idx = torch.tensor(list(unique.keys()))
        gt_idx = torch.tensor([v[0] for v in unique.values()])

        pb = pred_boxes[pred_idx]
        gb = gt_boxes[gt_idx]

        pcx = (pb[:, 0] + pb[:, 2]) / 2
        pcy = (pb[:, 1] + pb[:, 3]) / 2
        gcx = (gb[:, 0] + gb[:, 2]) / 2
        gcy = (gb[:, 1] + gb[:, 3]) / 2

        rmse_sum += ((pcx - gcx) ** 2 + (pcy - gcy) ** 2).sum().item()
        rmse_n += len(pred_idx) * 2

    return np.sqrt(rmse_sum / rmse_n) if rmse_n > 0 else 0.0


def generate_tiles(img):
    h, w = img.shape[:2]

    x_starts = []
    x = 0
    while x + TILE_SIZE < w:
        x_starts.append(x)
        x += STRIDE
    x_starts.append(w - TILE_SIZE)

    y_starts = []
    y = 0
    while y + TILE_SIZE < h:
        y_starts.append(y)
        y += STRIDE
    y_starts.append(h - TILE_SIZE)

    tiles = []
    for y in y_starts:
        for x in x_starts:
            tile = img[y : y + TILE_SIZE, x : x + TILE_SIZE]
            tiles.append((tile, x, y))

    return tiles


def project_boxes_to_image(boxes, offset_x, offset_y):
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes


def tiled_inference(model, img, conf=0.25, iou=0.5):
    all_boxes = []
    all_scores = []

    tiles = generate_tiles(img)

    for tile, ox, oy in tiles:
        results = model.predict(tile, conf=conf, verbose=False)
        if len(results[0].boxes) == 0:
            continue

        boxes = results[0].boxes.xyxy.cpu()
        scores = results[0].boxes.conf.cpu()

        boxes = project_boxes_to_image(boxes, ox, oy)

        all_boxes.append(boxes)
        all_scores.append(scores)

    if not all_boxes:
        return torch.empty((0, 4))

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)

    keep = torch.ops.torchvision.nms(boxes, scores, iou)
    return boxes[keep]


def main():
    args = get_args()

    best_precision = 0.0
    best_precission_model = None
    best_map50 = 0.0
    best_map50_model = None

    models = os.listdir(args.results_folder)

    models_output = {}

    print('[process_yolo_results] :: Starting the script\n')

    for model in models:
        print(f'\n\n\t[process_yolo_results] :: Testing {model}\n\n')

        route = os.path.join(args.results_folder, model)
        models_output[model] = {}

        # if the model could not be trained, skip it
        if model == 'detect' or not os.listdir(os.path.join(route, 'weights')):
            print(f'\tNo model founded for the testing evaluation\n\n')
            continue

        df = pd.read_csv(os.path.join(route, 'results.csv'))
        # it will just contain the metrics/precision(B),metrics/mAP50(B),metrics/mAP50-95(B)
        metrics = df[
            [
                'epoch',
                'metrics/precision(B)',
                'metrics/mAP50(B)',
                'metrics/mAP50-95(B)',
            ]
        ]

        models_output[model]['training_metrics'] = metrics

        # now it is necessary to obtain the metrics with the test set
        YOLO_VAL = YOLO(f'{route}/weights/best.pt')
        prediction_metrics = YOLO_VAL.val(
            data=os.path.join(args.data, 'data.yaml'),
            split='test',
            imgsz=640,
            half=True,
            device='cuda',
            save=True,
            name=route,
            save_conf=True,
            save_txt=False,
            save_json=False,
            verbose=False,
        )

        YOLO_PRED = YOLO(f'{route}/weights/best.pt')

        rmse = compute_rmse_from_model(
            model=YOLO_PRED,
            img_dir=os.path.join(args.data, 'test', 'images'),
            gt_dir=os.path.join(args.data, 'test', 'labels'),
            tiled=args.tiled,
        )

        models_output[model]['test_metrics'] = {
            'mAP50': prediction_metrics.box.map50,
            'mAP50-95': prediction_metrics.box.map,
            'precision': prediction_metrics.box.p,
            'recall': prediction_metrics.box.r,
            'rmse': rmse,
        }

        # without any more relevant analysis, its complicated to define which one is the
        # best. However, it is possible to save the best precission and mAP50
        if prediction_metrics.box.p[0] >= best_precision:
            best_precision = prediction_metrics.box.p[0]  # only one class
            best_precission_model = model

        if prediction_metrics.box.map50 >= best_map50:
            best_map50 = prediction_metrics.box.map50
            best_map50_model = model

        print(
            '[process_yolo_results] :: The script has ended\n',
            f'\t The model with the best precission was: {best_precission_model} with {best_precision}\n',
            f'\t The model with the best map50 was: {best_map50_model} with {best_map50}\n',
        )

        os.makedirs(f'{TESTING_EXPORT_DIR}/{model}', exist_ok=True)

        # Training metrics → CSV
        training_csv_path = os.path.join(
            f'{TESTING_EXPORT_DIR}/{model}', f'{model}_training_metrics.csv'
        )
        models_output[model]['training_metrics'].to_csv(
            training_csv_path, index=False
        )

        # Test metrics → CSV
        test_metrics_df = pd.DataFrame([models_output[model]['test_metrics']])
        test_csv_path = os.path.join(
            f'{TESTING_EXPORT_DIR}/{model}', f'{model}_test_metrics.csv'
        )
        test_metrics_df.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    main()
