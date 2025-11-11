"""
This script contains the code for visualising and computing accuracy metrics
for the T-Rex 2 predictions.
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def get_args():
    parser = argparse.ArgumentParser(description='T-Rex2 Results Renderer')
    parser.add_argument(
        '--predictions',
        required=True,
        type=str,
        help='Path to T-Rex predictions JSON (trex_predictions.json)',
    )
    parser.add_argument(
        '--labels',
        required=True,
        type=str,
        help='Path to ground-truth COCO annotations JSON',
    )
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for matching boxes (default=0.5)',
    )
    return parser.parse_args()


def iou_xywh(a, b):
    """Compute IoU between two sets of boxes in [x, y, w, h]"""
    a = np.array(a)
    b = np.array(b)
    if len(a.shape) == 1:
        a = a[None, :]
    if len(b.shape) == 1:
        b = b[None, :]

    a_xy1 = a[:, :2]
    a_xy2 = a[:, :2] + a[:, 2:]
    b_xy1 = b[:, :2]
    b_xy2 = b[:, :2] + b[:, 2:]

    inter_xy1 = np.maximum(a_xy1[:, None, :], b_xy1[None, :, :])
    inter_xy2 = np.minimum(a_xy2[:, None, :], b_xy2[None, :, :])
    inter_wh = np.clip(inter_xy2 - inter_xy1, 0, None)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter + 1e-9
    return inter / union


def match_boxes(gt_boxes, pred_boxes, iou_thr=0.5):
    """Match predictions to ground truth boxes based on IoU threshold"""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return None, 0, 0, 0
    if len(gt_boxes) == 0:
        return None, 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return None, 0, 0, len(gt_boxes)

    iou_matrix = iou_xywh(gt_boxes, pred_boxes)
    gi, pj = linear_sum_assignment(-iou_matrix)
    matches = [(g, p) for g, p in zip(gi, pj) if iou_matrix[g, p] >= iou_thr]
    TP = len(matches)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP

    if TP > 0:
        mean_iou = np.mean([iou_matrix[g, p] for g, p in matches])
    else:
        mean_iou = None
    return mean_iou, TP, FP, FN


def main():
    args = get_args()

    with open(args.predictions) as f:
        preds = json.load(f)
    with open(args.labels) as f:
        labels = json.load(f)

    image_map = {img['file_name']: img['id'] for img in labels['images']}
    gt_ann = labels['annotations']

    results = []
    total_iou, total_tp, total_fp, total_fn = [], 0, 0, 0

    for img_name, pred_data in preds.items():
        if pred_data == '-1' or not pred_data:
            continue

        # ground-truth boxes
        img_id = image_map.get(img_name)
        gt_boxes = [a['bbox'] for a in gt_ann if a['image_id'] == img_id]

        # predicted boxes (T-Rex outputs x1,y1,x2,y2)
        try:
            pred_boxes_raw = [
                p['bbox'] for p in pred_data['data']['result']['objects']
            ]
            # Convert [x1,y1,x2,y2] → [x,y,w,h]
            pred_boxes = [
                [x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in pred_boxes_raw
            ]
            pred_scores = [
                p['score'] for p in pred_data['data']['result']['objects']
            ]
        except Exception:
            pred_boxes, pred_scores = [], []

        iou, TP, FP, FN = match_boxes(gt_boxes, pred_boxes, args.iou_thr)
        results.append(
            {'image': img_name, 'IOU': iou, 'TP': TP, 'FP': FP, 'FN': FN}
        )

        if iou is not None:
            total_iou.append(iou)
        total_tp += TP
        total_fp += FP
        total_fn += FN

    # global metrics
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    iou = np.mean(total_iou)

    df = pd.DataFrame(results)
    print(df.head())
    print('\n=== Global metrics ===')
    print(f'Precision: {precision:.3f}')
    print(f'Recall:    {recall:.3f}')
    print(f'F1-score:  {f1:.3f}')
    print(f'Average IoU (Intersection over Union): {iou:.3f}\n')

    out_path = os.path.splitext(args.predictions)[0] + '_metrics.csv'
    df.to_csv(out_path, index=False)
    print(f'\nPer-image results saved to {out_path}')


if __name__ == '__main__':
    main()
