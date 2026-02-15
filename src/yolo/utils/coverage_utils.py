import numpy as np
from PIL import Image


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
