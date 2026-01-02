"""
This script contains the code for tiling the original dataset. Given the original labelled dataset (1920x1080 aspect ratio)
samples are relatevely small when converted into images of 640px during training, reducing precission.

Creating a new labelled dataset improves the model, at the cost of increasing training time.
"""

import os
import argparse
from typing import Tuple, List

import cv2

# pixels that overlap for the same images = 640 x 640 * 25% = 160
# asume tile size of 640 for every tile, 640 - 160 = 480 of new image in every iter
TILE_SIZE = 640
OVERLAP = 160
STRIDE = TILE_SIZE - OVERLAP


def get_args():
    parser = argparse.ArgumentParser(
        description="CARLA's Dataset Tiling Script. Transform the original dataset into a tiled one for increased accuracy performance"
    )

    parser.add_argument('--data', type=str, help='The route of the dataset.')

    parser.add_argument(
        '--output-data',
        type=str,
        help='Output route for the new transformed images',
        default='data',
    )

    return parser.parse_args()


def get_routes(data_route: str) -> Tuple[List, List]:
    """
    Gets both routes, for the images and labels for all the splits - train, val and test

    @param data_route: str -> the route where the dataset structure definition is located
    @returns Tuple(List, List): returns a tuple with the form (images_routes, labels_routes)
    """

    image_split = ['test/', 'train/', 'valid/']
    image_category = ['images', 'labels']

    from itertools import product

    product_routes = product(image_split, image_category)
    images_routes = []
    labels_routes = []
    for split, category in product_routes:
        path = os.path.join(data_route, split, category)
        if category == 'images':
            images_routes.append(path)
        else:
            labels_routes.append(path)

    return (images_routes, labels_routes)


def process_labels(label_path, tile_x, tile_y, img_w, img_h) -> List[str]:
    """
    Transforms YOLO labels into pixels in order to process it for the new labels of the tiles
    
    @param label_path: Description
    @param tile_x: Description
    @param tile_y: Description
    @param img_w: Description
    @param img_h: Description

    @returns List[str] with the data
    """
    new_labels = []

    if not os.path.exists(label_path):
        return new_labels

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, w, h = map(float, line.split())

        x_center = xc * img_w
        y_center = yc * img_h
        bw = w * img_w
        bh = h * img_h

        x1 = x_center - bw / 2
        y1 = y_center - bh / 2
        x2 = x_center + bw / 2
        y2 = y_center + bh / 2

        ix1 = max(x1, tile_x)
        iy1 = max(y1, tile_y)
        ix2 = min(x2, tile_x + TILE_SIZE)
        iy2 = min(y2, tile_y + TILE_SIZE)

        if ix1 >= ix2 or iy1 >= iy2:
            continue

        ix1 -= tile_x
        iy1 -= tile_y
        ix2 -= tile_x
        iy2 -= tile_y

        bw_new = ix2 - ix1
        bh_new = iy2 - iy1
        cx_new = ix1 + bw_new / 2
        cy_new = iy1 + bh_new / 2

        new_labels.append(
            f"{int(cls)} "
            f"{cx_new / TILE_SIZE} "
            f"{cy_new / TILE_SIZE} "
            f"{bw_new / TILE_SIZE} "
            f"{bh_new / TILE_SIZE}\n"
        )

    return new_labels


def main():
    args = get_args()

    images_routes, labels_routes = get_routes(args.data)
    os.makedirs(args.output_data, exist_ok=True)

    for i in range(len(images_routes)):
        image_dir = images_routes[i]
        labels_dir = labels_routes[i]

        images = os.listdir(image_dir)

        for img_route in images:
            if os.path.splitext(img_route)[1] == ".npy": continue
            # do not forget of the entire relative path
            img_route = os.path.join(image_dir, img_route)

            img = cv2.imread(img_route)
            # check the image exists
            assert img is not None,  f"{img_route} could not be processed. Be sure it does exists in the directory during the execution of the program\n"

            h, w = img.shape[:2]
            base_name, ext = os.path.splitext(os.path.basename(img_route))

            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            # for the tiling process, the counter starts on (x, y) = 0. For each tile, the cutting process is exatcly: [x: X +- 640px, y: Y +. 640px].
            # Knowing that the images from the dataset are (1920, 1080)
            # x = 0, 480, 960, 1280
            # y = 0, 480
            # as the 25% of 640 is 160px -> 160 + 480 = 640.
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

            tile_id = 0
            for y in y_starts:
                for x in x_starts:
                    tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]
                    tile_name = f"{base_name}_tile_{tile_id}{ext}"
                    cv2.imwrite(os.path.join(args.output_data, tile_name), tile)

                    tile_labels = process_labels(label_path, x, y, w, h)
                    if tile_labels:
                        tile_label_name = f"{base_name}_tile_{tile_id}.txt"
                        with open(os.path.join(args.output_data, tile_label_name), "w") as f:
                            f.writelines(tile_labels)

                    tile_id += 1
    return 0


if __name__ == '__main__':
    main()
