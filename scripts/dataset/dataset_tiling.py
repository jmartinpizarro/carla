"""
This script contains the code for tiling the original dataset. Given the original labelled dataset (1920x1080 aspect ratio)
samples are relatively small when converted into images of 640px during training, reducing precision.
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

SPLITS = ['train', 'test', 'valid']
CATEGORIES = ['images', 'labels']


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
    image_routes = []
    label_routes = []

    for split in SPLITS:
        images_dir = os.path.join(data_route, split, 'images')
        labels_dir = os.path.join(data_route, split, 'labels')
        image_routes.append(images_dir)
        label_routes.append(labels_dir)

    return image_routes, label_routes


def ensure_output_structure(output_route: str) -> None:
    """
    Creates the train/test/valid + images/labels structure inside output_route,
    mirroring the original dataset layout so tiles land in the same split
    the source image belonged to.
    """
    for split in SPLITS:
        for category in CATEGORIES:
            os.makedirs(os.path.join(output_route, split, category), exist_ok=True)


def read_yolo_labels(label_path: str) -> List[List[float]]:
    """
    Reads a YOLO-format label file (class x_center y_center width height, normalized).
    Returns an empty list if the file doesn't exist (image with no annotations).
    """
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            boxes.append([cls, x, y, w, h])
    return boxes


def clip_and_convert_box(box, img_w, img_h, x0, y0, tile_w, tile_h):
    """
    Converts a normalized YOLO box (relative to the full image) into a normalized
    YOLO box relative to a tile [x0, x0+tile_w) x [y0, y0+tile_h).
    Returns None if the box has no meaningful overlap with the tile.
    """
    cls, xc, yc, w, h = box

    # absolute pixel coords in the original image
    abs_xc = xc * img_w
    abs_yc = yc * img_h
    abs_w = w * img_w
    abs_h = h * img_h

    x_min = abs_xc - abs_w / 2
    y_min = abs_yc - abs_h / 2
    x_max = abs_xc + abs_w / 2
    y_max = abs_yc + abs_h / 2

    # clip to tile bounds
    tile_x_min = max(x_min, x0)
    tile_y_min = max(y_min, y0)
    tile_x_max = min(x_max, x0 + tile_w)
    tile_y_max = min(y_max, y0 + tile_h)

    new_w = tile_x_max - tile_x_min
    new_h = tile_y_max - tile_y_min

    # discard boxes that barely overlap the tile (avoids tiny fragment boxes)
    if new_w <= 2 or new_h <= 2:
        return None

    # re-center relative to the tile, then normalize
    new_xc = (tile_x_min + tile_x_max) / 2 - x0
    new_yc = (tile_y_min + tile_y_max) / 2 - y0

    return [
        cls,
        new_xc / tile_w,
        new_yc / tile_h,
        new_w / tile_w,
        new_h / tile_h,
    ]


def tile_image_and_labels(
    image_path: str,
    label_path: str,
    out_images_dir: str,
    out_labels_dir: str,
) -> None:
    """
    Slices a single image into TILE_SIZE x TILE_SIZE tiles (with OVERLAP),
    recomputes YOLO labels per tile, and writes both into the given
    output split folders.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    img_h, img_w = img.shape[:2]
    boxes = read_yolo_labels(label_path)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    tile_idx = 0
    for y0 in range(0, max(img_h - OVERLAP, 1), STRIDE):
        for x0 in range(0, max(img_w - OVERLAP, 1), STRIDE):
            # clamp tile so it never goes outside the image
            x_end = min(x0 + TILE_SIZE, img_w)
            y_end = min(y0 + TILE_SIZE, img_h)
            x_start = max(x_end - TILE_SIZE, 0)
            y_start = max(y_end - TILE_SIZE, 0)

            tile_w = x_end - x_start
            tile_h = y_end - y_start

            tile_img = img[y_start:y_end, x_start:x_end]

            tile_boxes = []
            for box in boxes:
                new_box = clip_and_convert_box(
                    box, img_w, img_h, x_start, y_start, tile_w, tile_h
                )
                if new_box is not None:
                    tile_boxes.append(new_box)

            tile_name = f"{base_name}_tile{tile_idx}"
            out_img_path = os.path.join(out_images_dir, tile_name + ".jpg")
            out_label_path = os.path.join(out_labels_dir, tile_name + ".txt")

            cv2.imwrite(out_img_path, tile_img)
            with open(out_label_path, 'w') as f:
                for cls, xc, yc, w, h in tile_boxes:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            tile_idx += 1

            if x_end == img_w:
                break
        if y_end == img_h:
            break


def process_dataset(data_route: str, output_route: str) -> None:
    ensure_output_structure(output_route)
    image_routes, label_routes = get_routes(data_route)

    for split, images_dir, labels_dir in zip(SPLITS, image_routes, label_routes):
        if not os.path.isdir(images_dir):
            print(f"[INFO] Skipping split '{split}', no images dir at {images_dir}")
            continue

        out_images_dir = os.path.join(output_route, split, 'images')
        out_labels_dir = os.path.join(output_route, split, 'labels')

        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"[INFO] Processing split '{split}': {len(image_files)} images")

        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            tile_image_and_labels(
                image_path, label_path, out_images_dir, out_labels_dir
            )


def main():
    args = get_args()
    process_dataset(args.data, args.output_data)


if __name__ == '__main__':
    main()
