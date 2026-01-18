from ultralytics import YOLO
import cv2
import os
import torch
from torchvision.ops import nms

MODEL_ROUTE = (
    'results_yolo_tiled_v1/yolov8n_e200_b16_s42_box5.0_f203a/weights/best.pt'
)
IMAGE_TO_PREDICT = 'data/DJI_20251205101055_0007_D.MP4'
CONF_THRESHOLD = 0.31
CONF_IOU = 0.5

# Text config
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 2
color = (0, 0, 255)
thickness = 3

# Load model
model = YOLO(MODEL_ROUTE)

GRID_SIZE = 640

os.makedirs('debug_tiles', exist_ok=True)


def merge_adjacent_boxes(boxes, scores, classes, margin=5):
    """
    Merges two boxes if they are adyacent each other. It can apply a margin to reduce errors
    """
    if len(boxes) == 0:
        return boxes, scores, classes

    boxes = boxes.clone()
    scores = scores.clone()
    classes = classes.clone()

    merged = []
    merged_scores = []
    merged_classes = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue

        current_box = boxes[i].clone()
        current_score = scores[i]
        current_class = classes[i]
        group = [i]

        # Search for adyacent boxes
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue

            if are_adjacent(current_box, boxes[j], margin):
                # expand the current box so the new one can be merged
                current_box[0] = min(current_box[0], boxes[j][0])  # x1
                current_box[1] = min(current_box[1], boxes[j][1])  # y1
                current_box[2] = max(current_box[2], boxes[j][2])  # x2
                current_box[3] = max(current_box[3], boxes[j][3])  # y2

                # the maximum score is taken because both predictions are correct
                # thus both are valid. We take the best one.
                current_score = max(current_score, scores[j])
                group.append(j)
                used.add(j)

        merged.append(current_box)
        merged_scores.append(current_score)
        merged_classes.append(current_class)
        used.add(i)

    return (
        torch.stack(merged),
        torch.tensor(merged_scores),
        torch.tensor(merged_classes),
    )


def are_adjacent(box1, box2, margin=5):
    """
    Verifies if both boxes are adyacent.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # boxes are adyacent if they are near one axis and it solapes the other one
    horizontal_close = (
        abs(x2_1 - x1_2) <= margin  # right border box1 close left box2
        or abs(x2_2 - x1_1) <= margin  # right border box2 close left box1
    )

    vertical_close = (
        abs(y2_1 - y1_2) <= margin  # bottom border box1 close upper box2
        or abs(y2_2 - y1_1) <= margin  # bottom border box2 close upper box1
    )

    horizontal_overlap = not (x2_1 < x1_2 - margin or x2_2 < x1_1 - margin)
    vertical_overlap = not (y2_1 < y1_2 - margin or y2_2 < y1_1 - margin)

    return (horizontal_close and vertical_overlap) or (
        vertical_close and horizontal_overlap
    )


def generate_grid(image, grid_size=640):
    """Divides the image in a grid of n x n. No overlapping"""
    h, w = image.shape[:2]
    grids = []
    offsets = []

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            grid = image[y : y + grid_size, x : x + grid_size]

            if grid.shape[0] < grid_size or grid.shape[1] < grid_size:
                padded = cv2.copyMakeBorder(
                    grid,
                    0,
                    grid_size - grid.shape[0],
                    0,
                    grid_size - grid.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )
                grids.append(padded)
            else:
                grids.append(grid)

            offsets.append((x, y))

    return grids, offsets


def process_frame_with_grids(frame, model, conf_threshold, save_debug=False):
    """
    Process a frame converted into a grid. The inference processes all grids
    @param frame: cv2 Object
    """
    grids, offsets = generate_grid(frame, GRID_SIZE)

    # BATCH INFERENCE
    batch_results = model(grids, conf=conf_threshold, verbose=False)

    all_boxes = []
    all_scores = []
    all_classes = []

    for idx, (r, (ox, oy)) in enumerate(zip(batch_results, offsets)):
        if save_debug:
            grid_with_boxes = grids[idx].copy()

        if r.boxes is None or len(r.boxes) == 0:
            if save_debug:
                cv2.imwrite(
                    f'debug_tiles/grid_{idx}_predicted.png', grid_with_boxes
                )
            continue

        boxes = r.boxes.xyxy.cpu()
        scores = r.boxes.conf.cpu()
        classes = r.boxes.cls.cpu()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box

            if save_debug:
                cv2.rectangle(
                    grid_with_boxes,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    grid_with_boxes,
                    f'{score:.2f}',
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            all_boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
            all_scores.append(score)
            all_classes.append(cls)

        if save_debug:
            cv2.imwrite(
                f'debug_tiles/grid_{idx}_predicted.png', grid_with_boxes
            )

    if len(all_boxes) == 0:
        return [], [], []

    boxes = torch.tensor(all_boxes)
    scores = torch.tensor(all_scores)
    classes = torch.tensor(all_classes)

    # merge boxes that are adjacent - this happens because of how our grid-based
    # inference work
    boxes, scores, classes = merge_adjacent_boxes(
        boxes, scores, classes, margin=5
    )

    # apply nms to assure that no duplicates are remained
    keep = nms(boxes, scores, iou_threshold=0.3)

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    return boxes, scores, classes


def main():
    # If it's an image → just render once
    if not IMAGE_TO_PREDICT.lower().endswith('.mp4'):
        img = cv2.imread(IMAGE_TO_PREDICT)
        output = img.copy()

        boxes, scores, classes = process_frame_with_grids(
            img, model, CONF_THRESHOLD, save_debug=True
        )

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.putText(
            output,
            f"Cardilla's Number: {len(boxes)}",
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imwrite('output.png', output)
        cv2.imshow('YOLO Prediction', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If it's a video → process frame by frame
    else:
        cap = cv2.VideoCapture(IMAGE_TO_PREDICT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        frame_count = 0

        print(f'Processing video: {total_frames} frames at {fps} FPS')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, scores, classes = process_frame_with_grids(
                frame, model, CONF_THRESHOLD
            )

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.putText(
                frame,
                f"Cardilla's Number: {len(boxes)}",
                org,
                font,
                fontScale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            out.write(frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f'Processed {frame_count}/{total_frames} frames ({100 * frame_count / total_frames:.1f}%)'
                )

            cv2.imshow('YOLO Video Prediction', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

        print(f'Video saved: output.mp4')


if __name__ == '__main__':
    main()
