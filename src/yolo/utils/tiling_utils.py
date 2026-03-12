"""
Contains functions used for tiling-based inference
"""

import cv2
import time
import torch
from torchvision.ops import nms


def _sync_cuda_for_timing():
    """Synchronize CUDA to avoid underestimating asynchronous GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def process_frame_with_grids(
    frame,
    model,
    conf_threshold=0.4,
    save_debug=False,
    grid_size=640,
    batch_size=2,
):
    """
    Process a frame converted into a grid. Inference is executed in mini-batches
    of tiles and returns the latency.
    @param frame: cv2 Object
    """
    grids, offsets = generate_grid(frame, grid_size=grid_size)
    batch_size = max(1, int(batch_size))

    t_latency: float = 0.0

    # BATCH INFERENCE (chunked)
    _sync_cuda_for_timing()
    t_inf_start = time.perf_counter()

    batch_results = []
    for batch_start in range(0, len(grids), batch_size):
        batch_end = min(batch_start + batch_size, len(grids))
        grid_batch = grids[batch_start:batch_end]
        partial_results = model(grid_batch, conf=conf_threshold, verbose=False)
        batch_results.extend(partial_results)

    _sync_cuda_for_timing()
    t_inf_end = time.perf_counter()
    t_inf_latency = t_inf_end - t_inf_start

    all_boxes = []
    all_scores = []
    all_classes = []
    all_tile_indices = []

    # postprocessing conversions
    t_postprocessing_start = time.perf_counter()
    for idx, (r, (ox, oy)) in enumerate(zip(batch_results, offsets)):
        if save_debug:
            grid_with_boxes = grids[idx].copy()

        if r.boxes is None or len(r.boxes) == 0:
            if save_debug:
                cv2.imwrite(f'debug_tiles/grid_{idx}_predicted.png', grid_with_boxes)
            continue

        # Keep tensors on-device during accumulation; move once to CPU at the end.
        boxes = r.boxes.xyxy
        scores = r.boxes.conf
        classes = r.boxes.cls

        offset = boxes.new_tensor([ox, oy, ox, oy])
        boxes = boxes + offset

        if save_debug:
            boxes_cpu = boxes.detach().cpu()
            scores_cpu = scores.detach().cpu()
            for box, score in zip(boxes_cpu, scores_cpu):
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(
                    grid_with_boxes,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    grid_with_boxes,
                    f'{float(score):.2f}',
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)

        tile_index = (oy // grid_size, ox // grid_size)
        all_tile_indices.extend([tile_index] * int(boxes.shape[0]))

        if save_debug:
            cv2.imwrite(f'debug_tiles/grid_{idx}_predicted.png', grid_with_boxes)

    if len(all_boxes) == 0:
        t_postprocessing_latency = time.perf_counter() - t_postprocessing_start
        return [], [], [], t_inf_latency + t_postprocessing_latency

    boxes = torch.cat(all_boxes, dim=0).cpu()
    scores = torch.cat(all_scores, dim=0).cpu()
    classes = torch.cat(all_classes, dim=0).cpu()

    # merge boxes that are adjacent - this happens because of how our grid-based
    # inference work
    boxes, scores, classes = merge_adjacent_boxes_across_tiles(
        boxes,
        scores,
        classes,
        all_tile_indices,
        grid_size=grid_size,
        margin=5,
    )

    # apply nms to assure that no duplicates are remained
    keep = nms(boxes, scores, iou_threshold=0.3)

    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    t_postprocessing_end = time.perf_counter()
    t_postprocessing_latency = t_postprocessing_end - t_postprocessing_start

    t_latency = t_inf_latency + t_postprocessing_latency

    return boxes, scores, classes, t_latency


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


def are_adjacent_across_tiles(box1, tile1, box2, tile2, grid_size=640, margin=5):
    """Check adjacency only across neighboring tiles (4-neighbors)."""
    r1, c1 = tile1
    r2, c2 = tile2

    dr = r2 - r1
    dc = c2 - c1

    if abs(dr) + abs(dc) != 1:
        return False

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    tile1_x0 = c1 * grid_size
    tile1_y0 = r1 * grid_size
    tile1_x1 = tile1_x0 + grid_size
    tile1_y1 = tile1_y0 + grid_size

    tile2_x0 = c2 * grid_size
    tile2_y0 = r2 * grid_size
    tile2_x1 = tile2_x0 + grid_size
    tile2_y1 = tile2_y0 + grid_size

    if dc == 1:
        left_touch = x2_1 >= tile1_x1 - margin
        right_touch = x1_2 <= tile2_x0 + margin
        overlap = not (y2_1 < y1_2 - margin or y2_2 < y1_1 - margin)
        return left_touch and right_touch and overlap

    if dc == -1:
        left_touch = x2_2 >= tile2_x1 - margin
        right_touch = x1_1 <= tile1_x0 + margin
        overlap = not (y2_1 < y1_2 - margin or y2_2 < y1_1 - margin)
        return left_touch and right_touch and overlap

    if dr == 1:
        top_touch = y2_1 >= tile1_y1 - margin
        bottom_touch = y1_2 <= tile2_y0 + margin
        overlap = not (x2_1 < x1_2 - margin or x2_2 < x1_1 - margin)
        return top_touch and bottom_touch and overlap

    if dr == -1:
        top_touch = y2_2 >= tile2_y1 - margin
        bottom_touch = y1_1 <= tile1_y0 + margin
        overlap = not (x2_1 < x1_2 - margin or x2_2 < x1_1 - margin)
        return top_touch and bottom_touch and overlap

    return False


def merge_adjacent_boxes_across_tiles(
    boxes, scores, classes, tile_indices, grid_size=640, margin=5
):
    """
    Merges adyacent boxes only across neighboring tiles to avoid snowball growth.
    """
    if len(boxes) == 0:
        return boxes, scores, classes

    boxes = boxes.clone()
    scores = scores.clone()
    classes = classes.clone()

    visited = set()
    merged = []
    merged_scores = []
    merged_classes = []

    for i in range(len(boxes)):
        if i in visited:
            continue

        stack = [i]
        component = []
        visited.add(i)

        while stack:
            idx = stack.pop()
            component.append(idx)

            for j in range(len(boxes)):
                if j in visited:
                    continue
                if classes[j] != classes[idx]:
                    continue
                if are_adjacent_across_tiles(
                    boxes[idx],
                    tile_indices[idx],
                    boxes[j],
                    tile_indices[j],
                    grid_size=grid_size,
                    margin=margin,
                ):
                    visited.add(j)
                    stack.append(j)

        comp_boxes = boxes[component]
        x1 = torch.min(comp_boxes[:, 0])
        y1 = torch.min(comp_boxes[:, 1])
        x2 = torch.max(comp_boxes[:, 2])
        y2 = torch.max(comp_boxes[:, 3])
        merged.append(torch.tensor([x1, y1, x2, y2]))
        merged_scores.append(torch.max(scores[component]))
        merged_classes.append(classes[component[0]])

    return (
        torch.stack(merged),
        torch.tensor(merged_scores),
        torch.tensor(merged_classes),
    )


def calculate_density_percentage(frame, boxes) -> float:
    """
    Calculates the percentage of occupied area by the prediction boxes in a given frame

    :param frame: the frame (CV2 image) to process
    :param boxes: boxes in YOLO output transformed into absolute position
    """
    try:
        h, w = frame.shape[:2]
    except Exception:
        raise AttributeError(
            'There is no .shape attribute in the frame. The frame must be CV2 object.\n'
        )

    # The idea is to calculate the percentage of pixels marked as a 'cardilla' with
    # respect to the total amount of pixels in the entire frame

    total_image_area = h * w

    boxes_area = []
    for box in boxes:
        (
            x1,
            y1,
            x2,
            y2,
        ) = box
        # area of a rectangle = b * h
        area = (x2 - x1) * (y2 - y1)
        boxes_area.append(area)

    total_occupied_area = sum(boxes_area)

    return total_occupied_area / total_image_area
