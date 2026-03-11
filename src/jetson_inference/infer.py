"""
RT-inference script for Jetson Orin Nano. Optimised with torch-tensorRT.
"""

import argparse
import os

import torch
import torch_tensorrt
import cv2
import numpy as np
from torchvision.ops import nms

from src.yolo.utils.tiling_utils import generate_grid


def get_args():
    parse = argparse.ArgumentParser('RT-Infer Script for Jetson')

    parse.add_argument(
        '--model',
        required=True,
        type=str,
        help='Route to the TorchScript model (.pt or .ts)',
    )
    parse.add_argument(
        '--compiled-model',
        required=False,
        type=str,
        default=None,
        help='Route to pre-compiled TensorRT model',
    )
    parse.add_argument(
        '--tiled',
        required=False,
        type=bool,
        default=True,
        help='The model uses tiling strategies or not',
    )
    parse.add_argument(
        '--data', required=True, type=str, help='The file (image or video) to process'
    )
    parse.add_argument(
        '--output',
        required=False,
        type=str,
        default='output.mp4',
        help='Output video file path (for videos)',
    )
    parse.add_argument(
        '--output-dir',
        required=False,
        type=str,
        default='.',
        help='Output directory for images',
    )

    return parse.parse_args()


def load_or_compile_model(model_path, compiled_model_path, tiled=True):
    """
    Load model: if compiled version exists, use it; otherwise load and compile.
    Returns the model ready for inference.
    """
    # If pre-compiled model exists, use it
    if compiled_model_path and os.path.exists(compiled_model_path):
        print(f'[infer] :: Loading pre-compiled model from {compiled_model_path}')
        try:
            model = torch.jit.load(compiled_model_path).eval().cuda()
            return model
        except Exception as e:
            print(f'[infer] :: Error loading pre-compiled model: {e}')
            print('[infer] :: Will compile from original model...')

    # Load original model
    print(f'[infer] :: Loading model from {model_path}')
    try:
        model = torch.jit.load(model_path).eval().cuda()
    except Exception as e:
        print(f'[infer] :: Error loading model: {e}')
        return None

    # Compile with TensorRT if not already compiled
    print('[infer] :: Compiling with TensorRT...')
    try:
        if tiled:
            inputs = torch_tensorrt.Input((6, 3, 640, 640))
        else:
            inputs = torch_tensorrt.Input((1, 3, 1920, 1080))

        trt_model = torch_tensorrt.compile(
            model, inputs=[inputs], enabled_precisions={torch.float16}
        )

        # Save compiled model for future use
        if compiled_model_path:
            torch.jit.save(trt_model, compiled_model_path)
            print(f'[infer] :: Compiled model saved to {compiled_model_path}')

        return trt_model
    except Exception as e:
        print(f'[infer] :: Error during TensorRT compilation: {e}')
        print('[infer] :: Using non-compiled model')
        return model


def process_frame_direct(frame, model):
    """
    Process a single frame directly without tiling.
    Returns boxes in format [[x1, y1, x2, y2], ...]
    """
    height, width = frame.shape[:2]

    # Resize to expected input size (1920, 1080) if needed
    if (height, width) != (1080, 1920):
        frame_resized = cv2.resize(frame, (1920, 1080))
    else:
        frame_resized = frame.copy()

    # Convert to tensor
    frame_tensor = torch.from_numpy(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

    with torch.no_grad():
        detections = model(frame_tensor)

    # Parse detections
    boxes = []
    if detections is not None:
        if isinstance(detections, dict):
            det = detections.get('output', detections.get('boxes', None))
        elif isinstance(detections, (list, tuple)):
            det = detections[0] if len(detections) > 0 else None
        else:
            det = detections

        if det is not None and len(det) > 0:
            for detection in det:
                x1, y1, x2, y2 = (
                    int(detection[0]),
                    int(detection[1]),
                    int(detection[2]),
                    int(detection[3]),
                )
                boxes.append([x1, y1, x2, y2])

    return boxes


def process_frame_tiled(frame, model, grid_size=640):
    """
    Process a frame using tiling strategy. Divides frame into 640x640 tiles,
    processes each tile, adjusts coordinates, and applies NMS.
    Returns boxes in format [[x1, y1, x2, y2], ...]
    """
    grids, offsets = generate_grid(frame, grid_size=grid_size)

    all_boxes = []
    all_scores = []

    # Process grids in batches of max 6 (model input size)
    batch_size = 6
    for batch_idx in range(0, len(grids), batch_size):
        batch_grids = grids[batch_idx : batch_idx + batch_size]
        batch_offsets = offsets[batch_idx : batch_idx + batch_size]

        # Pad batch to 6 if needed
        while len(batch_grids) < batch_size:
            batch_grids.append(np.zeros((grid_size, grid_size, 3), dtype=np.uint8))
            batch_offsets.append((0, 0))

        # Convert batch to tensor
        batch_tensor = []
        for grid in batch_grids:
            grid_tensor = torch.from_numpy(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
            grid_tensor = grid_tensor.permute(2, 0, 1).float() / 255.0
            batch_tensor.append(grid_tensor)

        batch_tensor = torch.stack(batch_tensor).cuda()

        # Inference
        with torch.no_grad():
            detections = model(batch_tensor)

        # Parse detections and adjust coordinates
        if detections is not None:
            if isinstance(detections, dict):
                dets = detections.get('output', detections.get('boxes', None))
            elif isinstance(detections, (list, tuple)):
                dets = detections[0] if len(detections) > 0 else None
            else:
                dets = detections

            if dets is not None and len(dets) > 0:
                for grid_idx, (detection_batch, (ox, oy)) in enumerate(
                    zip(dets, batch_offsets[: len(dets)])
                ):
                    if (
                        isinstance(detection_batch, torch.Tensor)
                        and len(detection_batch) > 0
                    ):
                        for detection in detection_batch:
                            x1, y1, x2, y2 = (
                                float(detection[0]),
                                float(detection[1]),
                                float(detection[2]),
                                float(detection[3]),
                            )

                            # Adjust coordinates using offset
                            x1_adjusted = x1 + ox
                            y1_adjusted = y1 + oy
                            x2_adjusted = x2 + ox
                            y2_adjusted = y2 + oy

                            all_boxes.append(
                                [x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted]
                            )
                            # Assume confidence is 1.0 if not provided
                            all_scores.append(1.0)

    # Apply NMS to remove duplicates
    if len(all_boxes) > 0:
        boxes_tensor = torch.tensor(all_boxes)
        scores_tensor = torch.tensor(all_scores)

        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
        boxes_filtered = boxes_tensor[keep].cpu().numpy().astype(int).tolist()

        return boxes_filtered

    return []


def run_inference(model, input_data, tiled=True, output_video=None, output_dir='.'):
    """
    Run inference on video or image with proper tiled/non-tiled preprocessing.
    Draws bounding boxes on frames and saves output.
    Returns dict {frame_id: [[x1, y1, x2, y2], ...]}
    """
    is_video = input_data.lower().endswith('.mp4')
    r_boxes = {}

    if is_video:
        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            print(f'[infer] :: Error opening video file: {input_data}')
            return r_boxes

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f'[infer] :: Error creating video writer for {output_video}')
            cap.release()
            return r_boxes
    else:
        frame = cv2.imread(input_data)
        if frame is None:
            print(f'[infer] :: Error reading image file: {input_data}')
            return r_boxes
        height, width, _ = frame.shape
        os.makedirs(output_dir, exist_ok=True)

    frame_count = 0

    while True:
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_count > 0:
                break

        try:
            # Process frame depending on tiled strategy
            if tiled:
                boxes = process_frame_tiled(frame, model)
            else:
                boxes = process_frame_direct(frame, model)

            # Store boxes
            r_boxes[frame_count] = boxes

            # Draw boxes on frame
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        except Exception as e:
            print(f'[infer] :: Error during inference on frame {frame_count}: {e}')
            continue

        # Write or save frame
        if is_video:
            out.write(frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f'[infer] :: Processed {frame_count}/{total_frames} frames ({100 * frame_count / total_frames:.1f}%)'
                )
        else:
            output_path = os.path.join(output_dir, 'output.jpg')
            cv2.imwrite(output_path, frame)
            print(f'[infer] :: Image saved to {output_path}')
            frame_count += 1

    if is_video:
        out.release()
        cap.release()
        print(f'[infer] :: Video saved to {output_video}')

    print(f'[infer] :: Inference complete. Processed {frame_count} frames.')
    return r_boxes


def main():
    args = get_args()

    # Determine compiled model path if not provided
    compiled_model_path = args.compiled_model
    if compiled_model_path is None:
        compiled_model_path = args.model.replace('.pt', '_trt.ts').replace(
            '.ts', '_trt.ts'
        )

    # Load or compile model
    # Note: tiled (True) compiles with input (6, 3, 640, 640) for tiled inference
    #       no-tiled (False) compiles with input (1, 3, 1920, 1080) for full frame inference
    model = load_or_compile_model(args.model, compiled_model_path, args.tiled)
    if model is None:
        print('[infer] :: Failed to load/compile model. Exiting.')
        return

    # Run inference with visualization
    print(f'[infer] :: Starting inference... (tiled={args.tiled})')
    r_boxes = run_inference(
        model,
        args.data,
        tiled=args.tiled,
        output_video=args.output,
        output_dir=args.output_dir,
    )

    print(
        f'[infer] :: Done! Found {sum(len(v) for v in r_boxes.values())} total detections.'
    )
    if r_boxes and len(r_boxes) <= 5:
        print(f'[infer] :: Detections: {r_boxes}')


if __name__ == '__main__':
    main()
