import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import time

ENGINE_PATH = (
    'results_yolo_lowFlight_v11/yolov8n_e200_b16_s42_box5.0_d59ef/weights/best.engine'
)
VIDEO_PATH = 'data/DJI_20260213114207_0018_D.MP4'
CONF_THRESHOLD = 0.5
TILED = False  # Set to True for tiled inference, False for full-frame inference
GRID_SIZE = 640


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers(engine):
    inputs = {}
    outputs = {}
    stream = cuda.Stream()
    for name in engine:
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs[name] = (host_mem, device_mem)
        else:
            outputs[name] = (host_mem, device_mem)
    return inputs, outputs, stream


def generate_grids(image, grid_size=640):
    """Divides the image into a grid of n x n tiles. No overlapping."""
    h, w = image.shape[:2]
    grids = []
    offsets = []

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            grid = image[y : y + grid_size, x : x + grid_size]

            # Pad if necessary
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


def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


def infer(context, inputs, outputs, stream, img_batch):
    """
    Perform inference on a batch of images.
    img_batch: numpy array of shape (batch_size, 3, 640, 640)
    """
    input_name = list(inputs.keys())[0]

    # Copy input to GPU
    img_flat = img_batch.ravel()
    np.copyto(inputs[input_name][0], img_flat)
    cuda.memcpy_htod_async(inputs[input_name][1], inputs[input_name][0], stream)

    for name, (_, device_mem) in inputs.items():
        context.set_tensor_address(name, int(device_mem))
    for name, (_, device_mem) in outputs.items():
        context.set_tensor_address(name, int(device_mem))

    context.execute_async_v3(stream_handle=stream.handle)
    output_name = list(outputs.keys())[0]
    cuda.memcpy_dtoh_async(outputs[output_name][0], outputs[output_name][1], stream)
    stream.synchronize()

    return outputs[output_name][0]


def postprocess(output, orig_h, orig_w, conf_threshold=CONF_THRESHOLD):
    # output shape: (batch_size, 5, 8400) -> cx, cy, w, h, conf
    if output.ndim == 2:
        # Single frame case: (5, 8400)
        preds = output.reshape(5, 8400)
        batch_size = 1
    else:
        # Batch case: (batch_size, 5, 8400)
        preds = output.reshape(output.shape[0], 5, -1)
        batch_size = output.shape[0]

    all_boxes = []

    for b in range(batch_size):
        if output.ndim == 2:
            frame_preds = preds
        else:
            frame_preds = preds[b]

        for i in range(frame_preds.shape[1]):
            conf = frame_preds[4, i]
            if conf < conf_threshold:
                continue

            cx, cy, w, h = (
                frame_preds[0, i],
                frame_preds[1, i],
                frame_preds[2, i],
                frame_preds[3, i],
            )

            # Scale from 640x640 normalized to original frame dimensions
            x1 = int((cx - w / 2) / 640 * orig_w)
            y1 = int((cy - h / 2) / 640 * orig_h)
            x2 = int((cx + w / 2) / 640 * orig_w)
            y2 = int((cy + h / 2) / 640 * orig_h)

            all_boxes.append((x1, y1, x2, y2, float(conf)))

    # NMS
    if not all_boxes:
        return []

    rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in all_boxes]
    scores = [c for *_, c in all_boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, conf_threshold, nms_threshold=0.45)
    return [all_boxes[i] for i in indices.flatten()] if len(indices) > 0 else []


def postprocess_tiled(outputs, offsets, orig_h, orig_w, conf_threshold=CONF_THRESHOLD):
    """
    Postprocess batch of tiled outputs.
    outputs: list of output arrays from each tile
    offsets: list of (x, y) offsets for each tile
    """
    all_boxes = []

    # Process each tile
    for output, (tile_x, tile_y) in zip(outputs, offsets):
        # output shape: (5, 8400)
        preds = output.reshape(5, -1)

        for i in range(preds.shape[1]):
            conf = preds[4, i]
            if conf < conf_threshold:
                continue

            cx, cy, w, h = preds[0, i], preds[1, i], preds[2, i], preds[3, i]

            # Convert from tile normalized coords to original frame coords
            x1 = int(tile_x + (cx - w / 2))
            y1 = int(tile_y + (cy - h / 2))
            x2 = int(tile_x + (cx + w / 2))
            y2 = int(tile_y + (cy + h / 2))

            # Clamp to frame boundaries
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            if x1 < x2 and y1 < y2:
                all_boxes.append((x1, y1, x2, y2, float(conf)))

    # NMS
    if not all_boxes:
        return []

    rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in all_boxes]
    scores = [c for *_, c in all_boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, conf_threshold, nms_threshold=0.45)
    return [all_boxes[i] for i in indices.flatten()] if len(indices) > 0 else []


def draw(frame, boxes):
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'{conf:.2f}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame


def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    inputs, outputs, stream = allocate_buffers(engine)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir el video: {VIDEO_PATH}')

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    times = []
    frame_count = 0

    # warmup
    for _ in range(10):
        if TILED:
            warmup_frame = np.random.randint(
                0, 255, (GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8
            )
            img = preprocess(warmup_frame)
        else:
            warmup_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img = preprocess(warmup_frame)
        infer(context, inputs, outputs, stream, img)

    print(f'Mode: {"TILED" if TILED else "FULL-FRAME"}')
    print(f'Processing {VIDEO_PATH}...\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if TILED:
            grids, offsets = generate_grids(frame, grid_size=GRID_SIZE)
            t0 = time.perf_counter()

            # Process each tile individually (TensorRT engine typically has batch_size=1)
            outputs_list = []
            for grid in grids:
                img = preprocess(grid)
                output = infer(context, inputs, outputs, stream, img)
                outputs_list.append(output)

            t1 = time.perf_counter()
            times.append(t1 - t0)
            boxes = postprocess_tiled(outputs_list, offsets, orig_h, orig_w)
        else:
            img = preprocess(frame)
            t0 = time.perf_counter()
            output = infer(context, inputs, outputs, stream, img)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            boxes = postprocess(output, orig_h, orig_w)

        frame = draw(frame, boxes)

        cv2.imshow('TensorRT Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f'Processed {frame_count} frames...')

    cap.release()
    cv2.destroyAllWindows()

    mean_latency = sum(times) / len(times)
    fps = 1.0 / mean_latency
    print('\n=== Results ===')
    print(f'Mode: {"TILED" if TILED else "FULL-FRAME"}')
    print(f'Frames processed: {frame_count}')
    print(f'Mean latency: {mean_latency:.6f} s')
    print(f'FPS: {fps:.2f}')
    print(f'Min latency: {min(times):.6f} s')
    print(f'Max latency: {max(times):.6f} s')


if __name__ == '__main__':
    main()
