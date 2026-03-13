import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import platform
import time

ENGINE_PATH = 'results_yolo_/results_yolo_lowFlight_v11/yolov8n_e200_b16_s42_box5.0_d59ef/weights/best.engine'
VIDEO_PATH = 'data/DJI_20260213113651_0017_D.MP4'
CONF_THRESHOLD = 0.5
TILED = False  # Set to True for tiled inference, False for full-frame inference
GRID_SIZE = 640
REQUESTED_BATCH_SIZE = 3


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


def get_input_output_names(engine):
    """Return first input/output tensor names from the engine."""
    input_name = None
    output_name = None
    for name in engine:
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT and input_name is None:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT and output_name is None:
            output_name = name

    if input_name is None or output_name is None:
        raise RuntimeError('No se pudieron identificar los tensores de entrada/salida.')

    return input_name, output_name


def get_engine_batch_size(engine, input_name):
    """Infer the batch size supported by the engine for the input tensor."""
    shape = tuple(engine.get_tensor_shape(input_name))
    if len(shape) >= 1 and shape[0] > 0:
        return int(shape[0])

    # Dynamic shape fallback: use max profile shape
    try:
        _, _, max_shape = engine.get_tensor_profile_shape(input_name, 0)
        if len(max_shape) >= 1 and max_shape[0] > 0:
            return int(max_shape[0])
    except Exception:
        pass

    raise RuntimeError(
        f'No se pudo inferir batch size del engine para {input_name}. shape={shape}'
    )


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


def preprocess_batch(frames):
    """Preprocess multiple frames into a batch. Returns shape (batch_size, 3, 640, 640)"""
    batch = []
    for frame in frames:
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        batch.append(img)
    batch = np.stack(batch, axis=0)
    return batch


def infer(context, inputs, outputs, stream, img_batch):
    """
    Perform inference on a batch of images.
    img_batch: numpy array of shape (batch_size, 3, 640, 640)
    """
    input_name = list(inputs.keys())[0]
    output_name = list(outputs.keys())[0]

    # Copy input to GPU
    img_flat = img_batch.ravel()
    if img_flat.size != inputs[input_name][0].size:
        raise ValueError(
            f'Batch incompatible con el engine. input_elems={img_flat.size}, '
            f'buffer_elems={inputs[input_name][0].size}'
        )
    np.copyto(inputs[input_name][0], img_flat)
    cuda.memcpy_htod_async(inputs[input_name][1], inputs[input_name][0], stream)

    for name, (_, device_mem) in inputs.items():
        context.set_tensor_address(name, int(device_mem))
    for name, (_, device_mem) in outputs.items():
        context.set_tensor_address(name, int(device_mem))

    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[output_name][0], outputs[output_name][1], stream)
    stream.synchronize()

    return outputs[output_name][0].copy()


def _build_engine_batch(valid_batch, engine_batch_size):
    """Pad a valid batch to engine batch size (fixed-shape engines)."""
    valid_count = valid_batch.shape[0]
    if valid_count == engine_batch_size:
        return valid_batch, valid_count

    pad_count = engine_batch_size - valid_count
    pad_tile = valid_batch[-1:, ...]
    pad_batch = np.repeat(pad_tile, pad_count, axis=0)
    return np.concatenate([valid_batch, pad_batch], axis=0), valid_count


def _split_output_per_item(output_batch, engine_batch_size):
    """Split flattened output into one chunk per batch item."""
    if output_batch.size % engine_batch_size != 0:
        raise ValueError(
            f'Output inválido: size={output_batch.size}, batch={engine_batch_size}'
        )

    elems_per_item = output_batch.size // engine_batch_size
    return [
        output_batch[i * elems_per_item : (i + 1) * elems_per_item]
        for i in range(engine_batch_size)
    ]


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


def process_grids_batched(
    grids,
    offsets,
    context,
    inputs,
    outputs,
    stream,
    orig_h,
    orig_w,
    engine_batch_size,
    requested_batch_size=REQUESTED_BATCH_SIZE,
    conf_threshold=CONF_THRESHOLD,
):
    """
    Process grids in batches and return combined detections.

    Args:
        grids: list of grid images
        offsets: list of (x, y) offsets for each grid
        engine_batch_size: fixed batch size expected by TensorRT engine
        requested_batch_size: desired number of valid tiles per chunk

    Returns:
        list of boxes in original frame coordinates
    """
    all_boxes = []
    chunk_size = min(requested_batch_size, engine_batch_size)

    # Process grids in batches
    for batch_start in range(0, len(grids), chunk_size):
        batch_end = min(batch_start + chunk_size, len(grids))
        batch_grids = grids[batch_start:batch_end]
        batch_offsets = offsets[batch_start:batch_end]

        # Preprocess batch
        valid_batch = preprocess_batch(batch_grids)
        engine_batch, valid_count = _build_engine_batch(valid_batch, engine_batch_size)

        # Inference
        output_batch = infer(context, inputs, outputs, stream, engine_batch)
        per_item_outputs = _split_output_per_item(output_batch, engine_batch_size)

        # Process only valid (non-padded) outputs
        for output, (tile_x, tile_y) in zip(
            per_item_outputs[:valid_count], batch_offsets
        ):
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

    # NMS on all detections
    if not all_boxes:
        return []

    rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in all_boxes]
    scores = [c for *_, c in all_boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, conf_threshold, nms_threshold=0.45)
    return [all_boxes[i] for i in indices.flatten()] if len(indices) > 0 else []


def main():
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    cap = None

    try:
        engine = load_engine(ENGINE_PATH)
        context = engine.create_execution_context()
        inputs, outputs, stream = allocate_buffers(engine)
        input_name, _ = get_input_output_names(engine)
        engine_batch_size = get_engine_batch_size(engine, input_name)

        if platform.machine() == 'aarch64':
            pipeline = (
                f'filesrc location={VIDEO_PATH} ! '
                'qtdemux ! '
                'h264parse ! '
                'nvv4l2decoder ! '
                'nvvidconv ! '
                'video/x-raw, format=BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=BGR ! '
                'appsink'
            )
            # maintain nvdec for gpu usage, ano not CPU (jetson may got fucked if we are using this)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        else:
            cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            raise RuntimeError(f'No se pudo abrir el video: {VIDEO_PATH}')

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        times = []
        frame_count = 0

        # warmup using engine batch size
        for _ in range(10):
            warmup_frames = [
                np.random.randint(0, 255, (GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
                for _ in range(engine_batch_size)
            ]
            warmup_batch = preprocess_batch(warmup_frames)
            infer(context, inputs, outputs, stream, warmup_batch)

        effective_chunk = min(REQUESTED_BATCH_SIZE, engine_batch_size)
        print(f'Mode: {"TILED" if TILED else "FULL-FRAME"}')
        print(f'Engine batch size: {engine_batch_size}')
        if TILED:
            print(
                f'Requested tiled batch: {REQUESTED_BATCH_SIZE} -> effective: {effective_chunk}'
            )
        print(f'Processing {VIDEO_PATH}...\n')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if TILED:
                grids, offsets = generate_grids(frame, grid_size=GRID_SIZE)
                t0 = time.perf_counter()
                boxes = process_grids_batched(
                    grids,
                    offsets,
                    context,
                    inputs,
                    outputs,
                    stream,
                    orig_h,
                    orig_w,
                    engine_batch_size=engine_batch_size,
                    requested_batch_size=REQUESTED_BATCH_SIZE,
                )
                t1 = time.perf_counter()
                times.append(t1 - t0)
            else:
                img = preprocess(frame)
                img_engine, _ = _build_engine_batch(img, engine_batch_size)
                t0 = time.perf_counter()
                output_batch = infer(context, inputs, outputs, stream, img_engine)
                t1 = time.perf_counter()
                times.append(t1 - t0)
                output = _split_output_per_item(output_batch, engine_batch_size)[0]
                boxes = postprocess(output.reshape(5, -1), orig_h, orig_w)

            frame = draw(frame, boxes)

            cv2.imshow('TensorRT Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f'Processed {frame_count} frames...')

        mean_latency = sum(times) / len(times)
        fps = 1.0 / mean_latency
        print('\n=== Results ===')
        print(f'Mode: {"TILED" if TILED else "FULL-FRAME"}')
        print(f'Frames processed: {frame_count}')
        print(f'Mean latency: {mean_latency:.6f} s')
        print(f'FPS: {fps:.2f}')
        print(f'Min latency: {min(times):.6f} s')
        print(f'Max latency: {max(times):.6f} s')
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        ctx.pop()


if __name__ == '__main__':
    main()
