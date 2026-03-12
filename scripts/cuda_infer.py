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


def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


def infer(context, inputs, outputs, stream, img):
    input_name = list(inputs.keys())[0]
    np.copyto(inputs[input_name][0], img.ravel())
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
    # output shape: (5, 8400) -> cx, cy, w, h, conf
    preds = output.reshape(5, 8400)
    boxes = []

    for i in range(8400):
        conf = preds[4, i]
        if conf < conf_threshold:
            continue

        cx, cy, w, h = preds[0, i], preds[1, i], preds[2, i], preds[3, i]

        # de coordenadas normalizadas 640x640 a píxeles del frame original
        x1 = int((cx - w / 2) / 640 * orig_w)
        y1 = int((cy - h / 2) / 640 * orig_h)
        x2 = int((cx + w / 2) / 640 * orig_w)
        y2 = int((cy + h / 2) / 640 * orig_h)

        boxes.append((x1, y1, x2, y2, float(conf)))

    # NMS
    if not boxes:
        return []

    rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in boxes]
    scores = [c for *_, c in boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, CONF_THRESHOLD, nms_threshold=0.45)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []


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
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = preprocess(img)
        infer(context, inputs, outputs, stream, img)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

    cap.release()
    cv2.destroyAllWindows()

    mean_latency = sum(times) / len(times)
    fps = 1.0 / mean_latency
    print(f'Frames processed: {frame_count}')
    print(f'Mean latency: {mean_latency:.6f} s')
    print(f'FPS: {fps:.2f}')


if __name__ == '__main__':
    main()
