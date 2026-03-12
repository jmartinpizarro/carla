import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import time

ENGINE_PATH = 'results_yolo_/results_yolo_tiled_v2/yolo11n_e200_b16_s42_box5.0_8b803/weights/best.engine'
VIDEO_PATH = 'data/DJI_20260213113651_0017_D.MP4'


def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        shape = engine.get_binding_shape(binding)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream


def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


def infer(context, bindings, inputs, outputs, stream, img):
    np.copyto(inputs[0][0], img.ravel())

    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)

    stream.synchronize()

    return outputs[0][0]


def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    cap = cv2.VideoCapture(VIDEO_PATH)

    times = []
    frame_count = 0

    # warmup
    for _ in range(10):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = preprocess(img)
        infer(context, bindings, inputs, outputs, stream, img)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess(frame)

        t0 = time.perf_counter()
        infer(context, bindings, inputs, outputs, stream, img)
        t1 = time.perf_counter()

        times.append(t1 - t0)
        frame_count += 1

    cap.release()

    mean_latency = sum(times) / len(times)
    fps = 1.0 / mean_latency

    print(f'Frames processed: {frame_count}')
    print(f'Mean latency: {mean_latency:.6f} s')
    print(f'FPS: {fps:.2f}')


if __name__ == '__main__':
    main()
