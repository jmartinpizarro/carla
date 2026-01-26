# TensorRT Inference in C++ #

Because of the need to implement an efficient pipeline (where it is allowed a latency of a few docens of miliseconds),
TensorRT with the Python API is the perfect option. Assuming every system will be using, at least, an Nvidia Jetson Nano
Orin, this Python pipeline has been designed for real-time detection with an already-trained CARLA-model.

## Development ##

Simulating a controlled environment, exists two types of *dockerfiles*:

- `Dockerfile.jetson`: the Dockerfile used for RT-inference on the Jetson Orin Nano. Uses a `l4t-cuda` image.
- `Dockerfile.pc`: the Dockerfile used for development (x86 environment). Uses a classic `nvidia/cuda` image.

For the RT-inference, it is used Python TensorRT and PyTorch with CUDA 12.6.

```bash
# for the Jetson
docker build -f Dockerfile.jetson -t carla-jetson-infer .
# for development
docker build -f Dockerfile.pc -t carla-development-infer .
```

For running the containers:

```bash
# for the Jetson
docker run --rm -it --runtime nvidia -v /home/jmartinpizarro/videos:/videos -v /home/jmartinpizarro/results:/results carla-jetson-infer /bin/bash
# for development
docker run --rm -it --runtime nvidia -v /home/jmartinpizarro/videos:/videos -v /home/jmartinpizarro/results:/results carla-development-infer /bin/bash
```

Inside the container, run the following command for checking that everything works fine:

```bash
python
import torch
import torch_tensorrt
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```