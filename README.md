# Keras to TensorRT Examples #
This is a simple demonstration for running Keras model model on **Tensorflow with TensorRT integration**(TFTRT) or on TensorRT directly without invoking ["freeze_graph.py"](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

Note: Recommend that use NVIDIA Tensorflow docker image to run these examples. You can download the images from [NVIDIA NGC](https://www.nvidia.com/en-us/gpu-cloud/).

## Requirement ##
* Python (both 2 and 3 are ok)
* TensorRT (> 3.0)
* Tensorflow with TensorRT integration (> 1.7)
* Keras

if you want to run model on TensorRT directly, Pycuda is also needed:
* Pycuda (> 2017.1.1)

## Examples ##
[tftrt_example.py](/tftrt_example.py) demonstrates how to run Keras model on TFTRT. This approach supports both NCHW and NHWC format because Tensorflow can handles format issue.
```shell
$ python tftrt_example.py
```

[tftrt_resnet_example.py](/tftrt_resnet_example.py) demonstrates how to run Keras Applications ResNet50 on TFTRT.
```shell
$ python tftrt_resnet_example.py
```

[tftrt_multi_inputs_mutli_outputs_example.py](/tftrt_multi_inputs_mutli_outputs_example.py) demonstrates how to run a multi-input/output Keras model on TFTRT.
```shell
$ python tftrt_multi_inputs_mutli_outputs_example.py
```

[trt_example.py](/trt_example.py) demonstrates how to run Keras model on TensorRT which can achieve fastest speed.
Because TensorRT didn't fully support NHWC yet, this approach only suits NCHW format.
```shell
$ python trt_example.py
```

## Appendix ##

[get_mnist_model.py](/get_mnist_model.py) can generate needed Keras models with two different input formats, 
one for NCHW foramt, another one for NHWC format.

Note: the needed models were already provided in repo.
```shell
$ python get_mnist_model.py -h # shows help message
$ python get_mnist_model.py -f 0 # generates model for NCHW format
$ python get_mnist_model.py -f 1 # generates model for NHWC format
```

## Reference ##
* [TensorRT Integration Speeds Up TensorFlow Inference](https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/)
* [NVIDIA Tensorflow Release Note](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/overview.html#overview)
