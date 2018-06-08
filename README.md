# Keras to TensorRT Examples #
This repo shows how to run Keras model on TensorRT

## Requirement ##
* Python 3
* Pycuda
* TensorRT 4.0
* Tensorflow 1.7 with TensorRT integration
* Keras

Please make sure that Tensorflow has enabled TensorRT support before run examples

## Examples ##
[example_00_get_model.py](/example_00_get_model.py) can generate needed Keras models with two different input formats, 
one for NCHW foramt, another one for NHWC format.
This is an optional, the needed models were already provided in repo.
```shell
$ python3 example_00_get_model.py -h # type this command to see detail info
```

[example_01_trt.py](/example_01_trt.py) shows how to run Keras model on TensorRT which can achieve fastest speed.
Because TensorRT didn't fully support NHWC yet, this approach only suits NCHW format.
```shell
$ python3 example_01_trt.py
```

[example_02_tftrt.py](/example_02_tftrt.py) shows how to run Keras model on Tensorflow 1.7 with TensorRT integration.
This approach supports both NCHW and NHWC format because Tensorflow can handles format issue.
```shell
$ python3 example_02_tftrt.py
```

[example_03_resnet50.py](/example_03_resnet50.py) shows how to run Keras Applications ResNet50 on Tensorflow 1.7 with TensorRT
```shell
$ python3 example_03_resnet50.py
```

[example_04_multi_input.py](/example_04_multi_input.py) shows how to run a multi-input/output Keras model on Tensorflow 1.7 with TensorRT
```shell
$ python3 example_04_multi_input.py
```
