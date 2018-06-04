'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K


import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt
from tensorrt.parsers import uffparser

import numpy as np
from pdb import set_trace
import sys
import time

def get_dataset(num_classes, img_h, img_w):
  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_h, img_w)
    input_shape = (1, img_h, img_w)
  else:
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    input_shape = (img_h, img_w, 1)

  x_test = x_test.astype('float32')
  x_test /= 255

  # convert class vectors to binary class matrices
  y_test  = keras.utils.to_categorical(y_test,  num_classes)
  return x_test, y_test


def show_number(img, img_h, img_w):
  res = img.reshape((img_h, img_w))
  str=''
  for y in range(0, img_h):
    for x in range(0, img_w):
      data = res[y][x]
      if (data > 0.5) : str += '@'
      else : str += ' '
    str += '\n'
  print(str)


def verify(pre, ans):
  passed = 0
  num_test = ans.shape[0]

  for i in range(0, num_test):
    a = np.argmax(ans[i])
    p = np.argmax(pre[i])
    if (p == a) : passed = passed + 1

  if (passed / num_test > 0.99) : print('PASSED')
  else : print('FAILURE', passed)

  p = np.argmax(pre[0])
  print('first inference result:', p, '\n\n')


def get_tf_graph(model):
  with K.get_session() as sess:
      image_batch_t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_tensor')
      K.set_learning_phase(0)
      conf_t = model(image_batch_t)
      output_names = [conf_t.name[:-2]]
      graphdef = sess.graph.as_graph_def()
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_names)
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

  return frozen_graph, 'image_tensor', output_names


def get_tf_sess(graph_def, input_str, output_str):
  # load TF-TRT graph into memory and extract input & output nodes
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=graph_def, return_elements=[input_str, output_str[0]])
    inp = inp.outputs[0]
    out = out.outputs[0]
  # allow_growth and restrict Tensorflow to claim all GPU memory
  # currently TensorRT engine uses independent memory allocation outside of TF
  config=tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
  # we can now import trt_graph into Tensorflow and execute it. If given target
  sess = tf.Session(graph=g, config=config)
  return sess, inp, out


def run_tf_sess(sess, inp, out, input_data):
  val = sess.run(out, {inp: input_data})
  return val


################################################################################
# execute a graphdef
################################################################################
def run_graphdef(graph_def, input_str, output_str, input_data):
  sess, inp, out = get_tf_sess(graph_def, input_str, output_str)
  val = run_tf_sess(sess, inp, out, input_data)
  return val


################################################################################
# conversion example
################################################################################
def convert_tftrt_fp(orig_graph, batch_size, precision, output_str):
  # convert native Tensorflow graphdef into a mixed TF-TRT graph
  trt_graph = tftrt.create_inference_graph(
      input_graph_def=orig_graph,       # native Tensorflow graphdef
      outputs=[output_str],             # list of names for output node
      max_batch_size=batch_size,        # maximum/optimum batchsize for TF-TRT
                                        # mixed graphdef
      max_workspace_size_bytes=1 << 25, # maximum workspace (in MB) for each 
                                        # TRT engine to allocate
      precision_mode=precision,         # TRT Engine precision
                                        # "FP32","FP16" or "INT8"
      minimum_segment_size=2            # minimum number of nodes in an engine,
                                        # this parameter allows the converter to
                                        # skip subgraph with total node number
                                        # less than the threshold
  )

  # we can now import trt_graph into Tensorflow and execute it. If given target
  # precision_mode as 'FP32' or 'FP16'.
  if precision=='FP16' or precision=='FP32':
    return trt_graph


def main(argv):
  num_classes = 10
  # input image dimensions
  img_h, img_w = 28, 28

  model = load_model("my_model.h5")
  model.summary()

  x_test, y_test = get_dataset(num_classes, img_h, img_w)
  score = model.evaluate(x_test, y_test, verbose=0)
  # make sure load right model
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  show_number(x_test[0], img_h, img_w)
  t0 = time.time()
  y_predict = model.predict(x_test)
  t1 = time.time()
  print('Keras time', t1 - t0)
  verify(y_predict, y_test)

  tf_graph, input_name, output_name = get_tf_graph(model)
  #print('***debug***, tf_graph', type(tf_graph), tf_graph)
  t0 = time.time()
  y_predict_tf = run_graphdef(tf_graph, input_name, output_name, x_test)
  t1 = time.time()
  print('Tensorflow time', t1 - t0)
  verify(y_predict_tf, y_test)

  num_test = x_test.shape[0]
  batch_size = 1000
  trt_graph = convert_tftrt_fp(tf_graph, batch_size, 'FP32', output_name[0]) 
  sess, inp, out = get_tf_sess(trt_graph, input_name, output_name)
  y_predict_trt = np.zeros(y_test.shape, y_test.dtype)

  t0 = time.time()

  for i in range(0, num_test, batch_size):
    x_part = x_test[i : i + batch_size, : ]
    y_part = run_tf_sess(sess, inp, out, x_part)
    y_predict_trt[i : i + batch_size, : ] = y_part

  t1 = time.time()
  
  print('TensorRT time', t1 - t0)
  verify(y_predict_trt, y_test)


if __name__ == "__main__":
  main(sys.argv)
