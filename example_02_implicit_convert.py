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

import tensorrt as trt
import uff
import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt
from tensorrt.parsers import uffparser

import numpy as np
from pdb import set_trace
import sys

def get_tf_graph(model):
  with K.get_session() as sess:
      #image_batch_t = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
      image_batch_t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_tensor')
      K.set_learning_phase(0)
      conf_t = model(image_batch_t)
      output_names = [conf_t.name[:-2]]
      print('***debug***, output_names', output_names)
      graphdef = sess.graph.as_graph_def()
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_names)
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

  print('***debug***, frozen_graph', type(frozen_graph))
  return frozen_graph, 'image_tensor', output_names

################################################################################
# execute a graphdef
################################################################################
def run_graphdef(graph_def, input_str, output_str, input_data):
  #set_trace()
  # load TF-TRT graph into memory and extract input & output nodes
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=graph_def, return_elements=[input_str, output_str[0]])
    #print('***debug***, inp', type(inp))
    #print('***debug***, out', type(out))
    #print('***debug***, inp', inp)
    #print('***debug***, out', out)
    inp = inp.outputs[0]
    out = out.outputs[0]

  #print('***debug***, inp', type(inp))
  #print('***debug***, out', type(out))
  #print('***debug***, inp', inp)
  #print('***debug***, out', out)


  # allow_growth and restrict Tensorflow to claim all GPU memory
  # currently TensorRT engine uses independent memory allocation outside of TF
  config=tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
  # we can now import trt_graph into Tensorflow and execute it. If given target
  with tf.Session(graph=g, config=config) as sess:
    val = sess.run(out, {inp: input_data})
  return val
################################################################################

def summary(model):
  model.summary()
  nb_layers = len(model.layers)
  for i in range(0, nb_layers):
    print('***debug***, layer name ', i, model.layers[i].name)
  for i in range(0, nb_layers):
    print('***debug***, input name ', i, model.layers[i].input.name)
  for i in range(0, nb_layers):
    print('***debug***, output name', i, model.layers[i].output.name)

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
  img_rows, img_cols = 28, 28

  # the data, split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  model = load_model("my_model.h5")
  summary(model)

  score = model.evaluate(x_test, y_test, verbose=0)
  # make sure load right model
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  predict = model.predict(x_test)
  # print predict to compared TRT result
  print('***debug***, x_test.shape', x_test.shape)
  print('***debug***, predict[0]', predict[0])

  tf_graph, input_name, output_name = get_tf_graph(model)
  print('***debug***, input_name', type(input_name), input_name)
  print('***debug***, output_name', type(output_name), output_name)
  #print('***debug***, tf_graph', type(tf_graph), tf_graph)
  print('***debug***, tf_graph', type(tf_graph))  

  tf_result = run_graphdef(tf_graph, input_name, output_name, x_test)
  print('***debug***, tf_result', type(tf_result), tf_result.shape)
  print('***debug***, tf_result', tf_result[0])

  print('***debug***, after run_graphdef')
  trt_graph = convert_tftrt_fp(tf_graph, 1, 'FP32', output_name[0]) 
  #print('***debug***, trt_graph', type(trt_graph), trt_graph)
  print('***debug***, trt_graph', type(trt_graph))

  print('***debug***, after convert_tftrt_fp')
  x_1_test = x_test[0]
  x_1_test = x_1_test.reshape((1, 28, 28, 1))
  print('***debug***, x_1_test.shape', x_1_test.shape)
  trt_result = run_graphdef(trt_graph, input_name, output_name, x_1_test)
  print('***debug***, trt_result', trt_result)


if __name__ == "__main__":
  main(sys.argv)
