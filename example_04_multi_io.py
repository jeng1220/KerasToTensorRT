import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, merge, Concatenate, Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt

import utils.ascii as helper
import numpy as np
import time
from enum import Enum
from optparse import OptionParser
from pdb import set_trace

class ImageForamtType(Enum):
  NCHW = '0'
  NHWC = '1'

class Dataset():
  def __init__(self, x0, x1, y0, y1):
    self.x0 = x0
    self.x1 = x1
    self.y0 = y0
    self.y1 = y1

def preprocess_data(x, y, num_classes):
  num_tests = int(x.shape[0] / 2)
  x0 = x[0 : num_tests]
  x1 = x[num_tests : 2 * num_tests]

  t0 = y[0 : num_tests]
  t1 = y[num_tests : 2 * num_tests]
  
  y0 = keras.utils.to_categorical(t0 * 10 + t1, num_classes[0])
  y1 = keras.utils.to_categorical(t0 *  1 + t1, num_classes[1])

  return Dataset(x0, x1, y0, y1)

def get_data(img_format):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  img_h, img_w = 28, 28
  num_classes = [100, 20]

  if (img_format == ImageForamtType.NCHW):
    K.set_image_data_format('channels_first')
    x_train  = x_train.reshape(x_train.shape[0], 1, img_h, img_w)
    x_test   = x_test.reshape(x_test.shape[0],   1, img_h, img_w)
    img_shape = (1, img_h, img_w)
  else:
    K.set_image_data_format('channels_last')
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0],    img_h, img_w, 1)
    img_shape = (img_h, img_w, 1)

  x_train = x_train.astype('float32')
  x_test  = x_test.astype('float32')
  x_train /= 255
  x_test  /= 255

  train = preprocess_data(x_train, y_train, num_classes)
  test  = preprocess_data(x_test,  y_test,  num_classes)

  return num_classes, img_shape, train, test

def custom_model(img_shape, num_classes):
  y0_shape = (num_classes[0], )
  y1_shape = (num_classes[1], )

  input_a = Input(shape=img_shape, name='input_a')
  x0 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(input_a)
  x0 = MaxPooling2D(pool_size=(2, 2))(x0)

  input_b = Input(shape=img_shape, name='input_b')
  x1 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(input_b)
  x1 = MaxPooling2D(pool_size=(2, 2))(x1)

  if (img_shape[0] == 1) :
    x2 = Concatenate(axis=1)([x0, x1])
  else:
    x2 = Concatenate(axis=3)([x0, x1])
  x2 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(x2)
  x2 = MaxPooling2D(pool_size=(2, 2))(x2)

  x3 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(x2)
  x3 = MaxPooling2D(pool_size=(2, 2))(x3)
  x3 = Flatten()(x3)
  output_a = Dense(*y0_shape, activation='softmax', name='output_b')(x3)

  x4 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(x2)
  x4 = MaxPooling2D(pool_size=(2, 2))(x4)
  x4 = Flatten()(x4)
  output_b = Dense(*y1_shape, activation='softmax', name='output_a')(x4)  

  model = Model(inputs=[input_a, input_b], outputs=[output_a, output_b])
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  model.summary()
  return model

def get_keras_model(num_classes, img_shape, train, test):
  model = custom_model(img_shape, num_classes)
  batch_size = 1000
  epochs = 12
  
  x_train = [train.x0, train.x1]
  x_test = [test.x0, test.x1]

  y_train = [train.y0, train.y1]
  y_test = [test.y0, test.y1]

  model.fit(x_train, y_train,
    batch_size, epochs,
    verbose=1,
    validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  return model

class FrozenGraph(object):
  def __init__(self, model, shape):
    shape = (None, shape[0], shape[1], shape[2])
    x0_name = 'image_tensor_x0'
    x1_name = 'image_tensor_x1'
    with K.get_session() as sess:
        x0_tensor = tf.placeholder(tf.float32, shape, x0_name)
        x1_tensor = tf.placeholder(tf.float32, shape, x1_name)
        K.set_learning_phase(0)
        y_tensor = model([x0_tensor, x1_tensor])
        y0_name = y_tensor[0].name[:-2]
        y1_name = y_tensor[1].name[:-2]
        y_name = [y0_name, y1_name]
        graph = sess.graph.as_graph_def()
        graph = tf.graph_util.convert_variables_to_constants(sess, graph, y_name)
        graph = tf.graph_util.remove_training_nodes(graph)

    self.x_name = [x0_name, x1_name]
    self.y_name = y_name
    self.frozen = graph  

class TfEngine(object):
  def __init__(self, graph):
    g = tf.Graph()
    with g.as_default():
      x0_op, x1_op, y0_op, y1_op = tf.import_graph_def(
          graph_def=graph.frozen, return_elements=graph.x_name + graph.y_name)
      self.x0_tensor = x0_op.outputs[0]
      self.x1_tensor = x1_op.outputs[0]
      self.y_tensors = [y0_op.outputs[0], y1_op.outputs[0]]

    config = tf.ConfigProto(gpu_options=
      tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
      allow_growth=True))

    self.sess = tf.Session(graph=g, config=config)

  def infer(self, x0, x1):
    y0, y1 = self.sess.run(self.y_tensors,
      feed_dict={self.x0_tensor: x0, self.x1_tensor: x1})
    return y0, y1

class TftrtEngine(TfEngine):
  def __init__(self, graph, batch_size, precision):
    tftrt_graph = tftrt.create_inference_graph(
      graph.frozen,
      outputs=graph.y_name,
      max_batch_size=batch_size,
      max_workspace_size_bytes=1 << 25,
      precision_mode=precision,
      minimum_segment_size=2)

    graph.frozen = tftrt_graph
    super(TftrtEngine, self).__init__(graph)
    self.batch_size = batch_size

  def infer(self, x0, x1):
    num_tests = x0.shape[0]
    y0 = np.empty((num_tests, self.y_tensors[0].shape[1]), np.float32)
    y1 = np.empty((num_tests, self.y_tensors[1].shape[1]), np.float32)
    batch_size = self.batch_size

    for i in range(0, num_tests, batch_size):
      x0_part = x0[i : i + batch_size]
      x1_part = x1[i : i + batch_size]
      y0_part, y1_part = self.sess.run(self.y_tensors,
        feed_dict={self.x0_tensor: x0_part, self.x1_tensor: x1_part})
      y0[i : i + batch_size] = y0_part
      y1[i : i + batch_size] = y1_part
    return y0, y1

def verify(result, ans):
  num_tests = ans.shape[0]
  error = 0
  for i in range(0, num_tests):
    a = np.argmax(ans[i])
    r = np.argmax(result[i])
    if (a != r) : error += 1

  if (error == 0) : print('PASSED')
  else            : print('FAILURE')

def example(img_format):
  num_classes, img_shape, train, test = get_data(img_format)
  model = get_keras_model(num_classes, img_shape, train, test)
  
  if (img_format == ImageForamtType.NCHW):
    img_h = img_shape[1]
    img_w = img_shape[2]
  else:
    img_h = img_shape[0]
    img_w = img_shape[1]

  helper.print_ascii(test.x0[0], img_h, img_w)
  helper.print_ascii(test.x1[0], img_h, img_w)
  x_test = [test.x0, test.x1]
  t0 = time.time()
  y0_keras, y1_keras = model.predict(x_test)
  t1 = time.time()
  print('Keras predict result:', np.argmax(y0_keras[0]), np.argmax(y1_keras[0]),
    '\nKeras time:', t1 - t0)

  frozen_graph = FrozenGraph(model, img_shape)

  tf_engine = TfEngine(frozen_graph)
  t0 = time.time()
  y0_tf, y1_tf = tf_engine.infer(test.x0, test.x1)
  t1 = time.time()
  print('Tensorflow time:', t1 - t0)
  verify(y0_tf, y0_keras)
  verify(y1_tf, y1_keras)

  tftrt_engine = TftrtEngine(frozen_graph, 1000, 'FP32')
  t0 = time.time()
  y0_tftrt, y1_tftrt = tftrt_engine.infer(test.x0, test.x1)
  t1 = time.time()
  print('TFTRT time', t1 - t0)
  verify(y0_tftrt, y0_keras)
  verify(y1_tftrt, y1_keras)

def main():
  parser = OptionParser()
  parser.add_option('-f', '--format', dest='format',
                    default='0',
                    help='image format, 0) NCHW, 1) NHWC, default NCHW')
  (options, args) = parser.parse_args()
  print(options)
  example(ImageForamtType(options.format))

if __name__ == '__main__':
  main()