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
  def __init__(self, x0, x1, y):
    self.x0 = x0
    self.x1 = x1
    self.y  = y

def preprocess_data(x, y, num_classes):
  num_tests = int(x.shape[0] / 2)
  x0 = x[0 : num_tests]
  x1 = x[num_tests : 2 * num_tests]

  y0 = y[0 : num_tests]
  y1 = y[num_tests : 2 * num_tests]  
  y = y0 * 10 + y1
  y = keras.utils.to_categorical(y, num_classes)

  return Dataset(x0, x1, y)

def get_data(img_format):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  img_h, img_w = 28, 28
  num_classes = 100

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
  output_shape = (num_classes, )

  input_a = Input(shape=img_shape, name='input_a')
  x0 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(input_a)
  x0 = MaxPooling2D(pool_size=(2, 2))(x0)

  input_b = Input(shape=img_shape, name='input_b')
  x1 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(input_b)
  x1 = MaxPooling2D(pool_size=(2, 2))(x1)

  x2 = Concatenate(axis=-1)([x0, x1])
  x2 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(x2)
  x2 = MaxPooling2D(pool_size=(2, 2))(x2)
  x2 = Flatten()(x2)
  output = Dense(*output_shape, activation='softmax',name='output')(x2)

  model = Model(inputs=[input_a, input_b], outputs=output)
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  model.summary()
  return model

def get_keras_model(num_classes, img_shape, train, test):
  model = custom_model(img_shape, num_classes)
  batch_size = 1000
  epochs = 4
  
  x_train = [train.x0, train.x1]
  x_test = [test.x0, test.x1]

  model.fit(x_train, train.y,
    batch_size, epochs,
    verbose=1,
    validation_data=(x_test, test.y))
  score = model.evaluate(x_test, test.y, verbose=1)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  return model

class FrozenGraph(object):
  def __init__(self, model, shape):
    shape = (None, shape[0], shape[1], shape[2])
    src_name0 = 'image_tensor_x0'
    src_name1 = 'image_tensor_x1'
    with K.get_session() as sess:
        x0_tensor = tf.placeholder(tf.float32, shape, src_name0)
        x1_tensor = tf.placeholder(tf.float32, shape, src_name1)
        K.set_learning_phase(0)
        conf_t = model([x0_tensor, x1_tensor])
        dst_name = [conf_t.name[:-2]]
        graph = sess.graph.as_graph_def()
        graph = tf.graph_util.convert_variables_to_constants(sess, graph, dst_name)
        graph = tf.graph_util.remove_training_nodes(graph)

    self.src_name = [src_name0, src_name1]
    self.dst_name = dst_name
    self.frozen = graph  

class TfEngine(object):
  def __init__(self, graph):
    g = tf.Graph()
    with g.as_default():
      x0_op, x1_op, y_op = tf.import_graph_def(
          graph_def=graph.frozen, return_elements=[graph.src_name[0], graph.src_name[1], graph.dst_name[0]])
      self.x0_tensor = x0_op.outputs[0]
      self.x1_tensor = x1_op.outputs[0]
      self.y_tensor  = y_op.outputs[0]

    config = tf.ConfigProto(gpu_options=
      tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
      allow_growth=True))

    self.sess = tf.Session(graph=g, config=config)

  def infer(self, x0, x1):
    y = self.sess.run(self.y_tensor,
      feed_dict={self.x0_tensor: x0, self.x1_tensor: x1})
    return y

class TftrtEngine(TfEngine):
  def __init__(self, graph, batch_size, precision):
    tftrt_graph = tftrt.create_inference_graph(
      graph.frozen,
      outputs=graph.dst_name,
      max_batch_size=batch_size,
      max_workspace_size_bytes=1 << 25,
      precision_mode=precision,
      minimum_segment_size=2)

    graph.frozen = tftrt_graph
    super(TftrtEngine, self).__init__(graph)
    self.batch_size = batch_size

  def infer(self, x0, x1):
    num_tests = x0.shape[0]
    num_classes = 100
    y = np.empty((num_tests, num_classes), np.float32)
    batch_size = self.batch_size

    for i in range(0, num_tests, batch_size):
      x_part0 = x0[i : i + batch_size]
      x_part1 = x1[i : i + batch_size]
      y_part = self.sess.run(self.y_tensor,
        feed_dict={self.x0_tensor: x_part0, self.x1_tensor: x_part1})
      y[i : i + batch_size] = y_part
    return y

def verify(ans, result):
  num_tests = ans.shape[0]
  error = 0
  for i in range(0, num_tests):
    a = np.argmax(ans[i])
    r = np.argmax(result[i])
    if (a != r) : error += 1
  if (error == 0) : print('PASSED')
  else : print('FAILURE')

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
  y_keras = model.predict(x_test)
  t1 = time.time()
  print('Keras predict result:', np.argmax(y_keras[0]),
    '\nKeras time:', t1 - t0)

  frozen_graph = FrozenGraph(model, img_shape)

  tf_engine = TfEngine(frozen_graph)
  t0 = time.time()
  y_tf = tf_engine.infer(test.x0, test.x1)
  t1 = time.time()
  print('Tensorflow time:', t1 - t0)
  verify(y_tf, y_keras)  

  tftrt_engine = TftrtEngine(frozen_graph, 1000, 'FP32')
  t0 = time.time()
  y_tftrt = tftrt_engine.infer(test.x0, test.x1)
  t1 = time.time()
  print('TFTRT time', t1 - t0)
  verify(y_tftrt, y_keras)  

def main():
  parser = OptionParser()
  parser.add_option('-f', '--format', dest='format',
                    default='0',
                    help='image format, 0) NCHW, 1) NHWC, default NCHW')
  (options, args) = parser.parse_args()

  example(options.format)

if __name__ == '__main__':
  main()