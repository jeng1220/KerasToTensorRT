from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K


import tensorflow as tf
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import numpy as np
import pdb
import sys


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
  #x_test = (1 - x_test) / 255

  # convert class vectors to binary class matrices
  y_test  = keras.utils.to_categorical(y_test,  num_classes)
  return x_test, y_test


def show_number(img, img_h, img_w):
  res = img.reshape((img_h, img_w))
  for y in range(0, img_h):
    str=''
    for x in range(0, img_w):
      data = res[y][x]
      if (data > 0.5) : str += '@'
      else : str += ' '
    print(str)


def get_trt_engine(model, batch_size=1):
  with K.get_session() as sess:
    image_batch_t = tf.placeholder(tf.float32, shape=(None, 1, 28, 28), name='image_tensor')
    #image_batch_t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_tensor')
    K.set_learning_phase(0)
    conf_t = model(image_batch_t)
    output_names = [conf_t.name[:-2]]
    #print('***debug***, output_names', output_names)
    graphdef = sess.graph.as_graph_def()      
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_names)
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

  uff_model = uff.from_tensorflow(frozen_graph, output_names)
  G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

  parser = uffparser.create_uff_parser()
  #input_shape = (28, 28, 1)
  input_shape = (1, 28, 28)
  #parser.register_input("placeholder", input_shape, 1)
  parser.register_input("image_tensor", input_shape, 0)
  parser.register_output(output_names[0])
  engine = trt.utils.uff_to_trt_engine(G_LOGGER,
    uff_model,
    parser,
    batch_size,
    1 << 25)

  parser.destroy()
  print('success')
  return engine


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


def trt_summary(engine):
  num_layer = engine.get_nb_layers()
  print('num_layer', num_layer)
  num_bind = engine.get_nb_bindings()
  for i in range(0, num_bind):
    dims = engine.get_binding_dimensions(i).to_DimsCHW()
    print(engine.get_binding_name(i), dims.C(), dims.H(), dims.W())


class Profiler(trt.infer.Profiler):
  """
  Example Implimentation of a Profiler
  Is identical to the Profiler class in trt.infer so it is possible
  to just use that instead of implementing this if further
  functionality is not needed
  """
  def __init__(self, timing_iter):
    trt.infer.Profiler.__init__(self)
    self.timing_iterations = timing_iter
    self.profile = []

  def report_layer_time(self, layerName, ms):
    record = next((r for r in self.profile if r[0] == layerName), (None, None))
    if record == (None, None):
        self.profile.append((layerName, ms))
    else:
        self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

  def print_layer_times(self):
    totalTime = 0
    for i in range(len(self.profile)):
        print("{:40.40} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
        totalTime += self.profile[i][1]
    print("Time over all layers: {:4.3f}".format(totalTime / self.timing_iterations))


def infer(engine, input_img, batch_size):
  #pdb.set_trace() 
  #load engine
  context = engine.create_execution_context()
  assert(engine.get_nb_bindings() == 2)
  #create output array to receive data
  assert(engine.binding_is_input(0) == True)

  src_dims = engine.get_binding_dimensions(0).to_DimsCHW()
  src_elt_count = src_dims.C() * src_dims.H() * src_dims.W() * batch_size

  dst_dims = engine.get_binding_dimensions(1).to_DimsCHW()
  dst_elt_count = dst_dims.C() * dst_dims.H() * dst_dims.W() * batch_size
  #Allocate pagelocked memory

  h_src = cuda.pagelocked_empty(src_elt_count, dtype = np.float32)
  h_dst = cuda.pagelocked_empty(dst_elt_count, dtype = np.float32)

  #alocate device memory
  d_src = cuda.mem_alloc(batch_size * h_src.size * h_src.dtype.itemsize)
  d_dst = cuda.mem_alloc(batch_size * h_dst.size * h_dst.dtype.itemsize)

  bindings = [int(d_src), int(d_dst)]

  stream = cuda.Stream()

  show_number(input_img, 28, 28)
  res = input_img.reshape((28, 28))
  for y in range(0, 28):
    for x in range(0, 28):
      h_src[y * 28 + x] = res[y][x]

  #transfer input data to device
  cuda.memcpy_htod_async(d_src, h_src, stream)
  #execute model 
  #TIMING_INTERATIONS = 1000
  #G_PROFILER = Profiler(TIMING_INTERATIONS)
  #context.set_profiler(G_PROFILER)
  context.set_debug_sync(True)

  context.enqueue(batch_size, bindings, stream.handle, None)

  #transfer predictions back
  cuda.memcpy_dtoh_async(h_dst, d_dst, stream)
  #G_PROFILER.print_layer_times()

  #return predictions
  return h_dst

def main(argv):
  num_classes = 10
  # input image dimensions
  img_h, img_w = 28, 28

  K.set_image_data_format('channels_first')
  model = load_model("my_model.h5")
  model.summary()
  
  x_test, y_test = get_dataset(num_classes, img_h, img_w)
  score = model.evaluate(x_test, y_test, verbose=0)
  # make sure load right model
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  show_number(x_test[0], img_h, img_w)
  y_predict = model.predict(x_test)
  verify(y_predict, y_test)

  #num_test = 1325
  num_test = 1
  trt_engine = get_trt_engine(model, num_test)
  assert(trt_engine)
  trt_summary(trt_engine)
  y_predict_trt = infer(trt_engine, x_test[0], num_test)
  print('first inference result:', np.argmax(y_predict_trt))

  print('\ncompare:')
  print('Keras\n',    y_predict[0])
  print('TensorRT\n', y_predict_trt)

if __name__ == "__main__":
  main(sys.argv)
