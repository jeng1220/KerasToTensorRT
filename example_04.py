import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, merge, Concatenate, Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt

from optparse import OptionParser
from pdb import set_trace
import utils.ascii as helper
import numpy as np

def custom_model(input_shape, num_classes):
  output_shape = (num_classes, )

  input_a = Input(shape=input_shape, name='input_a')
  x0 = Conv2D(32, kernel_size=(3, 3), 
    activation='relu')(input_a)
  x0 = MaxPooling2D(pool_size=(2, 2))(x0)

  input_b = Input(shape=input_shape, name='input_b')
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
  return output_shape, model

def preprocess_data(x, y, num_classes):
  #set_trace()
  num_test = int(x.shape[0] / 2)
  x_part_a = x[0 : num_test]
  x_part_b = x[num_test : 2 * num_test]

  y_part_a = y[0 : num_test]
  y_part_b = y[num_test : 2 * num_test]  
  y_out = y_part_a * 10 + y_part_b
  y_out = keras.utils.to_categorical(y_out, num_classes)

  return x_part_a, x_part_b, y_out

def get_tf_graph(model, s):
  with K.get_session() as sess:
      input_a = tf.placeholder(tf.float32, shape=(None, s[0], s[1], s[2]), name='image_tensor_a')
      input_b = tf.placeholder(tf.float32, shape=(None, s[0], s[1], s[2]), name='image_tensor_b')
      K.set_learning_phase(0)
      conf_t = model([input_a, input_b])
      dst_strs = [conf_t.name[:-2]]
      graphdef = sess.graph.as_graph_def()
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, dst_strs)
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

  return frozen_graph, ['image_tensor_a', 'image_tensor_b'], dst_strs

def get_tf_sess(graph_def, input_str, output_str):
  # load TF-TRT graph into memory and extract input & output nodes
  g = tf.Graph()
  with g.as_default():
    ina, inb, out = tf.import_graph_def(
        graph_def=graph_def, return_elements=[input_str[0], input_str[1], output_str[0]])
    ina = ina.outputs[0]
    inb = inb.outputs[0]
    out = out.outputs[0]
  # allow_growth and restrict Tensorflow to claim all GPU memory
  # currently TensorRT engine uses independent memory allocation outside of TF
  config = tf.ConfigProto(gpu_options=
             tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
             allow_growth=True))
  # we can now import trt_graph into Tensorflow and execute it. If given target
  sess = tf.Session(graph=g, config=config)
  return sess, [ina, inb], out

def run_tf_sess(sess, inp, out, input_data):
  #set_trace()
  val = sess.run(out, {inp[0]: input_data[0], inp[1]: input_data[1]})
  #val = sess.run(out)
  return val

def tensorflow_infer(graph_def, input_str, output_str, input_data):
  sess, inp, out = get_tf_sess(graph_def, input_str, output_str)
  val = run_tf_sess(sess, inp, out, input_data)
  return val

class tftrt_engine():
  def __init__(self, graph, src_str, dst_str, batch_size, precision):
    #set_trace()
    tftrt_graph = tftrt.create_inference_graph(
      input_graph_def=graph, # native Tensorflow graphdef
      outputs=dst_str,
      max_batch_size=batch_size,
      max_workspace_size_bytes=1 << 25,
      precision_mode=precision,
      minimum_segment_size=2)
    sess, src_str, dst_str = get_tf_sess(tftrt_graph, src_str, dst_str)
    self.sess = sess
    self.src_str = src_str
    self.dst_str = dst_str
    self.batch_size = batch_size

  def infer(self, x_a, x_b):
    num_test = x_a.shape[0]
    num_class = 100
    y = np.empty((num_test, num_class), np.float32)
    batch_size = self.batch_size
    for i in range(0, num_test, self.batch_size):
      x_part_a = x_a[i : i + batch_size, : ]
      x_part_b = x_b[i : i + batch_size, : ]
      y_part = run_tf_sess(self.sess, self.src_str, self.dst_str, [x_part_a, x_part_b])
      y[i : i + batch_size, : ] = y_part
    return y

def main():
  parser = OptionParser()
  parser.add_option('-f', '--format', dest='format',
                    default='0',
                    help='image format, 0) NCHW, 1) NHWC, default NCHW')
  (options, args) = parser.parse_args()

  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  img_h, img_w = 28, 28
  num_classes = 100

  if (options.format == '0'):
    K.set_image_data_format('channels_first')
    filename = 'ex04_nchw_model.h5'
    x_train = x_train.reshape(x_train.shape[0], 1, img_h, img_w)
    x_test = x_test.reshape(x_test.shape[0], 1, img_h, img_w)
    input_shape = (1, img_h, img_w)
  else:
    K.set_image_data_format('channels_last')
    filename = 'ex04_nhwc_model.h5'
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    input_shape = (img_h, img_w, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  x_train_a, x_train_b, y_train = preprocess_data(x_train, y_train, num_classes)
  x_test_a, x_test_b, y_test = preprocess_data(x_test, y_test, num_classes)

  output_shape, model = custom_model(input_shape, num_classes)
  batch_size = 100
  epochs = 1
  model.fit([x_train_a, x_train_b], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=([x_test_a, x_test_b], y_test))
  score = model.evaluate([x_test_a, x_test_b], y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  #model.save(filename)

  helper.print_ascii(x_test_a[0], img_h, img_w)
  helper.print_ascii(x_test_b[0], img_h, img_w)
  y_keras = model.predict([x_test_a, x_test_b])
  print(np.argmax(y_keras[0]))

  tf_graph, src_str, dst_str = get_tf_graph(model, input_shape)
  y_tf = tensorflow_infer(tf_graph, src_str, dst_str, [x_test_a, x_test_b])
  print(np.argmax(y_tf[0]))

  tftrt_graph = tftrt_engine(tf_graph, src_str, dst_str, 1000, 'FP32')
  y_tftrt = tftrt_graph.infer(x_test_a, x_test_b)
  print(np.argmax(y_tftrt[0]))

if __name__ == '__main__':
  main()