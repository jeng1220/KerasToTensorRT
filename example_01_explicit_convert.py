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
import sys

def get_trt_engine(model):
  with K.get_session() as sess:
      #graphdef = sess.graph.as_graph_def()
      #print('***debug***, graphdef', type(graphdef), graphdef)
      #image_batch_t = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
      image_batch_t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_tensor')
      K.set_learning_phase(0)
      conf_t = model(image_batch_t)
      output_names = [conf_t.name[:-2]]
      print('***debug***, output_names', output_names)
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
      1,
      1 << 10)

  parser.destroy()
  print('success')
  return engine


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
  for y in range(0, img_h):
    str=''
    for x in range(0, img_w):
      data = res[y][x]
      if (data > 0.5) : str += '@'
      else : str += ' '
    print(str)


def verify(pre, ans):
  ans = keras.backend.get_value(keras.backend.argmax(ans, axis=-1))
  pre = keras.backend.get_value(keras.backend.argmax(pre, axis=-1))
  passed = 0
  num_test = ans.shape[0]
  for i in range(0, num_test):
    if (pre[i] == ans[i]) : passed = passed + 1

  if (passed / num_test > 0.99) : print('PASSED')
  else : print('FAILURE', passed)

  print('first inference result:', pre[0])


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
  y_predict = model.predict(x_test)
  verify(y_predict, y_test)

  trt_engine = get_trt_engine(model)

if __name__ == "__main__":
  main(sys.argv)
