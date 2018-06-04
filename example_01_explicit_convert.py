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

def summary(model):
  model.summary()
  nb_layers = len(model.layers)
  for i in range(0, nb_layers):
    print('***debug***, layer name ', i, model.layers[i].name)
  for i in range(0, nb_layers):
    print('***debug***, input name ', i, model.layers[i].input.name)
  for i in range(0, nb_layers):
    print('***debug***, output name', i, model.layers[i].output.name)

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
  x_test  = x_test.astype('float32')
  x_train /= 255
  x_test  /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0],  'test samples')

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test  = keras.utils.to_categorical(y_test,  num_classes)

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
  
  trt_engine = get_trt_engine(model)

if __name__ == "__main__":
  main(sys.argv)
