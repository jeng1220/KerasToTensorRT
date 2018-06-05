from __future__ import print_function
import keras
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
import uff
import tensorrt as trt
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import numpy as np
import sys
import time

import utils.ascii as helper
import utils.dataset as data

def get_trt_engine(model, batch_size):
  with K.get_session() as sess:
    image_batch_t = tf.placeholder(tf.float32, shape=(None, 1, 28, 28), name='image_tensor')
    K.set_learning_phase(0)
    conf_t = model(image_batch_t)
    output_names = [conf_t.name[:-2]]
    graphdef = sess.graph.as_graph_def()      
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_names)
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

  uff_model = uff.from_tensorflow(frozen_graph, output_names)
  G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

  parser = uffparser.create_uff_parser()
  input_shape = (1, 28, 28)
  parser.register_input("image_tensor", input_shape, 0)
  parser.register_output(output_names[0])
  engine = trt.utils.uff_to_trt_engine(G_LOGGER,
    stream=uff_model,
    parser=parser,
    max_batch_size=batch_size,
    max_workspace_size=1 << 25,
    datatype='FP32')

  parser.destroy()
  return engine


def main(argv):
  K.set_image_data_format('channels_first')
  model = load_model("nchw_model.h5")
  model.summary()
  
  x_test, y_test = data.get_test_dataset()
  img_h = x_test.shape[2]
  img_w = x_test.shape[3]
  helper.print_ascii(x_test[0], img_h, img_w)

  t0 = time.time()
  y_predict = model.predict(x_test)
  t1 = time.time()
  print('Keras time', t1 - t0)
  data.verify(y_predict, y_test)

  num_test = x_test.shape[0]
  batch_size = 1000
  trt_engine = get_trt_engine(model, batch_size)
  trt_context = trt_engine.create_execution_context()

  x_part = x_test[ : batch_size]
  y_part = y_predict[ : batch_size]
  d_src = cuda.mem_alloc(x_part.nbytes)
  d_dst = cuda.mem_alloc(y_part.nbytes)
  bindings = [int(d_src), int(d_dst)]

  y_predict_trt = np.empty(y_predict.shape, y_predict.dtype)

  t0 = time.time()

  for i in range(0, num_test, batch_size):
    x_part = x_test[i : i + batch_size]
    y_part = y_predict_trt[i : i + batch_size]
    cuda.memcpy_htod(d_src, x_part)
    trt_context.execute(batch_size, bindings)
    cuda.memcpy_dtoh(y_part, d_dst)

  t1 = time.time()
  print('TensorRT time', t1 - t0)
  data.verify(y_predict_trt, y_test)


if __name__ == "__main__":
  main(sys.argv)
