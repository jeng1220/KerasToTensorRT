import keras
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt

import numpy as np
import sys
import time

import utils.ascii as helper
import utils.dataset as data

def get_tf_graph(model):
  with K.get_session() as sess:
      image_batch_t = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='image_tensor')
      K.set_learning_phase(0)
      conf_t = model(image_batch_t)
      dst_strs = [conf_t.name[:-2]]
      graphdef = sess.graph.as_graph_def()
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, dst_strs)
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

  return frozen_graph, 'image_tensor', dst_strs


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


# execute a graphdef
def tensorflow_infer(graph_def, input_str, output_str, input_data):
  sess, inp, out = get_tf_sess(graph_def, input_str, output_str)
  val = run_tf_sess(sess, inp, out, input_data)
  return val


class tftrt_engine():
  def __init__(self, graph, src_str, dst_str, batch_size, precision):
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

  def infer(self, x):
    num_test = x.shape[0]
    num_class = 10
    y = np.empty((num_test, num_class), np.float32)
    batch_size = self.batch_size
    for i in range(0, num_test, self.batch_size):
      x_part = x[i : i + batch_size, : ]
      y_part = run_tf_sess(self.sess, self.src_str, self.dst_str, x_part)
      y[i : i + batch_size, : ] = y_part
    return y


def main():
  # load pre-trained model
  model = load_model("nhwc_model.h5")
  model.summary()

  # load mnist dataset
  x_test, y_test = data.get_test_dataset()
  img_h = x_test.shape[1]
  img_w = x_test.shape[2]
  helper.print_ascii(x_test[0], img_h, img_w)

  # use Keras to do infer
  t0 = time.time()
  y_keras = model.predict(x_test)
  t1 = time.time()
  print('Keras time', t1 - t0)
  data.verify(y_keras, y_test)

  # use Tensorflow to infer
  tf_graph, src_str, dst_str = get_tf_graph(model)
  
  t0 = time.time()
  y_tf = tensorflow_infer(tf_graph, src_str, dst_str, x_test)
  t1 = time.time()
  print('Tensorflow time', t1 - t0)
  data.verify(y_tf, y_test)

  # use TensorRT in Tensorflow to infer
  engine = tftrt_engine(tf_graph, src_str, dst_str, 1000, 'FP32')

  t0 = time.time()
  y_tftrt = engine.infer(x_test)
  t1 = time.time()
  print('TensorRT time', t1 - t0)
  data.verify(y_tftrt, y_test)


if __name__ == "__main__":
  main()
