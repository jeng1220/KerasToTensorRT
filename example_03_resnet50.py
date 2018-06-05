import keras
from keras.applications.resnet50 import ResNet50
from keras import backend as K
import os
import time

import tensorflow as tf
from tensorflow.contrib import tensorrt as tftrt
import numpy as np

def get_tf_graph(model):
  with K.get_session() as sess:
      image_batch_t = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image_tensor')
      K.set_learning_phase(0)
      conf_t = model(image_batch_t)
      dst_strs = [conf_t.name[:-2]]
      graphdef = sess.graph.as_graph_def()
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, dst_strs)
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
  return frozen_graph, 'image_tensor', dst_strs

def run_graphdef(graph_def, input_str, output_str, input_data):
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
  with tf.Session(graph=g, config=config) as sess:
    val = sess.run(out, {inp: input_data})
  return val

# conversion example
def get_tftrt_graph(orig_graph, batch_size, precision, output_str):
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

def verify(a, b):
  num_test = a.shape[0]
  assert(num_test == b.shape[0])
  passed = 0

  for i in range(0, num_test):
    a2 = np.argmax(a[i])
    b2 = np.argmax(b[i])
    if (a2 == b2): passed += 1
  
  if (passed == num_test): print('PASSED\n')
  else: print('FAILURE\n')

def main():
  model = ResNet50(weights='imagenet')
  batch_size = 128
  x_test = np.random.random_sample((batch_size, 224, 224, 3))
  src_str = 'image_tensor'

  t0 = time.time()
  y_keras = model.predict(x_test)
  t1 = time.time()
  print('Keras time', t1 - t0)

  tf_graph, src_str, dst_str = get_tf_graph(model)

  t0 = time.time() 
  y_tf = run_graphdef(tf_graph, src_str, dst_str, x_test) 
  t1 = time.time()
  print('Tensorflow time', t1 - t0)
  verify(y_keras, y_tf)

  trt_graph = get_tftrt_graph(tf_graph, batch_size, 'FP32', dst_str[0])
  t0 = time.time()
  y_tftrt = run_graphdef(trt_graph, src_str, dst_str, x_test) 
  t1 = time.time()
  print('TensorRT time', t1 - t0)
  verify(y_tf, y_tftrt)

if __name__ == "__main__":
  main()
