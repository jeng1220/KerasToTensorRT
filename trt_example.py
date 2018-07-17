import keras
from keras.models import load_model
import keras.backend as K
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorflow as tf
import tensorrt as trt
import tensorrt.parsers.uffparser as uffparser
import time
import uff
import utils.ascii as helper
import utils.dataset as data
from pdb import set_trace

class TrtEngine():
  def __init__(self, model, batch_size):
    # get Tensorflow graph object from Keras
    with K.get_session() as sess:
      image_batch_t = tf.placeholder(tf.float32,
        shape=(None, 1, 28, 28),
        name='image_tensor')
      K.set_learning_phase(0)
      conf_t = model(image_batch_t)
      output_names = [conf_t.name[:-2]]
      graphdef = sess.graph.as_graph_def()      
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, output_names)
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    # convert TensorRT UFF object
    uff_model = uff.from_tensorflow(frozen_graph, output_names)
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    parser = uffparser.create_uff_parser()
    input_shape = (1, 28, 28)
    parser.register_input("image_tensor", input_shape, 0)
    parser.register_output(output_names[0])

    # create TensorRT inference engine
    engine = trt.utils.uff_to_trt_engine(G_LOGGER,
      stream=uff_model,
      parser=parser,
      max_batch_size=batch_size,
      max_workspace_size=1 << 25,
      datatype='FP32')

    parser.destroy()

    # allocate needed device buffers
    dims = engine.get_binding_dimensions(0).to_DimsCHW()
    nbytes = batch_size * dims.C() * dims.H() * dims.W() * np.dtype(np.float32).itemsize
    self.d_src  = cuda.mem_alloc(nbytes)

    dims = engine.get_binding_dimensions(1).to_DimsCHW()
    nbytes = batch_size * dims.C() * dims.H() * dims.W() * np.dtype(np.float32).itemsize
    self.d_dst  = cuda.mem_alloc(nbytes)

    self.engine = engine
    self.ctx    = engine.create_execution_context()
    self.batch_size = batch_size


  def save(self, str):
    trt.utils.write_engine_to_file(
      str, self.engine.serialize())

  def infer(self, x):
    # allocate destination host buffer
    batch_size = self.batch_size
    num_test = x.shape[0]
    dims = self.engine.get_binding_dimensions(1).to_DimsCHW()
    output_size = dims.C() * dims.H() * dims.W()

    y = np.empty((num_test, output_size), np.float32)    

    bindings = [int(self.d_src), int(self.d_dst)]
    # do inference
    for i in range(0, num_test, batch_size):
      x_part = x[i : i + batch_size]
      y_part = y[i : i + batch_size]
      cuda.memcpy_htod(self.d_src, x_part)
      self.ctx.execute(batch_size, bindings)
      cuda.memcpy_dtoh(y_part, self.d_dst)

    return y

def main():

  # load pre-trained model
  K.set_image_data_format('channels_first')
  model = load_model("nchw_model.h5")
  model.summary()
  
  # load mnist dataset
  x_test, y_test = data.get_test_dataset()
  img_h = x_test.shape[2]
  img_w = x_test.shape[3]
  helper.print_ascii(x_test[0], img_h, img_w)

  # use Keras to infer
  t0 = time.time()
  y_keras = model.predict(x_test)
  t1 = time.time()
  print('Keras time', t1 - t0)
  data.verify(y_keras, y_test)

  # use TensorRT to infer
  engine = TrtEngine(model, 1000)

  t0 = time.time()
  y_trt = engine.infer(x_test)
  t1 = time.time()
  print('TensorRT time', t1 - t0)
  data.verify(y_trt, y_test)

  engine.save('nchw_engine.bin')

if __name__ == "__main__":
  main()
