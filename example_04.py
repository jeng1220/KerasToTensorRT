import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, merge, Concatenate, Conv2D, MaxPooling2D
from keras import backend as K
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
  epochs = 4
  model.fit([x_train_a, x_train_b], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=([x_test_a, x_test_b], y_test))
  score = model.evaluate([x_test_a, x_test_b], y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  model.save(filename)

  helper.print_ascii(x_test_a[0], img_h, img_w)
  helper.print_ascii(x_test_b[0], img_h, img_w)
  print(np.argmax(y_test[0]))

if __name__ == '__main__':
  main()