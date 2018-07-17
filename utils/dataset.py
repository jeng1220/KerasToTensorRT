import keras
from keras.datasets import mnist
from keras import backend as K
import numpy as np

def get_test_dataset():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  img_h = x_test.shape[1]
  img_w = x_test.shape[2]

  if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_h, img_w)
  else:
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)

  x_test = x_test.astype('float32')
  x_test /= 255
  return x_test, y_test

def verify(predict, ans):
  passed = 0
  num_test = ans.shape[0]

  for i in range(0, num_test):
    a = ans[i]
    p = np.argmax(predict[i])
    if (p == a) : passed = passed + 1

  if (float(passed) / num_test > 0.99) : print('PASSED')
  else : print('FAILURE', passed)

  p = np.argmax(predict[0])
  print('first inference result:', p, '\n\n')
