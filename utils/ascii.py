import numpy as np


def print_ascii(img, img_h, img_w):
  assert(len(img.shape) > 2)

  res = img.reshape((img_h, img_w))
  for y in range(0, img_h):
    str=''
    for x in range(0, img_w):
      data = res[y][x]
      if (data > 0.5) : str += '@'
      else : str += ' '
    print(str)
  