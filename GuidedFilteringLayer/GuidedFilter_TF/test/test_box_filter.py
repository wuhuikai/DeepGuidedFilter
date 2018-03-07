import time

import numpy as np
import tensorflow as tf

from skimage import img_as_float
from skimage.io import imread

from guided_filter_tf.box_filter import box_filter

## BoxFilter
print('BoxFilter:')
# check forward
input = tf.constant(np.reshape(np.arange(1, 73), (1, 1, 8, 9)))
output = box_filter(input, 3)

with tf.Session() as sess:
    y = sess.run(output)

assert np.isclose(y.mean(),  1137.6,  0.1)
assert np.isclose(y.std(),    475.2,  0.1)
print('\tForward passed')
# forward on img
im = tf.constant(np.transpose(img_as_float(imread('test/rgb.jpg')), (2, 0, 1))[None])
output = box_filter(im, 64)
with tf.Session() as sess:
    start_time = time.time()
    r = sess.run(output)
    end_time = time.time()

print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time - start_time))
r = r.squeeze().transpose(1, 2, 0)
assert np.isclose(r[:, :, 0].mean(), 10305.0, 0.1)
assert np.isclose(r[:, :, 0].std(),  2206.4,  0.1)
assert np.isclose(r[:, :, 1].mean(), 7536.0,  0.1)
assert np.isclose(r[:, :, 1].std(),  2117.0,  0.1)
assert np.isclose(r[:, :, 2].mean(), 6203.0,  0.1)
assert np.isclose(r[:, :, 2].std(),  2772.3,  0.1)
print('\tPassed ...')
