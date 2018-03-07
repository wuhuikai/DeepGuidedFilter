import time

import numpy as np
import tensorflow as tf

from skimage import img_as_float
from skimage.io import imread, imsave

from guided_filter_tf.guided_filter import guided_filter

## GuidedFilter
print('GuidedFilter:')
## check forward
# forward on img
rgb = img_as_float(imread('test/rgb.jpg'))
gt  = img_as_float(imread('test/gt.jpg'))
x, y = [tf.constant(i.transpose((2, 0, 1))[None]) for i in [rgb, gt]]
output = guided_filter(x, y, 64, 0)

with tf.Session() as sess:
    start_time = time.time()
    r = sess.run(output)
    end_time = time.time()
print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time - start_time))

r = r.squeeze().transpose(1, 2, 0)
r = np.asarray(r.clip(0, 1) * 255, dtype=np.uint8)
imsave('test/r.jpg', r)