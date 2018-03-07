import time

import numpy as np
import tensorflow as tf

from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

from guided_filter_tf.guided_filter import fast_guided_filter

## GuidedFilter
print('FastGuidedFilter:')
## check forward
# forward on img
rgb = img_as_float(imread('test/rgb.jpg'))
gt  = img_as_float(imread('test/gt.jpg'))
x_w, x_h = rgb.shape[:2]
w, h = x_w // 8, x_h // 8
lr_rgb = resize(rgb, (w, h), order=0, mode='reflect')
lr_gt  = resize(gt,  (w, h), order=0, mode='reflect')
lr_x, lr_y, hr_x = [tf.constant(i.transpose((2, 0, 1))[None], dtype=tf.float32) for i in [lr_rgb, lr_gt, gt]]
output = fast_guided_filter(lr_x, lr_y, hr_x, 8, 0)

with tf.Session() as sess:
    start_time = time.time()
    r = sess.run(output)
    end_time = time.time()
print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time - start_time))

r = r.squeeze().transpose(1, 2, 0)
r = np.asarray(r.clip(0, 1) * 255, dtype=np.uint8)
imsave('test/r_fast.jpg', r)