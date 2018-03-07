#!/usr/bin/env python
# encoding: utf-8

import argparse
import numpy as np
import skimage
import skimage.io
import skimage.transform
import os
import time
import tensorflow as tf

import hdrnet.models as models
import hdrnet.utils as utils

LOOP = 100
TASK = 'auto_ps'
FULL_SIZE = 1536
SAVE_FOLDER = '../SU/time'

name = 'hdrnet'

def main(args):
  # -------- Load params ----------------------------------------------------
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)

    metapath = ".".join([checkpoint_path, "meta"])
    tf.train.import_meta_graph(metapath)

    model_params = utils.get_model_params(sess)

  # -------- Setup graph ----------------------------------------------------
  mdl = getattr(models, model_params['model_name'])

  tf.reset_default_graph()
  net_shape = model_params['net_input_size']
  t_fullres_input = tf.random_uniform((1, FULL_SIZE, FULL_SIZE, 3))
  t_lowres_input = tf.random_uniform((1, net_shape, net_shape, 3))

  with tf.variable_scope('inference'):
    prediction = mdl.inference(
        t_lowres_input, t_fullres_input, model_params, is_training=False)
  output = tf.cast(255.0*tf.squeeze(tf.clip_by_value(prediction, 0, 1)), tf.uint8)
  output = tf.reduce_sum(output)
  saver = tf.train.Saver()

  with tf.Session(config=config) as sess:
    saver.restore(sess, checkpoint_path)

    for _ in range(LOOP):
      sess.run(output)

    print 'Test {} ...'.format(name)

    t = time.time()
    for _ in range(LOOP):
      sess.run(output)
    mean_time = (time.time()-t)/float(LOOP)

    print '\tmean time: {}'.format(mean_time)

    # Log
    mode = 'a+' if os.path.isfile(os.path.join(SAVE_FOLDER, '{}_time.txt'.format(FULL_SIZE))) else 'w'
    with open(os.path.join(SAVE_FOLDER, '{}_time.txt'.format(FULL_SIZE)), mode) as f:
      f.write('{},{}\n'.format(name, mean_time))


if __name__ == '__main__':
  # -----------------------------------------------------------------------------
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_dir', default=None, help='path to the saved model variables')
  args = parser.parse_args()

  main(args)
