import os
import argparse

import numpy as np

from skimage.color import grey2rgb
from skimage.io import imread, imsave

parser = argparse.ArgumentParser(description='Make A-B pairs.')
parser.add_argument('--data_path',  type=str, required=True,   help='DATA_ROOT')
parser.add_argument('--mode',       type=str, default='train', help='train|valid|test')
args = parser.parse_args()

ROOT = args.data_path
file = 'list/{}_cvpr2013.txt'.format(args.mode)
target_folder = 'AB/{}/{}'.format(args.mode, args.mode)

target_folder = os.path.join(ROOT, target_folder)
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

with open(file) as f:
    for line in f:
        line = line.strip()

        A = imread(os.path.join(ROOT, line.replace('.png', '.jpg')))
        B = grey2rgb(imread(os.path.join(ROOT, line)))
        out = np.hstack([A, B])

        imsave(os.path.join(target_folder, line), out)
