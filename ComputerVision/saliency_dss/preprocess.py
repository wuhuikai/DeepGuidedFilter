import os

import numpy as np
from skimage.color import grey2rgb
from skimage.io import imread, imsave

ROOT = '/home/data3/wuhuikai/MSRA_B'
file = 'test_cvpr2013.txt'
target_folder = 'AB/test/test'

target_folder = os.path.join(ROOT, target_folder)
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

with open(os.path.join(ROOT, file)) as f:
    for line in f:
        line = line.strip()

        A = imread(os.path.join(ROOT, 'MSRA_B', line.replace('.png', '.jpg')))
        B = grey2rgb(imread(os.path.join(ROOT, 'annotation', line)))
        out = np.hstack([A, B])

        imsave(os.path.join(target_folder, line), out)
