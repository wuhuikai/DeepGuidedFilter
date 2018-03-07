import os
import csv

import warnings
warnings.simplefilter('ignore')

from skimage import img_as_uint
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from multiprocessing import Pool

FILE_NAME = '512'
TASK = 'rgb'
#######################################
N_PROCESS = 12
ROOT  = '../dataset/{}'.format(TASK)

def prepare_dataset(name, w, h):
    # IMG NAME
    save_path = os.path.join(SAVE, name)

    if os.path.isfile(save_path):
        return

    # Read IN
    rgb = resize(imread(os.path.join(ROOT, 'tiff', name)), (int(w), int(h)), mode='reflect')

    imsave(save_path, img_as_uint(rgb))

SAVE = os.path.join(ROOT, FILE_NAME)
if not os.path.isdir(SAVE):
    os.makedirs(SAVE)

with open('precomputed_size_{}.txt'.format(FILE_NAME)) as f:
    reader = csv.reader(f)
    Pool(N_PROCESS).starmap(prepare_dataset, reader)