import os
import csv
import random
import argparse
from skimage.io import imread
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Precompute size')
parser.add_argument('--min', type=int, default=512, help='MIN_SIZE_TOTAL')
parser.add_argument('--random', action='store_true', default=False, help='RANDOM')
args = parser.parse_args()

RANDOM = args.random
MIN_SIZE_TOTAL = args.min
#####################################
SEED = 1
N_PROCESS = 12
MAX_SIZE_TOTAL = 2048
IN_PATH  = '../dataset/rgb/tiff'
SAVE_PATH = 'precomputed_size'
#####################################

if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

random.seed(SEED)
FILE_NAME = 'random' if RANDOM else str(MIN_SIZE_TOTAL)

def compute_size(name):
    # Read IN
    im = imread(os.path.join(IN_PATH, name))
    w, h = im.shape[:2]
    if RANDOM:
        min_size = min(w, h)
        size = int(min_size * random.uniform(
                                min(MIN_SIZE_TOTAL / min_size, 1.0),
                                min(MAX_SIZE_TOTAL / min_size, 1.0)))
    else:
        size = MIN_SIZE_TOTAL
    if w < h:
        width, height = (size, int(size / w * h))
    else:
        width, height = (int(size / h * w), size)

    return name, width, height

imgs_name = sorted(os.listdir(IN_PATH))
result = Pool(N_PROCESS).map(compute_size, imgs_name)
with open(os.path.join(SAVE_PATH, 'precomputed_size_{}.txt'.format(FILE_NAME)), 'w') as f:
    csv.writer(f).writerows(result)