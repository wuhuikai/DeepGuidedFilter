import os
import csv
import argparse
import warnings
warnings.simplefilter('ignore')

from skimage import img_as_uint
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize

from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Resize Image')
parser.add_argument('--file_name', type=str, default='512', help='FILE_NAME')
parser.add_argument('--task', type=str, default='rgb', help='TASK')
args = parser.parse_args()

FILE_NAME = args.file_name
TASK = args.task
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

with open('precomputed_size/precomputed_size_{}.txt'.format(FILE_NAME)) as f:
    reader = csv.reader(f)
    Pool(N_PROCESS).starmap(prepare_dataset, reader)