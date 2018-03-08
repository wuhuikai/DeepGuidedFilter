import os
import glob

import warnings
warnings.simplefilter('ignore')

import rawpy

from skimage.io import imsave

from multiprocessing import Pool

RAW_PATH = '../dataset/fivek/raw_photos'
RGB_SAVE = '../dataset/rgb/tiff'
N_PROCESS = 48

def center_crop(im, tw, th):
    w, h = im.shape[:2]
    if w == tw and h == th:
        return im

    w_p = int(round((w - tw) / 2))
    h_p = int(round((h - th) / 2))

    return im[w_p:w_p+tw, h_p:h_p+th]

def preprocess(path):
    # IMG NAME
    name = os.path.splitext(os.path.basename(path))[0]
    rgb_path = os.path.join(RGB_SAVE, '{}.tif'.format(name))

    if os.path.isfile(rgb_path):
        return

    # Read RAW Image
    rgb = rawpy.imread(path).postprocess()

    # Color
    imsave(rgb_path, rgb)

if not os.path.isdir(RGB_SAVE):
    os.makedirs(RGB_SAVE)

imgs_path = sorted(glob.glob(os.path.join(RAW_PATH, 'HQ*', 'photos', '*.dng')))

Pool(N_PROCESS).map(preprocess, imgs_path)