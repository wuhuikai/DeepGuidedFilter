import io
import os
import glob

import warnings
warnings.simplefilter('ignore')

import numpy as np

import PIL
from PIL import Image
from PIL import ImageCms

from skimage.io import imread, imsave

from multiprocessing import Pool

RGB_PATH = '../../dataset/rgb/tiff'
GT_PATH  = '../../dataset/fivek/gts'
GT_SAVE  = '../../dataset/auto_ps/tiff'
N_PROCESS = 48

def preprocess(path):
    # IMG NAME
    name = os.path.splitext(os.path.basename(path))[0]
    gt_path  = os.path.join(GT_SAVE,  '{}.tif'.format(name))

    if os.path.isfile(gt_path):
        return

    rgb = imread(path)

    # Read GT
    try:
        gt = Image.open(os.path.join(GT_PATH, '{}.tif'.format(name)))
        gt = ImageCms.profileToProfile(gt,
                                       io.BytesIO(gt.info.get('icc_profile')),
                                       ImageCms.createProfile('sRGB'))
    except Exception as e:
        print(path)
        return

    w_rgb, h_rgb = rgb.shape[:2]
    h_gt,  w_gt  =  gt.size

    print(h_gt, w_gt)
    print(h_rgb, w_rgb)

    if w_rgb != w_gt or h_rgb != h_gt:
        gt  = gt.resize((h_rgb, w_rgb), resample=PIL.Image.BICUBIC)

        w_rgb, h_rgb = rgb.shape[:2]
        h_gt, w_gt = gt.size

    assert w_rgb == w_gt and h_rgb == h_gt

    # GT
    imsave(gt_path, np.asarray(gt))

if not os.path.isdir(GT_SAVE):
    os.makedirs(GT_SAVE)

imgs_path = sorted(glob.glob(os.path.join(RGB_PATH, '*.tif')))

Pool(N_PROCESS).map(preprocess, imgs_path)