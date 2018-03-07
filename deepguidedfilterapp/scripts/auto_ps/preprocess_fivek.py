import io
import os
import glob

import warnings
warnings.simplefilter('ignore')

import numpy as np

import rawpy

from PIL import Image
from PIL import ImageCms

from skimage.io import imsave

from multiprocessing import Pool

RAW_PATH = '../../dataset/fivek/raw_photos'
GT_PATH  = '../../dataset/fivek/gts'
RGB_SAVE = '../../dataset/rgb/tiff'
GT_SAVE  = '../../dataset/auto_ps/tiff'
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
    gt_path  = os.path.join(GT_SAVE,  '{}.tif'.format(name))

    if os.path.isfile(rgb_path) and os.path.isfile(gt_path):
        return

    # Read RAW Image
    rgb = rawpy.imread(path).postprocess()

    # Read GT
    try:
        gt = Image.open(os.path.join(GT_PATH, '{}.tif'.format(name)))
        gt = ImageCms.profileToProfile(gt,
                                       io.BytesIO(gt.info.get('icc_profile')),
                                       ImageCms.createProfile('sRGB'))
    except Exception as e:
        print(rgb_path)
        return

    gt = np.asarray(gt)
    w_rgb, h_rgb = rgb.shape[:2]
    w_gt,  h_gt  =  gt.shape[:2]

    if w_rgb != w_gt or h_rgb != h_gt:
        w, h = min(w_rgb, w_gt), min(h_rgb, h_gt)

        rgb = center_crop(rgb, w, h)
        gt  = center_crop(gt,  w, h)

        w_rgb, h_rgb = rgb.shape[:2]
        w_gt, h_gt = gt.shape[:2]

    assert w_rgb == w_gt and h_rgb == h_gt

    # Color
    imsave(rgb_path, rgb)
    # GT
    imsave(gt_path, gt)

if not os.path.isdir(RGB_SAVE):
    os.makedirs(RGB_SAVE)
if not os.path.isdir(GT_SAVE):
    os.makedirs(GT_SAVE)

imgs_path = sorted(glob.glob(os.path.join(RAW_PATH, 'HQ*', 'photos', '*.dng')))

Pool(N_PROCESS).map(preprocess, imgs_path)