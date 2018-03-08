import os
import numpy as np

import warnings
warnings.simplefilter('ignore')

from multiprocessing import Pool

from skimage import img_as_ubyte
from skimage.io import imread
from skimage.color import grey2rgb
from skimage.transform import resize
from skimage.measure import compare_mse, compare_psnr, compare_ssim

class Config(object):
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v

def tensor_to_img(tensor, transpose=False):
    im = np.asarray(np.clip(np.squeeze(tensor.numpy()) * 255, 0, 255), dtype=np.uint8)
    if transpose:
        im = im.transpose((1, 2, 0))

    return im

def calc_metric_with_np(pre_im, gt_im, multichannel=True):
    return compare_mse(pre_im, gt_im),\
           compare_psnr(pre_im, gt_im),\
           compare_ssim(pre_im, gt_im, multichannel=multichannel)

def calc_metric_per_img(im_name, pre_path, gt_path, opts={}):
    pre_im_path = os.path.join(pre_path, im_name)
    gt_im_path  = os.path.join(gt_path,  im_name)

    assert os.path.isfile(pre_im_path)
    assert os.path.isfile(gt_im_path) or os.path.islink(gt_im_path)

    pre = img_as_ubyte(imread(pre_im_path))
    gt  = img_as_ubyte(imread(gt_im_path))
    if gt.ndim == 2:
        gt = grey2rgb(gt)
    if pre.shape != gt.shape:
        gt = img_as_ubyte(resize(gt, pre.shape[:2], mode='reflect'))

    return calc_metric_with_np(pre, gt, **opts)

def calc_metric(pre_path, gt_path, n_process=8):
    params = [(im_name, pre_path, gt_path) for im_name in os.listdir(pre_path)]
    return np.asarray(Pool(n_process).starmap(calc_metric_per_img, params))