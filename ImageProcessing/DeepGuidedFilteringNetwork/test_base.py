import os

import torch

import numpy as np

from tqdm import tqdm

from dataset import SuDataset
from utils import tensor_to_img, calc_metric, Config

from skimage.io import imsave

default_config = Config(
    TASK = None,
    NAME = 'LR',
    SET_NAME = 1024,
    #################### CONSTANT #####################
    IMG = 'dataset',
    LIST = 'train_test_list',
    MODEL_PATH = 'checkpoints',
    RESULT_PATH = 'results',
    BATCH = 1,
    LOW_SIZE = 64,
    GPU = 0,
    # model
    model = None,
    # forward
    forward = None,
    # save paths
    save_paths = None,
    # compare paths
    compare_paths = None
)

def compare(pre_path, gt_path):
    print('{} v.s. {}'.format(pre_path, gt_path))
    results = calc_metric(pre_path, gt_path)
    avg_results = 'mse[{}]_psnr[{}]_ssim[{}]'.format(*results.mean(axis=0))

    np.save('{}_{}.npy'.format(pre_path, avg_results), results)

    print('\t'+avg_results)

def run(config):
    assert config.TASK is not None, 'Please set task name: TASK'

    assert config.save_paths is None     and config.compare_paths is None or \
           config.save_paths is not None and config.compare_paths is not None

    if config.save_paths is None:
        config.save_paths = [os.path.join(config.RESULT_PATH, config.TASK, config.NAME, 'PRE')]
    else:
        config.save_paths = [os.path.join(config.RESULT_PATH, config.TASK, config.NAME, path)
                                                                    for path in config.save_paths]

    for path in config.save_paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    # data set
    test_data = SuDataset(config.IMG,
                          os.path.join(
                              config.LIST,
                              config.TASK,
                              'test_{}.csv'.format(config.SET_NAME)
                          ),
                          low_size=config.LOW_SIZE)

    # GPU
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            config.model.cuda()

    # test
    i_bar = tqdm(total=len(test_data), desc='#Images')
    for idx, imgs in enumerate(test_data):
        name = os.path.basename(test_data.get_path(idx))

        imgs = config.forward(imgs, config)

        for path, img in zip(config.save_paths, imgs):
            imsave(os.path.join(path, name), tensor_to_img(img, transpose=True))

        i_bar.update()

    if config.compare_paths is None:
        compare(config.save_paths[0], os.path.join(config.IMG, config.TASK, str(config.SET_NAME)))
    else:
        for pre, gt in config.compare_paths:
            compare(pre, gt)