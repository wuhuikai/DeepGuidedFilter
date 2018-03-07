import os
from shutil import copyfile

from tqdm import tqdm

from test_base import compare, default_config

default_config.NAME = 'HR_INPUT'
default_config.SET_NAME = 1024

result_path = os.path.join(default_config.RESULT_PATH, default_config.TASK, default_config.NAME, 'PRE')
if not os.path.isdir(result_path):
    os.makedirs(result_path)

# data set
with open(os.path.join(default_config.LIST, default_config.TASK, 'test_{}.csv'.format(default_config.SET_NAME))) as f:
    imgs = sorted([line.strip().split(',') for line in f])
imgs = [[os.path.join(default_config.IMG, p) for p in group] for group in imgs]

# test
i_bar = tqdm(total=len(imgs), desc='#Images')
for rgb_path, _ in imgs:
    name = os.path.basename(rgb_path)
    save_path = os.path.join(result_path, name)

    copyfile(rgb_path, save_path)

    i_bar.update()

compare(result_path, os.path.join(default_config.IMG, default_config.TASK, str(default_config.SET_NAME)))