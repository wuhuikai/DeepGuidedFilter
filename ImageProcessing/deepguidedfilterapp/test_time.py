import os
import time

import torch
from torch.autograd import Variable

from module import build_lr_net, DeepGuidedFilter, DeepGuidedFilterAdvanced, DeepGuidedFilterDJF, FastGuidedFilter

from utils import task_name

TASK = task_name()

SAVE_FOLDER = 'time'
GPU = 0
LOW_SIZE = 64
FULL_SIZE = 1536
TOTAL_ITER = 100
MODEL_ID = 0

# model - forward
model_forward = [
    ('deep_guided_filter_layer', (FastGuidedFilter(1, 1e-8), lambda model, imgs: model(imgs[0], imgs[0], imgs[1]))),
    ('deep_guided_filter', (DeepGuidedFilter(), lambda model, imgs: model(imgs[0], imgs[1]))),
    ('deep_guided_filter_advanced', (DeepGuidedFilterAdvanced(), lambda model, imgs: model(imgs[0], imgs[1]))),
    ('deep_guided_filter_advanced_djf', (DeepGuidedFilterDJF(), lambda model, imgs: model(imgs[0], imgs[1]))),
    ('baseline', (build_lr_net(), lambda model, imgs: model(imgs[1])))
]

# mkdir
if not os.path.isdir(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# prepare img
imgs = [torch.rand((1, 3, LOW_SIZE, LOW_SIZE)), torch.rand((1, 3, FULL_SIZE, FULL_SIZE))]
if GPU >= 0:
    with torch.cuda.device(GPU):
        imgs = [img.cuda() for img in imgs]
imgs = [Variable(img, requires_grad=False) for img in imgs]

# prepare model
name, (model, forward) = model_forward[MODEL_ID]
if GPU >= 0:
    with torch.cuda.device(GPU):
        model = model.cuda()
model.eval()

# Warm up
for _ in range(TOTAL_ITER):
    forward(model, imgs)

# Test
print('Test {} ...'.format(name))

t = time.time()
for _ in range(TOTAL_ITER):
    forward(model, imgs)
mean_time = (time.time()-t)/TOTAL_ITER

print('\tmean time: {}'.format(mean_time))

# Log
file_name = '{}_time.txt' if GPU >= 0 else '{}_time_cpu.txt'
mode = 'a+' if os.path.isfile(os.path.join(SAVE_FOLDER, file_name.format(FULL_SIZE))) else 'w'
with open(os.path.join(SAVE_FOLDER, file_name.format(FULL_SIZE)), mode) as f:
    f.write('{},{}\n'.format(name, mean_time))
