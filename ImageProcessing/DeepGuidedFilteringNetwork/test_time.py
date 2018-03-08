import os
import time
import argparse

import torch
from torch.autograd import Variable

from module import DeepGuidedFilter, DeepGuidedFilterAdvanced, FastGuidedFilter

parser = argparse.ArgumentParser(description='Runing time')
parser.add_argument('--gpu',       type=int, default=   0, help='GPU')
parser.add_argument('--low_size',  type=int, default=  64, help='LOW_SIZE')
parser.add_argument('--full_size', type=int, default=2048, help='FULL_SIZE')
parser.add_argument('--iter_size', type=int, default= 100, help='TOTAL_ITER')
parser.add_argument('--model_id',  type=int, default=   0, help='MODEL_ID')
args = parser.parse_args()

SAVE_FOLDER = 'time'
GPU = args.gpu
LOW_SIZE = args.low_size
FULL_SIZE = args.full_size
TOTAL_ITER = args.iter_size
MODEL_ID = args.model_id

# model - forward
model_forward = [
    ('deep_guided_filter', (DeepGuidedFilter(), lambda model, imgs: model(imgs[0], imgs[1]))),
    ('deep_guided_filter_layer', (FastGuidedFilter(1, 1e-8), lambda model, imgs: model(imgs[0], imgs[0], imgs[1]))),
    ('deep_guided_filter_advanced', (DeepGuidedFilterAdvanced(), lambda model, imgs: model(imgs[0], imgs[1]))),
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
