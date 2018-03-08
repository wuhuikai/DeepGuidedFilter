import copy
import argparse

from train_base import *

from module import build_lr_net

parser = argparse.ArgumentParser(description='Train Deep Guided Filtering Networks')
parser.add_argument('--task',  type=str, default='l0_smooth',          help='TASK')
args = parser.parse_args()

config = copy.deepcopy(default_config)

config.TASK = args.task
config.NAME = 'LR'
config.N_EPOCH = 150
config.DATA_SET = 512

# model
config.model = build_lr_net()

def forward(imgs, config):
    x, y = imgs[2:]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x, y = x.cuda(), y.cuda()

    return config.model(Variable(x)), y

config.forward = forward

run(config)