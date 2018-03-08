from train_base import *

from module import build_lr_net

default_config.NAME = 'LR'
default_config.N_EPOCH = 150
default_config.DATA_SET = 512

# model
default_config.model = build_lr_net()

def forward(imgs, config):
    x, y = imgs[2:]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x, y = x.cuda(), y.cuda()

    return config.model(Variable(x)), y

default_config.forward = forward

run(default_config)