from train_base import *

from module import DeepGuidedFilterAdvanced

default_config.NAME = 'HR_AD'
default_config.N_EPOCH = 150
default_config.DATA_SET = 512

# model
default_config.model = DeepGuidedFilterAdvanced()

def forward(imgs, config):
    x_hr, gt_hr, x_lr = imgs[:3]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x_hr, gt_hr, x_lr = x_hr.cuda(), gt_hr.cuda(), x_lr.cuda()

    return config.model(Variable(x_lr), Variable(x_hr)), gt_hr

default_config.forward = forward
default_config.clip = 0.01

run(default_config, keep_vis=True)

##########################################
default_config.N_START = default_config.N_EPOCH
default_config.N_EPOCH = 30
default_config.DATA_SET = 'random'
default_config.exceed_limit = lambda size: size[0]*size[1] > 2048**2

run(default_config)