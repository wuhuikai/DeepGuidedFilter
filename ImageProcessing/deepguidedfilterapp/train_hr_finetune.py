from train_base import *

from module import DeepGuidedFilter

default_config.NAME = 'HR_FT'
default_config.N_EPOCH = 30
default_config.DATA_SET = 'random'

# model
default_config.model = DeepGuidedFilter()
default_config.model.init_lr(os.path.join(default_config.SAVE, default_config.TASK, 'LR', 'snapshots', 'net_latest.pth'))

def forward(imgs, config):
    x_hr, gt_hr, x_lr = imgs[:3]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x_hr, gt_hr, x_lr = x_hr.cuda(), gt_hr.cuda(), x_lr.cuda()

    return config.model(Variable(x_lr), Variable(x_hr)), gt_hr

default_config.forward = forward
default_config.exceed_limit = lambda size: size[0]*size[1] > 2048**2
default_config.clip = 0.01

run(default_config)