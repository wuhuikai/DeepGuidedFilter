import copy
import argparse

from train_base import *

from module import DeepGuidedFilter, DeepGuidedFilterAdvanced

parser = argparse.ArgumentParser(description='Finetune Deep Guided Filtering Networks')
parser.add_argument('--task',  type=str, default='l0_smooth',          help='TASK')
parser.add_argument('--name',  type=str, default='HR_FT',                 help='NAME')
parser.add_argument('--model', type=str, default='deep_guided_filter', help='model')
args = parser.parse_args()

config = copy.deepcopy(default_config)

config.TASK = args.task
config.NAME = args.name
config.N_EPOCH = 30
config.DATA_SET = 'random'

# model
if args.model == 'deep_guided_filter':
    config.model = DeepGuidedFilter()
elif args.model == 'deep_guided_filter_advanced':
    config.model = DeepGuidedFilterAdvanced()
else:
    print('Not a valid model!')
    exit(-1)
config.model.init_lr(os.path.join(config.SAVE, config.TASK, 'LR', 'snapshots', 'net_latest.pth'))

def forward(imgs, config):
    x_hr, gt_hr, x_lr = imgs[:3]
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            x_hr, gt_hr, x_lr = x_hr.cuda(), gt_hr.cuda(), x_lr.cuda()

    return config.model(Variable(x_lr), Variable(x_hr)), gt_hr

config.forward = forward
config.exceed_limit = lambda size: size[0]*size[1] > 2048**2
config.clip = 0.01

run(config)