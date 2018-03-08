import copy
import argparse

from torch.autograd import Variable

from module import DeepGuidedFilter, DeepGuidedFilterAdvanced

from test_base import *

parser = argparse.ArgumentParser(description='Evaluate Deep Guided Filtering Networks')
parser.add_argument('--task',  type=str, default='l0_smooth',          help='TASK')
parser.add_argument('--name',  type=str, default='HR',                 help='NAME')
parser.add_argument('--model', type=str, default='deep_guided_filter', help='model')
args = parser.parse_args()

config = copy.deepcopy(default_config)

config.TASK = args.task
config.NAME = args.name
config.SET_NAME = 1024

# model
if args.model in ['guided_filter', 'deep_guided_filter']:
    model = DeepGuidedFilter()
elif args.model == 'deep_guided_filter_advanced':
    model = DeepGuidedFilterAdvanced()
else:
    print('Not a valid model!')
    exit(-1)

if args.model in ['deep_guided_filter', 'deep_guided_filter_advanced']:
    model.load_state_dict(
        torch.load(
            os.path.join(config.MODEL_PATH,
                         config.TASK,
                         config.NAME,
                         'snapshots',
                         'net_latest.pth')
        )
    )
elif args.model == 'guided_filter':
    model.init_lr(os.path.join(config.MODEL_PATH, config.TASK, 'LR', 'snapshots', 'net_latest.pth'))
else:
    print('Not a valid model!')
    exit(-1)

config.model = model

# forward
def forward(imgs, config):
    lr_x, hr_x = imgs[2].unsqueeze(0), imgs[0].unsqueeze(0)
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            lr_x = lr_x.cuda()
            hr_x = hr_x.cuda()

    return config.model(Variable(lr_x), Variable(hr_x)).data.cpu()
config.forward = forward

run(config)