from torch.autograd import Variable

from guided_filter.guided_filter import FastGuidedFilter

from test_base import *

default_config.NAME = 'HR_LOW_GT'
default_config.SET_NAME = 1024

# model
model = FastGuidedFilter(1, 1e-8)
default_config.model = model

# forward
def forward(imgs, config):
    lr_x, lr_y, hr_x = imgs[2].unsqueeze(0), imgs[3].unsqueeze(0), imgs[0].unsqueeze(0)
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            lr_x = lr_x.cuda()
            lr_y = lr_y.cuda()
            hr_x = hr_x.cuda()

    return config.model(Variable(lr_x), Variable(lr_y), Variable(hr_x)).data.cpu()
default_config.forward = forward

run(default_config)
