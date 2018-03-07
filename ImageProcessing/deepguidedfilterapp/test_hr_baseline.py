from torch.autograd import Variable

from module import build_lr_net

from test_base import *

default_config.NAME = 'HR_BASE'
default_config.SET_NAME = 1024

# model
model = build_lr_net()
model.load_state_dict(
    torch.load(
        os.path.join(default_config.MODEL_PATH,
                     default_config.TASK,
                     default_config.NAME,
                     'snapshots',
                     'net_latest.pth')
    )
)
default_config.model = model

# forward
def forward(imgs, config):
    hr_x = imgs[0].unsqueeze(0)
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            hr_x = hr_x.cuda()

    return config.model(Variable(hr_x)).data.cpu()
default_config.forward = forward

run(default_config)
