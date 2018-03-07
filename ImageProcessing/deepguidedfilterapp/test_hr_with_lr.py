from torch.autograd import Variable

from module import build_lr_net

from test_base import *

default_config.NAME = 'HR_WITH_LR'
default_config.SET_NAME = 1024

# model
model = build_lr_net()
model_path = os.path.join(default_config.MODEL_PATH, default_config.TASK, 'LR',
                          'snapshots', 'net_latest.pth')
model.load_state_dict(torch.load(model_path))
default_config.model = model
# forward
def forward(imgs, config):
    input = imgs[0].unsqueeze(0)
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            input = input.cuda()

    return model(Variable(input)).data.cpu()
default_config.forward = forward

run(default_config)