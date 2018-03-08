from torch.autograd import Variable

from test_base import *
from module import build_lr_net

default_config.NAME = 'LR'
default_config.save_paths = ['PRE', 'IN', 'GT']
result_path = os.path.join(default_config.RESULT_PATH, default_config.TASK, default_config.NAME)
default_config.compare_paths = [(os.path.join(result_path, 'PRE'), os.path.join(result_path, 'GT')),
                                (os.path.join(result_path, 'IN'),  os.path.join(result_path, 'GT'))]

# model
model = build_lr_net()
model_path = os.path.join(default_config.MODEL_PATH, default_config.TASK, 'LR', 'snapshots', 'net_latest.pth')
model.load_state_dict(torch.load(model_path))
default_config.model = model

def forward(imgs, config):
    input = imgs[2].unsqueeze(0)
    if config.GPU >= 0:
        with torch.cuda.device(config.GPU):
            input = input.cuda()

    return config.model(Variable(input)).data.cpu(), imgs[2], imgs[3]

default_config.forward = forward

run(default_config)