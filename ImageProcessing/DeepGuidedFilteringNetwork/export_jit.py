import os
import argparse

import torch

from module import DeepGuidedFilter

parser = argparse.ArgumentParser(description='JIT')
parser.add_argument('--task', type=str, default='auto_ps', help='TASK')
args = parser.parse_args()


model = DeepGuidedFilter()
model_path = os.path.join('models', args.task, 'hr_net_latest.pth')
model.load_state_dict(torch.load(model_path), strict=False)
model.save(os.path.join('models', args.task, 'hr_net_latest_jit.pth'))
