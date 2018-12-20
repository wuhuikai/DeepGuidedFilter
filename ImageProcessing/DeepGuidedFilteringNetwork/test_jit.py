import os
import argparse

import numpy as np

import torch

from torch.nn import functional as F

from PIL import Image
from torchvision import transforms


def to_tensor(x):
    return transforms.ToTensor()(x).unsqueeze(0)


def to_img(tensor):
    return Image.fromarray(np.asarray(np.clip(np.squeeze(tensor.data.cpu().numpy()) * 255, 0, 255), dtype=np.uint8).transpose((1, 2, 0)))


parser = argparse.ArgumentParser(description='JIT')
parser.add_argument('--task', type=str, default='auto_ps', help='TASK')
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

model = torch.jit.load(os.path.join('models', args.task, 'hr_net_latest_jit.pth'))

im_hr = Image.open(args.image).convert('RGB')
im_lr = transforms.Resize(64, interpolation=Image.NEAREST)(im_hr)
im_hr = to_tensor(im_hr)
im_lr = to_tensor(im_lr)

A, b = model(im_lr, 3, 1e-5)
_, _, h_hrx, w_hrx = im_hr.size()
mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')
y = torch.clamp(mean_A*im_hr+mean_b, 0, 1)

to_img(y).save('images/{}_result.jpg'.format(args.task))
