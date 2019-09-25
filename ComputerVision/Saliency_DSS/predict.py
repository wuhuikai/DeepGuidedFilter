from __future__ import print_function

import os
import argparse

import numpy as np

import torch.utils.data
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from PIL import Image
from skimage.io import imsave

from torch.autograd import Variable

import models.dss as dss

from guided_filter_pytorch.guided_filter import GuidedFilter


parser = argparse.ArgumentParser()
parser.add_argument('--im_path', required=True, help="path to image")
parser.add_argument('--netG', required=True, help="path to netG (to continue training)")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nn_dgf', action='store_true', help='enables dgf')
parser.add_argument('--nn_dgf_r', type=int, default=8, help='dgf radius')
parser.add_argument('--nn_dgf_eps', type=float, default=1e-2, help='dgf eps')
parser.add_argument('--post_sigmoid', action='store_true', help='sigmoid after dgf')
parser.add_argument('--dgf', action='store_true', help='enables dgf')
parser.add_argument('--dgf_r', type=int, default=8, help='dgf radius')
parser.add_argument('--dgf_eps', type=float, default=1e-2, help='dgf eps')
parser.add_argument('--thres', type=int, default=161, help='clip by threshold')
opt = parser.parse_args()
print(opt)

cudnn.benchmark = True
netG = dss.network_dss(3, opt.nn_dgf, opt.nn_dgf_r, opt.nn_dgf_eps, opt.post_sigmoid)
netG.load_state_dict(torch.load(opt.netG))
if opt.dgf:
    dgf = GuidedFilter(opt.dgf_r, opt.dgf_eps)
if opt.cuda:
    netG = netG.cuda()
    if opt.dgf:
        dgf = dgf.cuda()

img = transforms.ToTensor()(Image.open(opt.im_path).convert('RGB'))
real_x = img.unsqueeze(0)
real_A = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
real_A.unsqueeze_(0)
input_G = Variable(real_A)
input_x = Variable(real_x)

if opt.cuda:
    input_G = input_G.cuda()
    input_x = input_x.cuda()

with torch.no_grad():
    fake_B, side_out1, side_out2, side_out3, side_out4, side_out5, side_out6 = netG(input_G, input_x)
    if opt.dgf:
        input_x = input_x.sum(1, keepdim=True)
        image_B = dgf(input_x, fake_B).clamp(0, 1)
    else:
        image_B = fake_B

    image_B = image_B.data.cpu().mul(255).numpy().squeeze().astype(np.uint8)
    if opt.thres > 0:
        image_B[image_B >= opt.thres] = 255
        image_B[image_B <= opt.thres] = 0

    output_directory = os.path.dirname(opt.im_path)
    output_name = os.path.splitext(os.path.basename(opt.im_path))[0]
    save_path = os.path.join(output_directory, '{}_labels.png'.format(output_name))
    imsave(save_path, image_B)
