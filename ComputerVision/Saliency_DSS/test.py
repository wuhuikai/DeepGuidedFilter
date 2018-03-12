from __future__ import print_function

import os
import argparse

import numpy as np

import torch.utils.data
import torch.backends.cudnn as cudnn

import torchvision.datasets as dset
import torchvision.transforms as transforms

from skimage.io import imsave

from torch.autograd import Variable

import models.dss as dss

from guided_filter_pytorch.guided_filter import GuidedFilter


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--netG', required=True, help="path to netG (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nn_dgf', action='store_true', help='enables dgf')
parser.add_argument('--nn_dgf_r', type=int, default=8, help='dgf radius')
parser.add_argument('--nn_dgf_eps', type=float, default=1e-2, help='dgf eps')
parser.add_argument('--post_sigmoid', action='store_true', help='sigmoid after dgf')
parser.add_argument('--dgf', action='store_true', help='enables dgf')
parser.add_argument('--dgf_r', type=int, default=8, help='dgf radius')
parser.add_argument('--dgf_eps', type=float, default=1e-2, help='dgf eps')
parser.add_argument('--experiment', default='results', help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

if not os.path.isdir(opt.experiment):
    os.makedirs(opt.experiment)

cudnn.benchmark = True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.ToTensor()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=int(opt.workers))

netG = dss.network_dss(3, opt.nn_dgf, opt.nn_dgf_r, opt.nn_dgf_eps, opt.post_sigmoid)
netG.load_state_dict(torch.load(opt.netG))
if opt.dgf:
    dgf = GuidedFilter(opt.dgf_r, opt.dgf_eps)
real_A = torch.FloatTensor()
real_x = torch.FloatTensor()
real_B = torch.FloatTensor()
if opt.cuda:
    netG = netG.cuda()
    if opt.dgf:
        dgf = dgf.cuda()

    real_A = real_A.cuda()
    real_x = real_x.cuda()
    real_B = real_B.cuda()

test_list = []
for idx, (data, _) in enumerate(iter(dataloader)):
    print(idx)
    if opt.cuda:
        data = data.cuda()

    w = data.size(3) // 2
    h = data.size(2)
    real_B.resize_(h, w).copy_(data[0, 0, :, w:])
    real_A.resize_(3, h, w).copy_(data[0, :3, :, 0:w])
    real_A = normalize(real_A)
    real_A.unsqueeze_(0)

    real_x.resize_(1, 3, h, w).copy_(data[:1, :3, :, 0:w])

    input_G = Variable(real_A, volatile=True)
    input_x = Variable(real_x, volatile=True)
    fake_B, side_out1, side_out2, side_out3, side_out4, side_out5, side_out6 = netG(input_G, input_x)

    if opt.dgf:
        input_x = input_x.sum(1, keepdim=True)
        image_B = dgf(input_x, fake_B).clamp(0, 1)
    else:
        image_B = fake_B
    image_B = image_B.data.cpu()

    imsave(os.path.join(opt.experiment, '{}_sal.png'.format(idx)), image_B.mul(255).numpy().squeeze().astype(np.uint8))
    imsave(os.path.join(opt.experiment, '{}.png'.format(idx)), real_B.mul(255).cpu().numpy().squeeze().astype(np.uint8))

    test_list.append('{}_sal.png {}.png'.format(idx, idx))

with open(os.path.join(opt.experiment, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_list))
