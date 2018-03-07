from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from skimage.io import imsave
from torch.autograd import Variable

import models.dss as dss
from deepguidedfilter.guided_filter import FastGuidedFilter

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

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

CRF = True

if not os.path.isdir(opt.experiment):
    os.makedirs(opt.experiment)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

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

real_A = torch.FloatTensor()
real_x = torch.FloatTensor()
real_B = torch.FloatTensor()

if opt.dgf:
    dgf = FastGuidedFilter(opt.dgf_r, opt.dgf_eps)

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

    if CRF:
        img = np.ascontiguousarray(np.asarray(np.squeeze(data[0, :3, :, 0:w].cpu().numpy()).transpose(1, 2, 0) * 255, np.uint8))
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)

        image_B = image_B[0].numpy()
        lp2 = np.concatenate((1 - image_B, image_B), axis=0)
        d.setUnaryEnergy(unary_from_softmax(lp2, scale=None, clip=1e-5))

        d.addPairwiseGaussian(sxy=(5, 5), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=(60, 60), srgb=(8, 8, 8), rgbim=img, compat=3, kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        image_B = torch.from_numpy(np.reshape(np.array(d.inference(5))[1], image_B.shape))

    imsave(os.path.join(opt.experiment, '{}_sal.png'.format(idx)), image_B.mul(255).numpy().squeeze().astype(np.uint8))
    imsave(os.path.join(opt.experiment, '{}.png'.format(idx)), real_B.mul(255).cpu().numpy().squeeze().astype(np.uint8))

    test_list.append('{}_sal.png {}.png'.format(idx, idx))

with open(os.path.join(opt.experiment, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_list))
