from __future__ import print_function

import os
import math
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs

import torchvision.datasets as dset
import torchvision.models as visionmodels
import torchvision.transforms as transforms

from torch.autograd import Variable

import models.dss as dss

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.00001, help='learning rate for Generator, default=0.00005')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--dgf', action='store_true', help='enables dgf')
parser.add_argument('--dgf_r', type=int, default=8, help='dgf radius')
parser.add_argument('--dgf_eps', type=float, default=1e-2, help='dgf eps')
parser.add_argument('--experiment', default='checkpoints', help='Where to store samples and models')
opt = parser.parse_args()
print(opt)


def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.Conv2d):
            n = p.kernel_size[0] * p.kernel_size[1] * p.out_channels
            p.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(p, nn.BatchNorm2d):
            p.weight.data.normal_(1.0, 0.02)
            p.bias.data.fill_(0)
        elif isinstance(p, nn.ConvTranspose2d):
            n = p.kernel_size[1]
            factor = (n + 1) // 2
            if n % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:n, :n]
            weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            p.weight.data.copy_(torch.from_numpy(weights_np))


# fine-tune from VGG16
def fine_tune(net):
    vgg_16 = visionmodels.vgg16(pretrained=True)
    net.conv1.conv1_1.weight.data.copy_(vgg_16.features[0].weight.data)
    net.conv1.conv1_2.weight.data.copy_(vgg_16.features[2].weight.data)
    net.conv2.conv2_1.weight.data.copy_(vgg_16.features[5].weight.data)
    net.conv2.conv2_2.weight.data.copy_(vgg_16.features[7].weight.data)
    net.conv3.conv3_1.weight.data.copy_(vgg_16.features[10].weight.data)
    net.conv3.conv3_2.weight.data.copy_(vgg_16.features[12].weight.data)
    net.conv3.conv3_3.weight.data.copy_(vgg_16.features[14].weight.data)
    net.conv4.conv4_1.weight.data.copy_(vgg_16.features[17].weight.data)
    net.conv4.conv4_2.weight.data.copy_(vgg_16.features[19].weight.data)
    net.conv4.conv4_3.weight.data.copy_(vgg_16.features[21].weight.data)
    net.conv5.conv5_1.weight.data.copy_(vgg_16.features[24].weight.data)
    net.conv5.conv5_2.weight.data.copy_(vgg_16.features[26].weight.data)
    net.conv5.conv5_3.weight.data.copy_(vgg_16.features[28].weight.data)

    net.conv1.conv1_1.bias.data.copy_(vgg_16.features[0].bias.data)
    net.conv1.conv1_2.bias.data.copy_(vgg_16.features[2].bias.data)
    net.conv2.conv2_1.bias.data.copy_(vgg_16.features[5].bias.data)
    net.conv2.conv2_2.bias.data.copy_(vgg_16.features[7].bias.data)
    net.conv3.conv3_1.bias.data.copy_(vgg_16.features[10].bias.data)
    net.conv3.conv3_2.bias.data.copy_(vgg_16.features[12].bias.data)
    net.conv3.conv3_3.bias.data.copy_(vgg_16.features[14].bias.data)
    net.conv4.conv4_1.bias.data.copy_(vgg_16.features[17].bias.data)
    net.conv4.conv4_2.bias.data.copy_(vgg_16.features[19].bias.data)
    net.conv4.conv4_3.bias.data.copy_(vgg_16.features[21].bias.data)
    net.conv5.conv5_1.bias.data.copy_(vgg_16.features[24].bias.data)
    net.conv5.conv5_2.bias.data.copy_(vgg_16.features[26].bias.data)
    net.conv5.conv5_3.bias.data.copy_(vgg_16.features[28].bias.data)
    return net


cudnn.benchmark = True

if not os.path.isdir(opt.experiment):
    os.makedirs(opt.experiment)

# folder dataset
dataset = dset.ImageFolder(root=os.path.join(opt.dataroot, 'train'),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=opt.workers)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

netG = dss.network_dss(3, opt.dgf, opt.dgf_r, opt.dgf_eps, False)
criterion = nn.BCELoss()
netG.apply(weights_init)
netG = fine_tune(netG)
print('=>finetune from the VGG16... done !')
if opt.netG:  # load checkpoint if needed
    checkpoint = torch.load(opt.netG)
    model_dict = netG.state_dict()
    model_dict.update(checkpoint)
    netG.load_state_dict(model_dict)
print(netG)

real_B = torch.FloatTensor()
real_A = torch.FloatTensor()
real_x = torch.FloatTensor()
if opt.cuda:
    netG.cuda()

# setup optimizer
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
schedulerG = lrs.MultiStepLR(optimizerG, milestones=[5, 15], gamma=0.1)

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    schedulerG.step()

    for idx, (data, _) in enumerate(data_iter):
        w = data.size(3) // 2
        h = data.size(2)
        real_B.resize_(data.size(0), 1, h, w).copy_(data[:, :1, :, w:])
        real_A.resize_(3, h, w).copy_(data[0, :3, :, 0:w])
        real_A = normalize(real_A)
        real_A.unsqueeze_(0)

        real_x.resize_(1, 3, h, w).copy_(data[:1, :3, :, 0:w])
        input_G = Variable(real_A)
        if opt.cuda:
            real_x = real_x.cuda()
            real_B = real_B.cuda()
            input_G = input_G.cuda()

        fake_B, side1, side2, side3, side4, side5, side6 = netG(input_G, Variable(real_x))

        errG_bce = criterion(fake_B, Variable(real_B))

        optimizerG.zero_grad()
        errG_bce.backward()
        optimizerG.step()

        gen_iterations += 1

        print('[%d/%d][%d/%d]  Loss_bce %f'
              % (epoch, opt.niter, idx, len(dataloader), errG_bce.item()))
        if gen_iterations % 2000 == 0:
            torch.save(netG.state_dict(), '{0}/bfn1_netG_iter_{1}.pth'.format(opt.experiment, gen_iterations))
