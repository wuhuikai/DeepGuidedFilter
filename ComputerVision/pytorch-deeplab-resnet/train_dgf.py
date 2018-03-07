import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from docopt import docopt
from torch.autograd import Variable

import deeplab_resnet

docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --LISTpath=<str>            Input image number list file [default: data/list/train_aug.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --snapshots=<str>           snapshots name [default: snapshots_dgf]
    --ft                        Finetune?
"""

args = docopt(docstr, version='v0.1')
print(args)

cudnn.enabled = True
gpu0 = int(args['--gpu0'])


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[0] * scale), int(img_temp.shape[1] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def get_data_from_chunk_v2(chunk):
    gt_path = args['--GTpath']
    img_path = args['--IMpath']

    scale = random.uniform(0.5, 1.3)
    dim = int(scale * 321)
    images = np.zeros((dim, dim, 3, len(chunk)))
    for i, piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg')).astype(float)
        ims = img_temp.copy() / 255.0
        ims = flip(ims, flip_p)

        img_temp = cv2.resize(img_temp, (321, 321)).astype(float)
        img_temp = scale_im(img_temp, scale)
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        img_temp = flip(img_temp, flip_p)
        images[:, :, :, i] = img_temp

        gt_temp = cv2.imread(os.path.join(gt_path, piece + '.png'))[:, :, 0]
        gt_temp[gt_temp == 255] = 0
        gt_temp = flip(gt_temp, flip_p)

    images = images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()

    ims = ims.transpose(2, 0, 1)
    ims = torch.from_numpy(ims[np.newaxis, :].copy()).float()
    labels = gt_temp[np.newaxis, :].copy()

    return images, labels, ims


def loss_calc(out, label, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax(dim=1)
    criterion = nn.NLLLoss2d()
    out = m(out)

    return criterion(out, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            for k in j.parameters():
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.Scale.layer5.parameters())
    b.append(model.guided_map_conv1.parameters())
    b.append(model.guided_map_conv2.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


if not os.path.exists('data/' + args['--snapshots']):
    os.makedirs('data/' + args['--snapshots'])

model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']), True, 4, 1e-2)

if args['--ft']:
    saved_state_dict = torch.load('data/snapshots/VOC12_scenes_12000.pth')
else:
    saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')

model.load_state_dict(saved_state_dict, strict=False)

max_iter = int(args['--maxIter'])
batch_size = 1
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.eval()  # use_global_stats = True

img_list = read_file(args['--LISTpath'])

data_list = []
# make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
for i in range(10):
    np.random.shuffle(img_list)
    data_list.extend(img_list)

model.cuda(gpu0)
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr},
                       {'params': get_10x_lr_params(model), 'lr': 10 * base_lr}], lr=base_lr, momentum=0.9,
                      weight_decay=weight_decay)

optimizer.zero_grad()
data_gen = chunker(data_list, batch_size)

for iter in range(max_iter + 1):
    chunk = data_gen.next()

    images, label, ims = get_data_from_chunk_v2(chunk)
    images = Variable(images).cuda(gpu0)
    ims = Variable(ims).cuda(gpu0)

    out = model(images, ims)
    loss = loss_calc(out, label, gpu0)
    iter_size = int(args['--iterSize'])
    loss = loss / iter_size
    loss.backward()

    if iter % 1 == 0:
        print 'iter = ', iter, 'of', max_iter, 'completed, loss = ', iter_size * (loss.data.cpu().numpy())

    if iter % iter_size == 0:
        optimizer.step()
        lr_ = lr_poly(base_lr, iter, max_iter, 0.9)
        print '(poly lr policy) learning rate', lr_
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_},
                               {'params': get_10x_lr_params(model), 'lr': 10 * lr_}], lr=lr_, momentum=0.9,
                              weight_decay=weight_decay)
        optimizer.zero_grad()

    if iter % 1000 == 0 and iter != 0:
        print 'taking snapshot ...'
        torch.save(model.state_dict(), 'data/' + args['--snapshots'] + '/VOC12_scenes_' + str(iter) + '.pth')
