import os
import random

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from docopt import docopt
from torch.autograd import Variable

import deeplab_resnet

docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train_dgf.py [options]

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
    --snapshots=<str>           snapshots name [default: snapshots]
    --dgf                       WITH Guided Filtering Layer ?
    --ft                        Finetune?
    --ft_model_path=<str>       Model path for finetune [default: None]
"""


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j + 1) // 2
    j = int(np.ceil((j + 1) / 2.0))
    j = (j + 1) // 2
    return j


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[0] * scale), int(img_temp.shape[1] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def get_data_from_chunk_v2(args, chunk):
    assert len(chunk) == 1

    gt_path = args['--GTpath']
    img_path = args['--IMpath']

    scale = random.uniform(0.5, 1.3)
    flip_p = random.uniform(0, 1)

    images = cv2.imread(os.path.join(img_path, chunk[0] + '.jpg')).astype(float)
    if args['--dgf']:
        ims = images.copy() / 255.0
        ims = flip(ims, flip_p)
        ims = ims.transpose((2, 0, 1))
        ims = torch.from_numpy(ims[np.newaxis, :].copy()).float()

    images = cv2.resize(images, (321, 321)).astype(float)
    images = scale_im(images, scale)
    images[:, :, 0] = images[:, :, 0] - 104.008
    images[:, :, 1] = images[:, :, 1] - 116.669
    images[:, :, 2] = images[:, :, 2] - 122.675
    images = flip(images, flip_p)
    images = images[:, :, :, np.newaxis]
    images = images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images.copy()).float()

    gt = cv2.imread(os.path.join(gt_path, chunk[0] + '.png'))[:, :, 0]
    gt[gt == 255] = 0
    gt = flip(gt, flip_p)
    if not args['--dgf']:
        dim = outS(321 * scale)  # 41
        gt = cv2.resize(gt, (dim, dim), interpolation=cv2.INTER_NEAREST).astype(float)
    labels = gt[np.newaxis, :].copy()

    if args['--dgf']:
        return (images, ims), labels
    else:
        return (images,), labels


def loss_calc(out, label, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    out = nn.LogSoftmax()(out)

    return nn.NLLLoss2d()(out, label)


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


def get_10x_lr_params(args, model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.Scale.layer5.parameters())
    if args['--dgf']:
        b.append(model.guided_map_conv1.parameters())
        b.append(model.guided_map_conv2.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def main():
    args = docopt(docstr, version='v0.1')
    print(args)

    cudnn.enabled = True
    gpu0 = int(args['--gpu0'])
    base_lr = float(args['--lr'])
    max_iter = int(args['--maxIter'])
    iter_size = int(args['--iterSize'])
    weight_decay = float(args['--wtDecay'])

    if not os.path.exists('data/' + args['--snapshots']):
        os.makedirs('data/' + args['--snapshots'])

    model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']), args['--dgf'], 4, 1e-2)

    if args['--ft']:
        saved_state_dict = torch.load(args['--ft_model_path'])
    else:
        saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
    model_dict = model.state_dict()
    model_dict.update(saved_state_dict)
    model.load_state_dict(model_dict)

    model.float().eval().cuda(gpu0)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr},
                           {'params': get_10x_lr_params(args, model), 'lr': 10 * base_lr}], lr=base_lr, momentum=0.9,
                          weight_decay=weight_decay)
    optimizer.zero_grad()

    img_list = read_file(args['--LISTpath'])
    data_list = []
    # make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
    for i in range(10):
        np.random.shuffle(img_list)
        data_list.extend(img_list)
    data_gen = chunker(data_list, 1)

    for iter in range(max_iter + 1):
        inputs, label = get_data_from_chunk_v2(args, next(data_gen))
        inputs = [Variable(input).cuda(gpu0) for input in inputs]

        loss = loss_calc(model(*inputs), label, gpu0) / iter_size
        loss.backward()

        if iter % 1 == 0:
            print('iter = ', iter, 'of', max_iter, 'completed, loss = ', iter_size * (loss.data.cpu().numpy()))

        if iter % iter_size == 0:
            optimizer.step()
            lr_ = lr_poly(base_lr, iter, max_iter, 0.9)
            print('(poly lr policy) learning rate', lr_)
            optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_},
                                   {'params': get_10x_lr_params(args, model), 'lr': 10 * lr_}],
                                  lr=lr_, momentum=0.9, weight_decay=weight_decay)
            optimizer.zero_grad()

        if iter % 1000 == 0 and iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), 'data/' + args['--snapshots'] + '/VOC12_scenes_' + str(iter) + '.pth')


if __name__ == '__main__':
    main()
