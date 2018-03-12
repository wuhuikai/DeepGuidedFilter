import os

import cv2
import numpy as np

import torch
import torch.nn as nn

from docopt import docopt
from skimage.io import imsave

from torch.autograd import Variable

import deeplab_resnet

from utils import decode_labels

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt_dgf.py [options]

Options:
    -h, --help                  Print this message
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
    --exp=<str>                 Experiment name [default: deeplab_res101]
    --snapshots=<str>           snapshots name [default: None]
    --dgf                       WITH Guided Filtering Layer ? 
"""


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def main():
    args = docopt(docstr, version='v0.1')
    print(args)

    gpu0 = int(args['--gpu0'])
    im_path = args['--testIMpath']
    gt_path = args['--testGTpath']

    model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']), args['--dgf'], 4, 1e-2)
    model.eval().cuda(gpu0)

    img_list = open('data/list/val.txt').readlines()
    saved_state_dict = torch.load(args['--snapshots'])
    model.load_state_dict(saved_state_dict)

    save_path = os.path.join('data', args['--exp'])
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)
    hist = np.zeros((max_label + 1, max_label + 1))
    for idx, i in enumerate(img_list):
        print('{}/{} ...'.format(idx + 1, len(img_list)))

        img = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg')).astype(float)
        img_original = img.copy() / 255.0
        img[:, :, 0] = img[:, :, 0] - 104.008
        img[:, :, 1] = img[:, :, 1] - 116.669
        img[:, :, 2] = img[:, :, 2] - 122.675

        if args['--dgf']:
            inputs = [img, img_original]
        else:
            inputs = [np.zeros((513, 513, 3))]
            inputs[0][:img.shape[0], :img.shape[1], :] = img

        output = model(*[Variable(torch.from_numpy(i[np.newaxis, :].transpose(0, 3, 1, 2)).float(),
                                  volatile=True).cuda(gpu0) for i in inputs])
        if not args['--dgf']:
            interp = nn.Upsample(size=(513, 513), mode='bilinear')
            output = interp(output)
            output = output[:, :, :img.shape[0], :img.shape[1]]

        output = output.cpu().data[0].numpy().transpose(1, 2, 0)
        output = np.argmax(output, axis=2)

        vis_output = decode_labels(output)
        imsave(os.path.join(save_path, i[:-1] + '.png'), vis_output)

        gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), 0)
        hist += fast_hist(gt.flatten(), output.flatten(), max_label + 1)

    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("Mean iou = ", np.sum(miou) / len(miou))


if __name__ == '__main__':
    main()