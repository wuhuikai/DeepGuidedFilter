import os

import cv2
import numpy as np
import torch
from docopt import docopt
from skimage.io import imsave
from torch.autograd import Variable
from torch.nn import functional as F

from deepguidedfilter.guided_filter import FastGuidedFilter
from utils import decode_labels

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
    --exp_np=<str>              Experiment name [default: deeplab_res101_np]
    --exp=<str>                 Experiment name [default: deeplab_res101]
    --save                      Save ?
    --vis                       Save ?
    --dgf
    --dgf_r=<int>               [default: 4]
    --dgf_eps=<float>           [default: 1e-2]
"""

args = docopt(docstr, version='v0.1')
print args

max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print 'pred shape', pred.shape, 'gt shape', gt.shape
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou


gpu0 = int(args['--gpu0'])
im_path = args['--testIMpath']
gt_path = args['--testGTpath']
img_list = open('data/list/val.txt').readlines()
args['--dgf_r'] = int(args['--dgf_r'])
args['--dgf_eps'] = float(args['--dgf_eps'])

np_path = os.path.join('data', args['--exp_np'])

save = args['--save']
vis = args['--vis']
if save or vis:
    save_path = os.path.join('data', 'eval_' + args['--exp'])
    if vis and not os.path.isdir(save_path):
        os.makedirs(save_path)
    if save and not os.path.isdir(save_path + '_np'):
        os.makedirs(save_path + '_np')

if args['--dgf']:
    dgf = FastGuidedFilter(args['--dgf_r'], args['--dgf_eps'])
    dgf = dgf.cuda(gpu0)

hist = np.zeros((max_label + 1, max_label + 1))
for idx, i in enumerate(img_list):
    print '{}/{} ...'.format(idx + 1, len(img_list))

    gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), 0)

    output = np.load(os.path.join(np_path, i[:-1] + '.npz.npy'))

    if args['--dgf']:
        im = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg')).astype(float).transpose(2, 0, 1) / 255.0
        im = Variable(torch.from_numpy(im[np.newaxis, :]).float().cuda(gpu0), volatile=True)
        im = im.sum(1, keepdim=True)

        pre = Variable(torch.from_numpy(output.transpose(2, 0, 1)[np.newaxis, :]).cuda(gpu0), volatile=True)

        pre = F.softmax(pre, dim=1)
        output = dgf(im, pre).data.cpu().numpy().squeeze().transpose(1, 2, 0)

    if save:
        np.save(os.path.join(save_path + '_np', i[:-1] + '.npz'), output)

    output = np.argmax(output, axis=2)

    if vis:
        vis_output = decode_labels(output)
        imsave(os.path.join(save_path, i[:-1] + '.png'), vis_output)

    iou_pytorch = get_iou(output, gt)
    hist += fast_hist(gt.flatten(), output.flatten(), max_label + 1)
miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
print "Mean iou = ", np.sum(miou) / len(miou)
