import os

import cv2
import numpy as np
import torch
from docopt import docopt
from skimage.io import imsave
from torch.autograd import Variable

import deeplab_resnet
from utils import decode_labels

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage: 
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
    --exp=<str>                 Experiment name [default: deeplab_res101_dgf]
    --snapshots=<str>           snapshots name [default: snapshots_dgf]
    --iter=<int>                Iter (dgf: 18    dgf_ft: 18) [default: 18] 
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
model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']), True, 4, 1e-2)
model.eval()
model.cuda(gpu0)
snapPrefix = args['--snapPrefix']
gt_path = args['--testGTpath']
img_list = open('data/list/val.txt').readlines()
iter = int(args['--iter'])
saved_state_dict = torch.load(os.path.join('data/' + args['--snapshots'], snapPrefix + str(iter) + '000.pth'))
model.load_state_dict(saved_state_dict)

save_path = os.path.join('data', args['--exp'])
if not os.path.isdir(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path + '_np'):
    os.makedirs(save_path + '_np')

hist = np.zeros((max_label + 1, max_label + 1))
for idx, i in enumerate(img_list):
    print '{}/{} ...'.format(idx + 1, len(img_list))

    img_temp = cv2.imread(os.path.join(im_path, i[:-1] + '.jpg')).astype(float)
    img_original = img_temp.copy() / 255.0
    img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
    img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
    img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
    img = img_temp
    gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), 0)

    output = model(
        Variable(torch.from_numpy(img[np.newaxis, :].transpose(0, 3, 1, 2)).float(), volatile=True).cuda(gpu0),
        Variable(torch.from_numpy(img_original[np.newaxis, :].transpose(0, 3, 1, 2)).float(), volatile=True).cuda(gpu0))
    output = output.cpu().data[0].numpy()

    output = output.transpose(1, 2, 0)

    np.save(os.path.join(save_path + '_np', i[:-1] + '.npz'), output)

    output = np.argmax(output, axis=2)

    vis_output = decode_labels(output)
    imsave(os.path.join(save_path, i[:-1] + '.png'), vis_output)

    iou_pytorch = get_iou(output, gt)
    hist += fast_hist(gt.flatten(), output.flatten(), max_label + 1)
miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
print 'pytorch', iter, "Mean iou = ", np.sum(miou) / len(miou)
