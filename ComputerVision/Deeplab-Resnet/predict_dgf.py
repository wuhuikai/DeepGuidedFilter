import os

import cv2
import numpy as np

import torch

from docopt import docopt
from skimage.io import imsave

from torch.autograd import Variable

import deeplab_resnet

from utils import decode_labels

docstr = """Predict with ResNet-DeepLab

Usage: 
    predict_dgf.py [options]

Options:
    -h, --help                  Print this message
    --img_path=<str>            Sketch images path prefix [default: None]
    --gpu0=<int>                GPU number [default: 0]
    --snapshots=<str>           snapshots name [default: None] 
"""

def main():
    args = docopt(docstr, version='v0.1')
    print(args)

    gpu0 = int(args['--gpu0'])

    model = deeplab_resnet.Res_Deeplab(21, True, 4, 1e-2)
    model.load_state_dict(torch.load(args['--snapshots']))
    model.eval().cuda(gpu0)

    im_path = args['--img_path']

    img = cv2.imread(im_path).astype(float)
    img_original = img.copy() / 255.0
    img[:, :, 0] = img[:, :, 0] - 104.008
    img[:, :, 1] = img[:, :, 1] - 116.669
    img[:, :, 2] = img[:, :, 2] - 122.675

    output = model(*[Variable(torch.from_numpy(i[np.newaxis, :].transpose(0, 3, 1, 2)).float(),
                              volatile=True).cuda(gpu0) for i in  [img, img_original]])
    output = output.cpu().data[0].numpy().transpose(1, 2, 0)
    output = np.argmax(output, axis=2)

    vis_output = decode_labels(output)

    output_directory = os.path.dirname(im_path)
    output_name = os.path.splitext(os.path.basename(im_path))[0]
    save_path = os.path.join(output_directory, '{}_labels.png'.format(output_name))
    imsave(save_path, vis_output)

if __name__ == '__main__':
    main()