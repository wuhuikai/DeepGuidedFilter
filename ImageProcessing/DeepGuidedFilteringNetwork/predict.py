import os
import argparse

import torch
import numpy as np

from tqdm import tqdm
from skimage.io import imsave
from torch.autograd import Variable

from dataset import PreSuDataset
from module import DeepGuidedFilter, DeepGuidedFilterAdvanced

def tensor_to_img(tensor, transpose=False):
    im = np.asarray(np.clip(np.squeeze(tensor.numpy()) * 255, 0, 255), dtype=np.uint8)
    if transpose:
        im = im.transpose((1, 2, 0))

    return im

parser = argparse.ArgumentParser(description='Predict with Deep Guided Filtering Networks')
parser.add_argument('--img_path',    type=str, default=None,                       help='IMG_PATH')
parser.add_argument('--img_list',    type=str, default=None,                       help='IMG_LIST')
parser.add_argument('--model_path',  type=str, required=True,                      help='MODEL_PATH')
parser.add_argument('--save_folder', type=str, required=True,                      help='SAVE_FOLDER')
parser.add_argument('--model',       type=str, default='deep_guided_filter',       help='model')

parser.add_argument('--low_size',    type=int, default=64,                         help='LOW_SIZE')
parser.add_argument('--gpu',         type=int, default=0,                          help='GPU')
parser.add_argument('--gray',                  default=False, action='store_true', help='GPU')
args = parser.parse_args()

# Test Images
img_list = []
if args.img_path is not None:
    img_list.append(args.img_path)
if args.img_list is not None:
    with open(args.img_list) as f:
        for line in f:
            img_list.append(line.strip())
assert len(img_list) > 0

# Save Folder
if not os.path.isdir(args.save_folder):
    os.makedirs(args.save_folder)

# Model
if args.model in ['guided_filter', 'deep_guided_filter']:
    model = DeepGuidedFilter()
elif args.model == 'deep_guided_filter_advanced':
    model = DeepGuidedFilterAdvanced()
else:
    print('Not a valid model!')
    exit(-1)

if args.model in ['deep_guided_filter', 'deep_guided_filter_advanced']:
    model.load_state_dict(torch.load(args.model_path))
elif args.model == 'guided_filter':
    model.init_lr(args.model_path)
else:
    print('Not a valid model!')
    exit(-1)

# data set
test_data = PreSuDataset(img_list, low_size=args.low_size)

# GPU
if args.gpu >= 0:
    with torch.cuda.device(args.gpu):
        model.cuda()

# test
i_bar = tqdm(total=len(test_data), desc='#Images')
for idx, imgs in enumerate(test_data):
    name = os.path.basename(test_data.get_path(idx))

    lr_x, hr_x = imgs[1].unsqueeze(0), imgs[0].unsqueeze(0)
    if args.gpu >= 0:
        with torch.cuda.device(args.gpu):
            lr_x = lr_x.cuda()
            hr_x = hr_x.cuda()
    imgs = model(Variable(lr_x), Variable(hr_x)).data.cpu()

    for img in imgs:
        img = tensor_to_img(img, transpose=True)
        if args.gray:
            img = img.mean(axis=2).astype(img.dtype)
        imsave(os.path.join(args.save_folder, name), img)

    i_bar.update()