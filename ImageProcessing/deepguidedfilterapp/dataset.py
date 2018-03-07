import os
import random

from PIL import Image

import torch.utils.data as data
from torchvision import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')

class Transforms(object):
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, imgs):
        return [self.transformer(img) for img in imgs]

class RandomTransforms(object):
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, imgs):
        if random.random() < 0.5:
            return imgs

        return [self.transformer(img) for img in imgs]

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        w, h = imgs[0].size
        if w == self.size and h == self.size:
            return imgs

        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)

        return [transforms.crop(img, y1, x1, self.size, self.size) for img in imgs]

class RandomRotate(object):
    def __call__(self, imgs):
        angle = random.randrange(4)
        if angle == 0:
            return imgs

        return [im.rotate(90*angle) for im in imgs]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs

class SuDataset(data.Dataset):
    def __init__(self, root, list_path, low_size=64, fine_size=-1, #flip=True, rotate=True,
                                    loader=default_loader):
        super(SuDataset, self).__init__()

        with open(list_path) as f:
            imgs = sorted([line.strip().split(',') for line in f])
        imgs = [[os.path.join(root, p) for p in group] for group in imgs]

        self.imgs = imgs
        self.loader = loader

        def append(imgs):
            imgs.append(transforms.resize(imgs[0], low_size, interpolation=Image.NEAREST))
            imgs.append(transforms.resize(imgs[1], low_size, interpolation=Image.NEAREST))
            return imgs

        self.transform = Compose([
                             RandomCrop(fine_size) if fine_size > 0 else None,
                             # Transforms(transforms.Scale(fine_size)) if fine_size > 0 else None,
                             # RandomTransforms(transforms.hflip) if flip else None,
                             # RandomTransforms(transforms.vflip) if flip else None,
                             # RandomRotate() if rotate else None,
                             transforms.Lambda(append),
                             Transforms(transforms.ToTensor())
                         ])

    def get_path(self, idx):
        return self.imgs[idx][0]

    def __getitem__(self, index):
        # input_img, gt_img
        imgs = [self.loader(path) for path in self.imgs[index]]

        # input_img, gt_img, low_res_input_img, low_res_gt_img
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs

    def __len__(self):
        return len(self.imgs)