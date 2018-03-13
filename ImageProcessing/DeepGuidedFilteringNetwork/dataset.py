import torch.utils.data as data

from PIL import Image

from torchvision import transforms

def default_loader(path):
    return Image.open(path).convert('RGB')

class Transforms(object):
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, imgs):
        return [self.transformer(img) for img in imgs]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs

class PreSuDataset(data.Dataset):
    def __init__(self, img_list, low_size=64, loader=default_loader):
        super(PreSuDataset, self).__init__()

        self.imgs = list(img_list)
        self.loader = loader

        def append(imgs):
            imgs.append(transforms.Scale(low_size, interpolation=Image.NEAREST)(imgs[0]))
            return imgs

        self.transform = Compose([
                             transforms.Lambda(append),
                             Transforms(transforms.ToTensor())
                         ])

    def get_path(self, idx):
        return self.imgs[idx]

    def __getitem__(self, index):
        # input_img
        imgs = [self.loader(self.imgs[index])]

        # input_img, low_res_input_img
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs

    def __len__(self):
        return len(self.imgs)