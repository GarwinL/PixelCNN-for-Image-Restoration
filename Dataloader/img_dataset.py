import os
import PIL.Image as Image

import torch.utils.data as data
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import numpy as np
from Dataloader import data_utils


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ToGrayscale(object):
    def __call__(self, img):
        return img.convert('L', (0.2989, 0.5870, 0.1140, 0))

class PowerTransform(object):
    def __init__(self, gamma, gamma2=None):
        self.gamma = gamma
        self.gamma2 = gamma2

    def __call__(self, T):
        if self.gamma2 is not None:
            gamma = np.random.rand(1)[0] * (self.gamma2 - self.gamma) + self.gamma
        else:
            gamma = self.gamma
        return T**gamma

class DiscreteIntensityScale(object):
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, T):
        return T*self.factors[np.random.randint(len(self.factors))]


class ContinuousIntensityScale(object):
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, T):
        fac = np.random.rand(1)[0] * (self.factors[1] - self.factors[0]) + self.factors[0]
        return T*fac

class MaybeFlip(object):
    def __call__(self, img):
        if img.size[1] > img.size[0]:
            img.transpose(Image.TRANSPOSE)


def make_dataset(dir, filter=None, depth=-1):
    images = []
    if filter is None:
        filter = lambda x: True
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(data_utils.walklevel(dir, level=depth)):
        for fname in sorted(fnames):
            if is_image_file(fname) and filter(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    return images

class PlainImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/xxx.png
        root/xxy.png
        root/xxz.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None,
                 loader=default_loader, cache=False, filter=None, depth=-1):
        self.cache = cache
        self.img_cache = {}
        if isinstance(root, list):
            imgs = []
            for r in root:
                imgs.extend(make_dataset(r, filter=filter, depth=depth))
        else:
            imgs = make_dataset(root, filter=filter, depth=depth)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target)
        """
        path = self.imgs[index]
        if not index in self.img_cache:
            img = self.loader(path)
            if self.cache:
                self.img_cache[index] = img
        else:
            img = self.img_cache[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, -1

    def __len__(self):
        return len(self.imgs)