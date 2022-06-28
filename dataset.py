import os

import random
import glob
import numpy as np
from PIL import Image


class Dataset(object):
    def __init__(self, LR_path, GT_path, transform=None):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.transform = transform

        self.LR_img = sorted(os.listdir(self.LR_path))
        self.GT_img = sorted(os.listdir(self.GT_path))

    def __getitem__(self, idx):
        img_item = {}
        lr = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[idx])).convert('RGB'))
        hr = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[idx])).convert('RGB'))

        img_item["GT"] = hr / 255.
        img_item["LR"] = lr / 255.

        if self.transform is not None:
            img_item = self.transform(img_item)

        img_item["GT"] = img_item["GT"].transpose(2, 0, 1).astype(np.float32)
        img_item["LR"] = img_item["LR"].transpose(2, 0, 1).astype(np.float32)
        
        return img_item

    def __len__(self):
        return len(self.LR_img)

class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']
        ih, iw = LR_img.shape[:2]

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)

        tx = ix * self.scale
        ty = iy * self.scale

        LR_patch = LR_img[iy: iy + self.patch_size, ix: ix + self.patch_size]
        GT_patch = GT_img[ty: ty + (self.scale * self.patch_size), tx: tx + (self.scale * self.patch_size)]

        return {'LR': LR_patch, 'GT': GT_patch}

class augmentation(object):

    def __call__(self, sample):
        LR_img, GT_img = sample['LR'], sample['GT']

        hor_flip = random.randrange(0, 2)
        ver_flip = random.randrange(0, 2)
        rot = random.randrange(0, 2)

        if hor_flip:
            temp_LR = np.fliplr(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.fliplr(GT_img)
            GT_img = temp_GT.copy()

            del temp_LR, temp_GT

        if ver_flip:
            temp_LR = np.flipud(LR_img)
            LR_img = temp_LR.copy()
            temp_GT = np.flipud(GT_img)
            GT_img = temp_GT.copy()

            del temp_LR, temp_GT

        if rot:
            LR_img = LR_img.transpose(1, 0, 2)
            GT_img = GT_img.transpose(1, 0, 2)

        return {'LR': LR_img, 'GT': GT_img}