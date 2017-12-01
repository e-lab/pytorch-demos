import numpy as np
from random import random
from glob import glob
from pathlib import Path
from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa

import torch
from torch.utils.data import Dataset
from torch.utils.serialization import load_lua
from torchvision import transforms

def ann2hm(ann, size):
    h, w = size
    sigma = 5e-3 * max(h, w)
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = torch.Tensor(np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2))))

    hm = torch.zeros((68, h, w))
    for i in range(68):
        x, y = ann[i]
        ul = [int(x - 3 * sigma), int(y - 3 * sigma)]
        br = [int(x + 3 * sigma + 1), int(y + 3 * sigma + 1)]
        if (ul[0] >= w or ul[1] >= h or
            br[0] < 0 or br[1] < 0):
            continue

        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)
        if (img_x[1] - img_x[0] <= 0 or img_y[1] - img_y[0] <= 0 or
            g_x[1] - g_x[0] <= 0 or g_y[1] - g_y[0] <= 0):
            continue
        hm[i,img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
    return hm

class LS3D(Dataset):

    def __init__(self, root, start = 0, end = 1, image_size = 256):

        root = Path(root)
        imgs = glob((root / '**' / '*.jpg').as_posix(),
                    recursive = True)
        anns = glob((root / '**' / '*.t7').as_posix(),
                    recursive = True)
        
        imgs = sorted(imgs)
        anns = sorted(anns)

        start = int(start*len(imgs))
        end = int(end*len(imgs))

        self.imgs = imgs[start:end]
        self.anns = anns[start:end]
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.scale = transforms.Scale(image_size)
        self.tensor = transforms.ToTensor()

        self.aug = iaa.SomeOf(3, [
            iaa.CropAndPad(percent=(-0.25, 0.25)),
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.Dropout(p=(0, 0.2)),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            iaa.Sometimes(0.50, iaa.Affine(scale=(0.5, 1.5))),
            iaa.Sometimes(0.50, iaa.Affine(rotate=(-60, 60))),
            iaa.Sometimes(0.50, iaa.Affine(shear=(-10, 10)))
        ])

    def __getitem__(self, index):

        img = self.imgs[index]
        ann = self.anns[index]

        img = Image.open(img).convert('RGB')
        ann = load_lua(ann)
        
        ow, oh = img.size
        img = self.scale(img)
        nw, nh = img.size
        sx, sy = nw/ow, nh/oh

        # scale annotation to image scale
        ann[:, 0] = ann[:, 0] * sx
        ann[:, 1] = ann[:, 1] * sy
        kpts = []
        for x, y in ann:
            kpts.append(ia.Keypoint(x=int(x), y=int(y)))
        
        # apply imgaug transforms
        img = np.asarray(img)
        kpts = ia.KeypointsOnImage(kpts, shape=img.shape)
        aug = self.aug.to_deterministic()
        img = aug.augment_images([img])[0]
        kpts = aug.augment_keypoints([kpts])[0]

        for i in range(len(kpts.keypoints)):
            kp = kpts.keypoints[i]
            ann[i,0] = int(kp.x); ann[i,1] = int(kp.y)

        img = Image.fromarray(img)
        
        # Random crop around annotation
        min_x, min_y = list(ann.min(0)[0].float())
        max_x, max_y = list(ann.max(0)[0].float())

        off_x = (self.image_size - (max_x - min_x)) * random()
        off_y = (self.image_size - (max_y - min_y)) * random()

        x1 = int(min_x - off_x)
        y1 = int(min_y - off_y)
        x2 = x1 + self.image_size
        y2 = y1 + self.image_size

        pad_x1 = -x1 if x1 < 0 else 0
        pad_y1 = -y1 if y1 < 0 else 0
        pad_x2 = x2-nw if x2 > nw else 0
        pad_y2 = y2-nh if y2 > nh else 0

        pad = transforms.Pad((pad_x1, pad_y1, pad_x2, pad_y2), 0)
        img = pad(img)
        
        img = self.tensor(img)

        ann[:, 0] = ann[:, 0] + pad_x1
        ann[:, 1] = ann[:, 1] + pad_y1

        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = x1 + self.image_size
        y2 = y1 + self.image_size

        ann[:, 0] = ann[:, 0] - x1
        ann[:, 1] = ann[:, 1] - y1

        img = img[:, y1:y2, x1:x2]
        
        #img = torch.from_numpy(img).permute(2,0,1).contiguous()
        #img = self.normalize(img)
        
        return img, ann2hm(ann, (self.image_size, self.image_size))

    def __len__(self):
        return min(len(self.imgs), len(self.anns))
