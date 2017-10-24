import json
from random import sample
import os
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

def collate_fn(args):
    images = []
    bboxes = []

    for img, bbox in args:
        images.append(img.unsqueeze(0))
        bboxes.append(bbox)
        
    images = torch.cat(images, 0)

    return images, bboxes

class COCO(data.Dataset):

    def __init__(self, root, annFile, image_size = 224):

        self.root = root
        self.data = {}
        self.categories = {}

        annotations = json.load(open(annFile))
        for img in annotations['images']:
            self.data[img['id']] = {
                'path' : img['file_name'],
                'bboxes' : []
            }
        for i, ann in enumerate(annotations['categories']):
            self.categories[ann['id']] = i

        for ann in annotations['annotations']:
            bbox = ann['bbox']
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            category = ann['category_id']
            if not category == 1:
                continue
            self.data[ann['image_id']]['bboxes'].append((bbox, category))

        self.idx = list(self.data.keys())
        self.image_size = image_size
        self.scale = transforms.Scale(image_size)

    def __getitem__(self, index):
        ann = self.data[self.idx[index]]

        img = Image.open(os.path.join(self.root, ann['path'])).convert('RGB')
        ow, oh = img.size
        img = self.scale(img)
        nw, nh = img.size
        sw, sh = nw/ow, nh/oh
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            normalize
        ])

        if nw > nh:
            minx = nw/2 - self.image_size/2
            miny = 0
            maxx = nw/2 + self.image_size/2
            maxy = nh
        else:
            minx = 0
            miny = nh/2 - self.image_size/2
            maxx = nw
            maxy = nh/2 + self.image_size/2

        img = transform(img)

        target = []
        for i, t in enumerate(list(ann['bboxes'])):
            new_target = [t[0][0]*sw - minx,
                          t[0][1]*sh - miny,
                          t[0][2]*sw - minx,
                          t[0][3]*sh - miny]

            if new_target[2] < minx: continue
            if new_target[3] < miny: continue

            if new_target[0] < minx: new_target[0] = minx
            if new_target[1] < miny: new_target[1] = miny
            if new_target[2] > maxx: new_target[2] = maxx
            if new_target[3] > maxy: new_target[3] = maxy

            if (new_target[2]-new_target[0])*(new_target[3]-new_target[1]) <= 0:
                continue

            target.append(([new_target[0]/self.image_size,
                            new_target[1]/self.image_size,
                            (new_target[2]-new_target[0])/self.image_size,
                            (new_target[3]-new_target[1])/self.image_size],
                           t[1]))

        return img, target

    def __len__(self):
        return len(self.idx)


