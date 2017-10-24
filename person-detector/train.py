import time
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.autograd import Variable as Var
from torch.utils.data import DataLoader

from models.detector import Detector
from models.loss import detection_loss
from data.coco import COCO, collate_fn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='root path of data')
    parser.add_argument('--batch_size', '-B', type=int, default=64,
                        metavar='b', help='batch size')
    parser.add_argument('--epoch', type=int, default=30,
                        help='number of epoch')
    parser.add_argument('--nworkers', type=int, default=1,
                        metavar='w', help='number of dataloader workers')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help = 'use gpu')
    parser.add_argument('--lr', '--learning-rate', default=1e-2,
                        type=float, help='initial learning rate')
    parser.add_argument('--log_interval', default=10, type=int,
                        help='Print log every N batches')
    return parser.parse_args()

args = get_args()
args.nclasses = 1
args.img_size = 224

if args.cuda:
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

def main():

    # data
    data_root = Path(args.data_root)
    train_dataset = COCO(
        annFile = str(data_root / 'annotations/instances_train2014.json'),
        root = str(data_root / 'train2014/'),
        image_size = args.img_size)
    val_dataset = COCO(
        annFile = str(data_root / 'annotations/instances_val2014.json'),
        root = str(data_root / 'val2014/'),
        image_size = args.img_size)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.nworkers,
        pin_memory = args.cuda,
        collate_fn = collate_fn
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.nworkers,
        pin_memory = args.cuda,
        collate_fn = collate_fn
    )

    # model
    model = Detector(args.nclasses + 1)
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr)

    if args.cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    prev_loss = np.inf
    for epoch in range(args.epoch):
        print('{:3d}/{:3d} epoch'.format(epoch+1, args.epoch))

        train(model, train_loader, optimizer)
        loss = validate(model, val_loader)

        if loss < prev_loss:
            torch.save(model, str('model.save'))
            prev_loss = loss

def train(model, data_loader, optimizer):

    model.train()

    batch_time = 0
    total_loss = {
        'prob' : 0,
        'bbox' : 0
    }

    for batch_no, (img, gt) in enumerate(data_loader):
        start_time = time.time()
        if args.cuda: img = img.cuda(async=True)

        prob_pred, bbox_pred = model(Var(img))

        prob_loss, bbox_loss = detection_loss(prob_pred, bbox_pred, gt)

        loss = prob_loss + bbox_loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time += time.time() - start_time
        total_loss['prob'] += prob_loss.data[0]
        total_loss['bbox'] += bbox_loss.data[0]

        if (batch_no+1) % args.log_interval == 0:
            avg_time = batch_time * 10e3 / args.log_interval
            total_loss['bbox'] /= args.log_interval
            total_loss['prob'] /= args.log_interval
            print('train | {:4d} /{:4d} batch | {:7.2f} ms/batch |'
                  ' prob_loss {:.3e} | bbox_loss {:.2e}'
                  .format(batch_no + 1, len(data_loader),
                          avg_time, total_loss['prob'], total_loss['bbox']))

            for k in total_loss: total_loss[k] = 0
            batch_time = 0

def validate(model, data_loader):

    model.eval()

    batch_time = 0
    total_loss = {
        'prob' : 0,
        'bbox' : 0
    }

    for batch_no, (img, gt) in enumerate(data_loader):
        start_time = time.time()

        prob_pred, bbox_pred = model(Var(img))

        prob_loss, bbox_loss = detection_loss(prob_pred, bbox_pred, gt)

        loss = prob_loss + bbox_loss

        batch_time += time.time() - start_time
        total_loss['prob'] += prob_loss.data[0]
        total_loss['bbox'] += bbox_loss.data[0]

    avg_time = batch_time * 10e3 / len(data_loader)
    total_loss['bbox'] /= len(data_loader)
    total_loss['prob'] /= len(data_loader)

    print('val | {:7.2f} ms/batch |'
          ' prob_loss {:.2e} | bbox_loss {:.2e}'.format(avg_time, total_loss['prob'],
                                                       total_loss['bbox']))

    return total_loss['bbox'] + total_loss['prob']

if __name__ == "__main__":
    main()
