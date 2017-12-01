import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.backends import cudnn
from torch.nn import functional as F
from torch.autograd import Variable as V
from torch.utils.data import DataLoader

#from models.openpose1 import OpenPose as Model
from models.linknet import FaceNet as Model
from data.ls3d import LS3D

from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-root', type=str, required=True)
    parser.add_argument('--val-root', type=str, required=True)
    parser.add_argument('--pre-trained', type=str, default='')
    parser.add_argument('--num-iter', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--nworkers', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log-interval', type=int, default=10)
    return parser.parse_args()

args = get_args()

if args.cuda:
    cudnn.benchmark = True
    #if torch.cuda.is_available():
    #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
def main():

    print(args)
    
    train_dataset = LS3D(args.train_root)
    val_dataset = LS3D(args.val_root)
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.nworkers,
        pin_memory = args.cuda
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.nworkers,
        pin_memory = args.cuda
    )

    model = Model()
    optimizer = optim.Adam(params = model.parameters(),
                           lr = args.lr)

    if args.pre_trained:
        state_dict = torch.load(args.pre_trained)
        weights = model.state_dict()
        for k in state_dict:
            if 'inner_model' in k and 'encoder' in k:
                weights[k.replace('module.inner_model.', '')] = state_dict[k]
        model.load_state_dict(weights)
    
    if args.cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    print(model)
    
    prev_loss = np.inf
    for epoch in range(args.epoch):
        print('{:3d}/{:3d} epoch'.format(epoch + 1, args.epoch))

        train(model, train_loader, optimizer)

        loss = validate(model, val_loader)

        if loss < prev_loss:
            print('Saving...')
            torch.save(model, str('model-%d.save'%epoch))
            prev_loss = loss
                    
def train(model, data_loader, optimizer):
    
    model.train()
            
    batch_time = 0
    total_loss = {
        'mse' : 0,
        'pos' : 0,
        'neg' : 0
    }

    for batch_no, (img, ann) in tqdm(enumerate(data_loader)):
        start_time = time.time()

        if args.cuda:
            img = img.cuda(async=True)
            ann = ann.cuda(async=True)
            
        preds = model(V(img))

        loss = 0
        mse_loss = 0
        pos_loss = 0
        neg_loss = 0
        for pred in preds:
            b, c, h, w = pred.size()
            _ann = F.adaptive_max_pool2d(ann, (h, w)).float()
            diff = ((_ann - pred) ** 2)
            mask = (_ann > 0).float()
            npos = int(mask.sum().data[0])
            pos_loss = (mask * diff).sum() / npos
            mask = 1 - mask
            neg_loss = (mask * diff).view(-1).topk(k = min(npos * 3, diff.numel()), sorted=False)[0]
            neg_loss = neg_loss.sum() / neg_loss.numel()
            mse = F.mse_loss(pred, _ann)
            #loss += pos_loss + neg_loss
            loss += mse

        total_loss['mse'] += mse.data[0]
        total_loss['pos'] += pos_loss.data[0]
        total_loss['neg'] += neg_loss.data[0]
            
        model.zero_grad()
        loss.backward()
        optimizer.step()
            
        batch_time += time.time() - start_time

        if (batch_no + 1) % args.log_interval == 0:
            avg_time = batch_time * 1e3 / args.log_interval
            for k in total_loss: total_loss[k] /= args.log_interval
            print('train | {:4d} /{:4d} batch | {:7.2f} ms/batch | mse loss {:.2e} | pos loss {:.2e} | neg loss {:.2e}'
                  .format(batch_no + 1, args.num_iter,
                          avg_time, total_loss['mse'], total_loss['pos'], total_loss['neg']))
            for k in total_loss: total_loss[k] = 0
            batch_time = 0

            if (batch_no + 1) >= args.num_iter: return

def validate(model, data_loader):
    
    model.eval()
            
    batch_time = 0
    total_loss = {
        'mse' : 0,
        'pos' : 0,
        'neg' : 0
    }

    for batch_no, (img, ann) in tqdm(enumerate(data_loader)):
        start_time = time.time()

        if args.cuda:
            img = img.cuda(async=True)
            ann = ann.cuda(async=True)
            
        preds = model(V(img, volatile=True))

        for pred in preds:
            b, c, h, w = pred.size()
            _ann = F.adaptive_max_pool2d(ann, (h, w)).float()
            diff = ((_ann - pred) ** 2)
            mask = (_ann > 0).float()
            npos = int(mask.sum().data[0])
            pos_loss = (mask * diff).sum() / npos
            mask = 1 - mask
            neg_loss = (mask * diff).view(-1).topk(k = min(npos * 3, diff.numel()), sorted=False)[0]
            neg_loss = neg_loss.sum() / neg_loss.numel()
            mse = F.mse_loss(pred, _ann)

        total_loss['mse'] += mse.data[0]
        total_loss['pos'] += pos_loss.data[0]
        total_loss['neg'] += neg_loss.data[0]
            
    avg_time = batch_time * 1e3 / len(data_loader)
    for k in total_loss: total_loss[k] /= len(data_loader)
    print('val | {:4d} batches | {:7.2f} ms/batch | mse loss {:.2e} | pos loss {:.2e} | neg loss {:.2e}'
          .format(len(data_loader),
                  avg_time, total_loss['mse'], total_loss['pos'], total_loss['neg']))
    return total_loss['mse']#total_loss['pos'] + total_loss['neg']
            
if __name__ == "__main__": main()
