#! /usr/local/bin/python3

# RNN example "abba" detector
#
# E. Culurciello, April 2017
#

import sys
import os
import time
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable

# create a fake dataset of symbols a,b:
data_size = 256
seq_len = 4 # abba sequence to be detected only!
data = np.random.randint(0, 2, data_size) # 0=1, 1=b, for example
label = np.zeros(data_size, dtype=int)
print('dataset is:', data, 'with length:', len(data))
for i in range(3, data_size-1):
   if (data[i-3]==0 and data[i-2]==1 and data[i-1]==1 and data[i]==0):
      label[i] += 1

print('labels is:', label, 'total number of example sequences:', np.sum(label))


# create model:
model = nn.RNN(1,1,1)
criterion = nn.L1Loss()

# test model:
# inp = Variable(torch.randn(seq_len).view(seq_len,1,1))
# h0 = Variable(torch.randn(seq_len).view(seq_len,1,1))
# print(inp, h0)
# output, hn = model(inp, h0)
# print('model test:', output,hn)


def train():
   model.train()
   hidden = Variable(torch.zeros(1,1,1))
   for i in tqdm(range(0, data_size-seq_len, seq_len)):
      X_batch = Variable(torch.from_numpy(data[i:i+seq_len]).view(seq_len,1,1).float())
      y_batch = Variable(torch.from_numpy(label[i:i+seq_len]).view(seq_len,1,1).float())
      model.zero_grad()
      output, hidden = model(X_batch, hidden)
      loss = criterion(output, y_batch)
      loss.backward(retain_variables=True)
      print('in/label/out:', data[i:i+seq_len], label[i:i+seq_len], output.data.view(1,4).numpy())
      # # print(X_batch, y_batch)
      if (data[i]==0 and data[i+1]==1 and data[i+2]==1 and data[i+3]==0):
         print('RIGHT')
      print(loss.data.numpy())


def test():
   model.eval()
   hidden = Variable(torch.zeros(1,1,1))
   for i in range(0, data_size-seq_len, seq_len):
      X_batch = Variable(torch.from_numpy(data[i:i+seq_len]).view(seq_len,1,1).float())
      y_batch = Variable(torch.from_numpy(label[i:i+seq_len]).view(seq_len,1,1).float())
      output, hidden = model(X_batch, hidden)
      print('in/label/out:', data[i:i+seq_len], label[i:i+seq_len], output.data.view(1,4).numpy())
      if (data[i]==0 and data[i+1]==1 and data[i+2]==1 and data[i+3]==0):
         print('RIGHT')

# train model:
train()
test()
