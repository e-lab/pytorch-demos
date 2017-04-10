#! /usr/local/bin/python3

# RNN example "abba" detector
# 
# another simple example:
# https://github.com/pytorch/examples/tree/master/time_sequence_prediction
# a more complex example:
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
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
import torch.optim as optim

np.set_printoptions(precision=2)
print('Simple RNN model to detect a abba/0110 sequence')

# create a fake dataset of symbols a,b:
num_symbols = 2 # a,b
data_size = 1024*8
seq_len = 4 # abba sequence to be detected only!
num_layers = 4
rdata = np.random.randint(0, num_symbols, data_size) # 0=1, 1=b, for example

# turn it into 1-hot encoding:
data = np.empty([data_size, num_symbols])
for i in range(0, data_size):
   data[i,:] = ( rdata[i], not rdata[i] ) # only works for 2 symbols for now!

print('dataset is:', data, 'with size:', data.shape)

# create labels:
label = np.zeros([data_size, num_symbols])
count = 0
for i in range(seq_len, data_size):
   label[i,:] = (1,0)
   if (rdata[i-3]==0 and rdata[i-2]==1 and rdata[i-1]==1 and rdata[i]==0):
      label[i,:] = (0,1) 
      count += 1

print('labels is:', label, 'total number of example sequences:', count)


# create model:
model = nn.RNN(num_symbols, num_symbols, num_layers) # see: http://pytorch.org/docs/nn.html#rnn
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# test model, see: http://pytorch.org/docs/nn.html#rnn
# inp = torch.zeros(seq_len, 1, num_symbols)
# inp[0,0,0]=1
# inp[1,0,1]=1
# inp[2,0,1]=1
# inp[3,0,0]=1
# h0 = torch.zeros(1, 1, num_symbols)
# print(inp, h0)
# output, hn = model( Variable(inp), Variable(h0))
# print('model test:', output,hn)


def repackage_hidden(h):
   """Wraps hidden states in new Variables, to detach them from their history."""
   if type(h) == Variable:
      return Variable(h.data)
   else:
      return tuple(repackage_hidden(v) for v in h)


num_epochs = 4


def train():
   model.train()
   hidden = Variable(torch.zeros(num_layers, 1, num_symbols))
   
   for epoch in range(num_epochs):  # loop over the dataset multiple times
      running_loss = 0.0
      for i in range(0, data_size-seq_len, seq_len):
         # get inputs:
         inputs = torch.from_numpy( data[i:i+seq_len,:]).view(seq_len, 1, num_symbols).float()
         labels = torch.from_numpy(label[i:i+seq_len,:]).view(seq_len, 1, num_symbols).float()
         
         # wrap them in Variable
         inputs, labels = Variable(inputs), Variable(labels)

         # Starting each batch, we detach the hidden state from how it was previously produced.
         # If we didn't, the model would try backpropagating all the way to start of the dataset.
         hidden = repackage_hidden(hidden)

         # forward, backward, optimize
         model.zero_grad()
         output, hidden = model(inputs, hidden)
         
         loss = criterion(output, labels)
         loss.backward()
         optimizer.step()

         # print info / statistics:
         # print('in:', data[i:i+seq_len,0], 'label:', label[i:i+seq_len,1], 'out:', output.data.numpy())
         # print(inputs, labels)
         # input()

         running_loss += loss.data[0]
         num_ave = 64
         if i % num_ave == 0:   # print every ave mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / num_ave))
            running_loss = 0.0
   
   print('Finished Training')


def test():
   model.eval()
   hidden = Variable(torch.zeros(num_layers, 1, num_symbols))
   for i in range(0, 32):
      inputs = torch.from_numpy( data[i:i+seq_len,:]).view(seq_len, 1, num_symbols).float()
      labels = torch.from_numpy(label[i:i+seq_len,:]).view(seq_len, 1, num_symbols).float()
      inputs = Variable(inputs)
      hidden = repackage_hidden(hidden)     
      output, hidden = model(inputs, hidden)
      print('in:', data[i:i+seq_len,0], 'label:', label[i:i+seq_len,1], 'out:', output.data.numpy())


# train model:
print('\nTRAINING ---')
train()
print('\n\nTESTING ---')
test()
