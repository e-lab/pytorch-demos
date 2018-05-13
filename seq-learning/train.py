# E. Culurciello, May 2018
# testing learning in LSTM, CNN, Attention neural networks: learning of sine-wave data
# refer to: https://towardsdatascience.com/memory-attention-sequences-37456d271992
# and: https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0
# inspired by https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from profile import *


parser = argparse.ArgumentParser(description='Learning sequences with LSTM, CNN, Attention')
parser.add_argument('--sequencer', type=str, default='LSTM', 
                    help='sequencer model to use: LSTM, CNN, Att')
parser.add_argument('--epochs', type=int, default=5, 
                    help='number of epochs to run')
parser.add_argument('--pprint', type=bool, default=True, 
                    help='Print PDF or display image only')
args = parser.parse_args()

# generate sine wave data: batch of N with length L points
def gen_data(T=20, L=1000, N=100):
    x = np.empty((N, L))
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T)
    return data

# profiling hooks:
def add_hooks(m):
    if len(list(m.children())) > 0: return
    m.register_buffer('total_ops', torch.zeros(1))
    m.register_buffer('total_params', torch.zeros(1))

    for p in m.parameters():
        m.total_params += torch.Tensor([p.numel()])

    if isinstance(m, nn.Conv2d):
        m.register_forward_hook(count_conv2d)
    if isinstance(m, nn.Conv1d):
        m.register_forward_hook(count_conv1d)
    elif isinstance(m, nn.BatchNorm2d):
        m.register_forward_hook(count_bn2d)
    elif isinstance(m, nn.ReLU):
        m.register_forward_hook(count_relu)
    elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        m.register_forward_hook(count_maxpool)
    elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        m.register_forward_hook(count_avgpool)
    elif isinstance(m, nn.Linear):
        m.register_forward_hook(count_linear)
    elif isinstance(m, nn.LSTMCell):
        m.register_forward_hook(count_lstmcell)
    elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        pass
    else:
        print("Not implemented for ", m)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.i1 = 51
        self.lstm1 = nn.LSTMCell(1, self.i1)
        self.lstm2 = nn.LSTMCell(self.i1, self.i1)
        self.linear = nn.Linear(self.i1, 1)
        print('LSTM model ops per sample:', 4*self.i1 + 4*self.i1^2 + self.i1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)
        h_t2 = torch.zeros(input.size(0), 51)
        c_t2 = torch.zeros(input.size(0), 51)

        for j, input_t in enumerate(input.chunk(input.size(1), dim=1)): # 999 ti
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output] # append to list

        for j in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2) # torch.Size([97, 1])
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs # torch.Size([97, 999])

# 1-layer CNN:
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.L = 4 # how many past samples needed as inputs
        self.doc1 = 8 # number of intermediate features
        self.c1 = nn.Conv1d(1, self.doc1, self.L)
        self.l1 = nn.Linear(self.doc1, 1)
        print('CNN model ops per sample:', self.L*self.doc1+self.doc1)

    def forward(self, input, future = 0):
        outputs = []
        for i in range (0, self.L):
            outputs += [torch.zeros(input.size(0), 1)] # fill with 0s

        for i in range(0, input.size(1)-self.L):
            input_b = input[:,i:i+self.L].unsqueeze(1) # torch.Size([97, 1, 10])
            o1 = self.c1(input_b)
            o1= o1.view(input.size(0), self.doc1)
            output = self.l1(o1)
            outputs += [output]

        for i in range(future):# if we should predict the future
            output_b = outputs[input.size(1)-self.L-1+i:input.size(1)-1+i]
            output_b = torch.stack(output_b, 1).squeeze().unsqueeze(1)
            o1 = self.c1(output_b)
            o1 = o1.view(input.size(0), self.doc1)
            output = self.l1(o1)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze()
        return outputs

# Attention network:
class Att(nn.Module):
    def __init__(self):
        super(Att, self).__init__()
        self.L = 8 # how many attention features to use
        self.a1 = Attention(self.L)
        self.l1 = nn.Linear(self.L, 1)
        print('Attention model ops per sample:', 2*self.L+3*self.L+self.L^2+self.L) # 2x mm, softmax, linear, linear l1

    def forward(self, input, future = 0):
        outputs = []
        for i in range (0, self.L):
            outputs += [torch.zeros(input.size(0), 1)] # fill with 0s
            o1 = torch.zeros(input.size(0), 1, self.L)

        for i in range(0, input.size(1)-self.L):
            context = input[:,i:i+self.L].unsqueeze(1)
            o1, attn = self.a1(o1, context)
            o1 = F.relu(o1)
            output = self.l1(o1)
            outputs += [output.squeeze(1)]

        for i in range(future):# if we should predict the future
            context = outputs[input.size(1)-self.L-1+i:input.size(1)-1+i]
            context = torch.stack(context, 1).squeeze().unsqueeze(1)
            o1, attn = self.a1(o1, context)
            o1 = F.relu(o1)
            output = self.l1(o1)
            outputs += [output.squeeze(1)]

        outputs = torch.stack(outputs, 1).squeeze()
        return outputs




# set random seed to 0
np.random.seed(0)
torch.manual_seed(0)

# load data and make training set
data = gen_data()
# train data:
input = torch.from_numpy(data[3:, :-1]).float()
target = torch.from_numpy(data[3:, 1:]).float()
#test data:
test_input = torch.from_numpy(data[:3, :-1]).float()
test_target = torch.from_numpy(data[:3, 1:]).float()

# print(input, target)
# plt.plot(np.arange(input.size(1)), input[1,:input.size(1)].numpy(), linewidth = 2.0)
# plt.show()

# build the model:
criterion = nn.MSELoss()
if args.sequencer == 'LSTM':
    model = LSTM()
    model.apply(add_hooks)
elif args.sequencer == 'CNN':
    model = CNN()
    model.apply(add_hooks)
elif args.sequencer == 'Att':
    from attention import Attention
    model = Att()
    model.apply(add_hooks)
else:
    print('No model specified!')
    os._exit()
    
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(model.parameters(), lr=0.8)

# begin to train
for i in range(args.epochs):
    print('STEP: ', i)
    def closure():
        optimizer.zero_grad()
        out = model(input)
        if args.sequencer == 'LSTM':
            loss = criterion(out, target)
        elif args.sequencer == 'CNN':
            loss = criterion(out[:,model.L:], target[:,model.L:])
        elif args.sequencer == 'Att':
            loss = criterion(out[:,model.L:], target[:,model.L:])
        print('Train loss:', loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)
    
    # begin to predict, no need to track gradient here
    with torch.no_grad():
        future = 1000
        pred = model(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('TEST loss:', loss.item())
        y = pred.detach().numpy()

    # if loss < 0.003:
        # break

print('Training finished!\n')

# draw the final result
plt.figure(figsize=(30,10))
plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(-2, 2)
def draw(yi, color):
    plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
    plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
draw(y[0], 'r')
draw(y[1], 'g')
draw(y[2], 'b')
if args.pprint:
    plt.savefig('pred_%s.pdf'%args.sequencer)
if not args.pprint:
    plt.show()
plt.close()

# profile networks and print:
total_ops = 0
total_params = 0
for m in model.modules():
    if len(list(m.children())) > 0: continue
    total_ops += m.total_ops
    total_params += m.total_params
total_ops = total_ops
total_params = total_params

print('Profiler results:')
print("#Ops: %f GOps"%(total_ops/1e9))
print("#Parameters: %f M"%(total_params/1e6))
