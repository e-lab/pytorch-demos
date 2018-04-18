# modified from: https://github.com/spro/practical-pytorch

import torch
import numpy
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
# argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=1000)
argparser.add_argument('--print_every', type=int, default=25)
argparser.add_argument('--hidden_size', type=int, default=50)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--nseq', type=int, default=128)
args = argparser.parse_args()

import os
filename='input.txt'
if not os.path.isfile(filename):
    os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
file, file_len = read_file('input.txt')

def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

model = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

in_array = numpy.zeros(args.nseq) # where we save inputs for conv(CNN)/attention models
def shift(l, n):
    return l[n:] + l[:n]

def train(inp, target):
    hidden = model.init_hidden()
    model.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        in_array[0] = inp[c]
        # print('be4', in_array) 
        # shift(in_array, 1)
        # print('after', in_array)
        # break 
        output, hidden = model(inp[c], hidden)
        # print( 'batch:', inp[c].data.numpy(), numpy.argmax(output.data.numpy()), target[c].data.numpy())
        loss += criterion(output, target[c])

    loss.backward()
    model_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in range(1, args.n_epochs + 1):
        loss = train(*random_training_set(args.chunk_len))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[elapsed time: %s, epoch: %d,  percent complete: %d%%, loss: %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            # print(generate(model, 'Wh', 100), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

