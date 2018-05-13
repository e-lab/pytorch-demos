# modified from https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        print('in', input)
        input = self.encoder(input.view(1, -1))
        # print('iin', input)
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        # print('out', output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))



class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        print('SIZES: ', input_size, hidden_size, output_size)

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.c1 = nn.Conv2d(1, hidden_size, 9, stride=4)
        self.c2 = nn.Conv2d(hidden_size, hidden_size*2, 9, stride=4)
        self.decoder = nn.Linear(hidden_size*4, output_size)

    def forward(self, input):
        # print('in', input)]
        input = self.encoder(input.view(1, -1))
        print('iin', input.shape)
        output = self.c1(input.view(1, 1, 128, 64))
        print('out', output.shape)
        # output = self.decoder(output.view(1, -1))
        # print('out', output)
        return output

    def init_hidden(self):
        return 0
