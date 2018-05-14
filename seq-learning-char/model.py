# E. Culurciello, May 2018
# testing learning in LSTM, CNN, Attention neural networks: learning of sine-wave data
# refer to: https://towardsdatascience.com/memory-attention-sequences-37456d271992
# and: https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0
# inspired from: https://github.com/spro/practical-pytorch

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
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)



class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, pint, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pint = pint # how much older samples to integrate
        self.output_size = output_size

        # print('SIZES: ', self.input_size, self.hidden_size, self.pint, self.output_size)

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
        self.c1 = nn.Conv2d(1, self.hidden_size, [self.pint/4, self.hidden_size], stride=[self.pint/8, self.hidden_size])
        self.c2 = nn.Conv2d(self.hidden_size, self.hidden_size, [7,1], stride=1)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        # print('in', input)
        input = self.encoder(input)#.view(1, -1))
        # print('iin', input.shape)
        oc1 = self.c1(input.view(1, 1, input.size(0), input.size(1)))
        # print('out_c1', oc1.shape)
        oc2 = self.c2(oc1)
        # print('out_c2', oc2.shape)
        output = self.decoder(oc2.view(1, -1))
        # print('out', output)
        return output

    def init_hidden(self):
        return 0
