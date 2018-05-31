# E. Culurciello, May 2018
# testing learning in LSTM, CNN, Attention neural networks: learning of sine-wave data
# refer to: https://towardsdatascience.com/memory-attention-sequences-37456d271992
# and: https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0
# and https://openreview.net/forum?id=rk8wKk-R-
# inspired from: https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import Attention


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
        print('SIZES: ', input_size, hidden_size, pint, output_size)

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.c1 = nn.Conv2d(1, hidden_size, [pint, hidden_size])
        self.relu = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, position):
        # print('in', input, position)
        enc = self.encoder(input)
        # print('enc', enc)
        inn = enc + position
        # print('enc+pos', inn, position)
        # print('inn', inn.shape) 
        oc1 = self.relu( self.c1(inn.view(1, 1, inn.size(0), inn.size(1))) )
        oc2 = self.relu( self.l1( oc1.view(1, -1) ) )
        output = self.decoder(oc2.view(1, -1))
        # print('out', output)
        return output
