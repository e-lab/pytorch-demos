# E. Culurciello, May 2018
# testing learning in LSTM, CNN, Attention neural networks: learning of sine-wave data
# refer to: https://towardsdatascience.com/memory-attention-sequences-37456d271992
# and: https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0
# and https://openreview.net/forum?id=rk8wKk-R-
# inspired from: https://github.com/spro/practical-pytorch

import torch
import numpy
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from helpers import *
from model import *
from generate import *
from profile import *

# Parse command line arguments
parser = argparse.ArgumentParser(description='Learning char sequences with LSTM, CNN, Attention')
parser.add_argument('--sequencer', type=str, default='GRU', 
                    help='sequencer model to use: GRU, CNN, Att')
parser.add_argument('--epochs', type=int, default=1000, 
                    help='number of epochs to run')
# parser.add_argument('--pprint', type=bool, default=True, 
                    # help='Print PDF or display image only')
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=2, help='GRU layers')
parser.add_argument('--pint', type=int, default=16, help='CNN/Att past samples to integrate')
parser.add_argument('--print_every', type=int, default=25)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--chunk_len', type=int, default=200)
args = parser.parse_args()

import os

# load or download tinyshakespeare data:
filename = 'tinyshakespeare.txt'
if not os.path.isfile(filename):
    os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    os.system('mv input.txt tinyshakespeare.txt')
file, file_len = read_file('tinyshakespeare.txt')


def random_training_set(chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


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


# define model:
if args.sequencer == 'GRU':
    model = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
    model.apply(add_hooks)
    hidden = model.init_hidden()
elif args.sequencer == 'CNN':
    model = CNN(n_characters, args.hidden_size, args.pint, n_characters)
    model.apply(add_hooks)
elif args.sequencer == 'Att':
    print('Not implemented yet!')
    os._exit()
    # model = Att(n_characters, args.hidden_size, args.pint, n_characters)
    # model.apply(add_hooks)
else:
    print('No model specified!')
    os._exit()

model_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0


def train(inp, target):
    if args.sequencer == 'GRU':
        hidden = model.init_hidden()
    
    model.zero_grad()
    loss = 0

    if args.sequencer == 'GRU':
        iters = args.chunk_len
        for c in range(iters):        
            # print('I,T', inp[c],target[c])
            output, hidden = model(inp[c], hidden)
            loss += criterion(output, target[c].view(1))

    elif args.sequencer == 'CNN' or 'Att':
        iters = args.chunk_len-args.pint
        for c in range(iters):
            # print('2-I,T', c, inp[c:c+args.pint], inp[c:c+args.pint].shape, target[c+args.pint-1])
            # predicted = ''
            # for i in range(inp[c:c+args.pint].shape[0]): 
                # predicted += all_characters[inp[c+i]]

            # print('prime predicted string:', predicted)
            # print('next char', all_characters[target[c+args.pint-1]], '\n')

            position = c/args.chunk_len-0.5 # positional vector to add to input
            output = model(inp[c:c+args.pint], position)
            loss += criterion(output, target[c+args.pint-1].view(1))

    loss.backward()
    model_optimizer.step()

    return loss.item() / iters

def save():
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '_' + args.sequencer + '.pt'
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)

try:
    print("Training for %d epochs..." % args.epochs)
    for epoch in range(1, args.epochs + 1):
        loss = train(*random_training_set(args.chunk_len))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[elapsed time: %s, epoch: %d,  percent complete: %d%%, loss: %.4f]' % (time_since(start), epoch, epoch / args.epochs * 100, loss))
            if args.sequencer == 'GRU':
                chunk_len = args.pint
                start_index = random.randint(0, file_len - chunk_len)
                end_index = start_index + chunk_len + 1
                init_str = file[start_index:end_index]
                print(generate_GRU(model, init_str, 100), '\n')
            elif args.sequencer == 'CNN' or 'Att':
                init_str,_ = random_training_set(args.pint)
                print(generate_CNN(model, init_str, 100), '\n')

    print("Saving...")
    save()

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

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

