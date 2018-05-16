# https://github.com/spro/practical-pytorch

import torch
import os

from helpers import *
from model import *

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

def generate_GRU(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
        
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    print(CP_B + predicted[:len(prime_str)] + CP_G + predicted[len(prime_str):] + CP_C)
    # return predicted


def generate_CNN(decoder, prime_str, predict_len=100, temperature=0.8):
    
    predicted = ''
    # print('prime_str',prime_str,prime_str.shape[0])
    for i in range(prime_str.shape[0]): 
        predicted += all_characters[prime_str[i]]

    # print('prime predicted string:', predicted)
    inp = prime_str

    for p in range(predict_len):
        # print('inp:', inp, inp.shape)
        position = p/predict_len-0.5 # positional vector to add to input
        output = decoder(inp, position)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        # print(top_i)
        predicted += predicted_char
        # print('predicted:', predicted)
        # shift to left and add new predicted char value:
        inp[:-1] = inp[1:]
        inp[-1] = top_i

    print(CP_B + predicted[:len(prime_str)] + CP_G + predicted[len(prime_str):] + CP_C)
    # return predicted