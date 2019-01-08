# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:32:12 2018

@author: garwi
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from network import PixelCNN

# Calculate net size
def modelsize(net, input_size):
    bits = 4
    input_bits = np.prod(input_size)*bits
    print('Input: ', input_bits) # 32768
    
    mods = list(net.modules())
    for i in range(1,len(mods)):
        m = mods[i]
        p = list(m.parameters())
        sizes = []
        for j in range(len(p)):
            sizes.append(np.array(p[j].size()))
    
    total_bits = 0
    for i in range(len(sizes)):
        s = sizes[i]
        bits = np.prod(np.array(s))*bits
        total_bits += bits
    
    print('Params: ', total_bits)
    
    input_ = torch.ones(*input_size, requires_grad=True)
    mods = list(net.modules())
    out_sizes = []
    for i in range(1, len(mods)):
        m = mods[i]
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out
    
    total_bits = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        bits = np.prod(np.array(s))*bits
        total_bits += bits
    
    # multiply by 2
    # we need to store values AND gradients
    total_bits *= 2
    print('Intermediate variables: ', total_bits)

# Load PixelCNN
net = PixelCNN(nr_resnet=5, nr_filters=160, 
               input_channels=1, nr_logistic_mix=10);

input_size = (1, 1, 32, 32)

modelsize(net, input_size)

