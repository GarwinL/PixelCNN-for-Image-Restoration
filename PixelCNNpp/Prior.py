# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:25:58 2018

@author: garwi
"""

import matplotlib.pyplot as plt
import torch
from dataloader import get_loader_cifar
from skimage import color
from network import PixelCNN
import numpy as np
from utils import *
import torch.nn.functional as F

def prior_dist_rgb(logit, x ,x_range):
    # Rearange to [batch_size, 32, 32, 100]
    logit = logit.permute(0, 2, 3, 1)
    x = x.permute(0, 2, 3, 1)
    ls = [int(y) for y in logit.size()]
    xs = [int(y) for y in x.size()]
    
    # Whole Logistic Mixture Model -- 
    # Convert weights,means, scales to [batch_size, 32, 32, 3, 1, nr_mix]
    nr_mix = 10   
    weights = F.softmax(logit[:,:,:,:nr_mix], dim=3).contiguous().view(ls[:3]+[1]+[nr_mix])
    weights_ = torch.zeros(xs + [1] + [nr_mix], requires_grad=False).to(device)
    weights_[:] = weights.unsqueeze(3)
    
    # logit -> [batch_size, 32, 32, channels, nr_mix*3]
    logit = logit[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = logit[:,:,:,:,:nr_mix]
    scales = torch.exp(-logit[:,:,:,:,nr_mix:nr_mix*2]).contiguous().view(xs+[1]+[nr_mix])
    coeffs = F.tanh(logit[:,:,:,:,nr_mix*2:nr_mix*3])
    
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], requires_grad=False).to(device)
    
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    means = means.contiguous().view(xs+[1]+[nr_mix])
    
    # Create x over intensities(-1 - 1) Format: [batch_size, 32, 32, range_size]
    x_calc=torch.zeros(xs+[int(x_range.size(0))])
    x_calc[:]=torch.reshape(x_range, (1,1,1,1,x_range.size(0)))
    x_calc=x_calc.to(device)
    
    
    logistic = logistic_mixture_continous(x_calc,means,scales, weights_)
    
    return logistic

def prior_dist(logit,x_range):
     # Rearange to [batch_size, 32, 32, 30]
    logit = logit.permute(0, 2, 3, 1)
    ls = [int(y) for y in logit.size()]
    
    # Whole Logistic Mixture Model -- 
    # Convert weights,means, scales to [batch_size, 32, 32, 1, nr_mix]
    nr_mix = 10
    weights = F.softmax(logit[:,:,:,:nr_mix], dim=3).contiguous().view(ls[:3]+[1]+[nr_mix])
    means = logit[:,:,:,nr_mix:2*nr_mix].contiguous().view(ls[:3]+[1]+[nr_mix])
    scales = torch.exp(-logit[:,:,:,nr_mix*2:nr_mix*3]).contiguous().view(ls[:3]+[1]+[nr_mix])
    
    
    # Create x over intensities(-1 - 1) Format: [batch_size, 32, 32, range_size]
    x=torch.zeros(ls[:3]+[int(x_range.size(0))])
    x[:]=torch.reshape(x_range, (1,1,1,x_range.size(0)))
    x=x.to(device)
    
    
    logistic = logistic_mixture_continous(x,means,scales, weights)
    
    return logistic

# =============================================================================
# def prior_dist_test(logit,x_range):
#      # Rearange to [batch_size, 32, 32, 30]
#     logit = logit.permute(0, 2, 3, 1)
#     ls = [int(y) for y in logit.size()]
#     
#     # Whole Logistic Mixture Model -- 
#     # Convert weights,means, scales to [batch_size, 32, 32, 1, nr_mix]
#     nr_mix = 10
#     weights = F.softmax(logit[:,:,:,:nr_mix], dim=3).contiguous().view(ls[:3]+[1]+[nr_mix])
#     means = logit[:,:,:,nr_mix:2*nr_mix].contiguous().view(ls[:3]+[1]+[nr_mix])
#     scales = torch.exp(-torch.clamp(logit[:,:,:,nr_mix*2:nr_mix*3], min=-7.)).contiguous().view(ls[:3]+[1]+[nr_mix])
#     
#     
#     # Create x over intensities(-1 - 1) Format: [batch_size, 32, 32, range_size]
#     x=torch.zeros(ls[:3]+[int(x_range.size(0))])
#     x[:]=torch.reshape(x_range, (1,1,1,x_range.size(0)))
#     x=x.to(device)
#     
#     
#     logistic = logistic_mixture_continous_test(x,means,scales, weights)
#     
#     return logistic
# =============================================================================

if __name__ == '__main__':
    
    #torch.cuda.empty_cache(); # Free cache not a solution
    
    # Load datasetloader
    test_loader = get_loader_cifar('../../datasets/CIFAR10', 1, train=False, num_workers=0, gray_scale=True);
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
    net = PixelCNN(nr_resnet=5, nr_filters=160, 
            input_channels=1, nr_logistic_mix=10);
    
    net.load_state_dict(torch.load('net_epoch_199.pt'))
 
    net.to(device)
    
    net.eval()
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    image, label = next(data_iter);
    #image, label = next(data_iter);
    #image, label = next(iter(test_loader))
    
    # [batch_size, 1, 32, 32]
    image = image.to(device)
    
    # [batch_size, 30, 32, 32]
    logit = net(image)
    
    step_size = 0.001
    x_range=torch.range(-1,1,step_size,dtype=torch.float32)
    
    # Plotting of arbitrary distribution
    logistic_plot = prior_dist(logit,x_range).cpu().detach().numpy()
    
    print(logistic_plot.shape)
    
    tf = 255/2; # Transform step size to [0,255]
    x_range1 = torch.range(0,255,tf*step_size,dtype=torch.float32)
    x_plot = x_range.cpu().detach().numpy()
    
    
    plt.plot(x_plot,(logistic_plot[0,0,1,:]))
    print(sum(logistic_plot[0,0,1,:])*step_size)



    
    
    