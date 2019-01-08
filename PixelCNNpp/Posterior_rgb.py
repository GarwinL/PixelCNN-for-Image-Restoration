# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:24:54 2018

@author: garwi
"""

import torch
from utils import log_logistic_mixture, log_prob_from_logits, log_logistic_mixture_continous
import torch.nn.functional as F

#Device for computation (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prior calculated by PixelCNN - x = [batch_size, 1, 32, 32], logit=[batch_size, 30, 32, 32]
def nll_prior(net,x):

    logit = net(x)
    
    logit = logit[0,:,:,:]
    
    x=x[0,:,:,:]
    
    # Rearange to [32, 32, 30]
    logit = logit.permute(1, 2, 0)
    x = x.permute(1, 2, 0)
    ls = [int(y) for y in logit.size()]
    xs = [int(y) for y in x.size()]
    
    # Whole Logistic Mixture Model -- 
    # Convert weights,means, scales to [32, 32, 3, nr_mix]
    nr_mix = 10   
    weights = log_prob_from_logits(logit[:,:,:nr_mix])
    weights_ = torch.zeros(xs + [nr_mix], requires_grad=False).to(device)
    weights_[:] = weights.unsqueeze(2)
    
    # logit -> [batch_size, 32, 32, channels, nr_mix*3]
    logit = logit[:, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = logit[:,:,:,:nr_mix]
    scales = torch.exp(-logit[:,:,:,nr_mix:nr_mix*2]).contiguous().view(xs+[nr_mix])
    coeffs = F.tanh(logit[:,:,:,nr_mix*2:nr_mix*3])
    
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], requires_grad=False).to(device)
    
    m2 = (means[:, :, 1, :] + coeffs[:, :, 0, :]
                * x[:, :, 0, :]).view(xs[0], xs[1], 1, nr_mix)

    m3 = (means[:, :, 2, :] + coeffs[:, :, 1, :] * x[:, :, 0, :] +
                coeffs[:, :, 2, :] * x[:, :, 1, :]).view(xs[0], xs[1], 1, nr_mix)

    means = torch.cat((means[:, :, 0, :].unsqueeze(2), m2, m3), dim=3)
    means = means.contiguous().view(xs+[nr_mix])    
    
    logistic = log_logistic_mixture_continous(x,means,scales, weights_)
    
    #print(logistic.size())
    
    return -logistic

# negative log likelihood
def nllh(x,y,sigma):
    x=x[0,:,:,:]
    y=y[0,:,:,:]
    x = x.permute(1, 2, 0)
    y = y.permute(1, 2, 0)
    #x = x.reshape(y.shape)
    #x = (x+1)*122.5
    #y = (y+1)*122.5
    nllh = 1/(2*sigma**2)*(x-y)**2;
    #print(torch.sum(nllh))
    return nllh;

# gradient of negative log likelihood
def grad_nllh(x,y,sigma):
    #x = x.reshape(y.shape)
    grad_nllh = (1/(sigma**2))*(x-y);
    return grad_nllh;

# log-posterior for restoration
def logposterior(x, y, sigma, alpha, logit):
    #x = x.reshape(y.shape)
    return torch.sum(nllh(x,y,sigma) + alpha*nll_prior(logit,x));