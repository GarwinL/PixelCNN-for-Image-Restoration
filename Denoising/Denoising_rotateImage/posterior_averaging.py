# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:53:57 2018

@author: garwi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:24:54 2018

@author: garwi
"""
import torch
from Denoising.utils import nl_logistic_mixture_continous_test
from Denoising.utils import img_rot90

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#logit = []

# Prior calculated by PixelCNN - x = [batch_size, 1, 32, 32], logit=[batch_size, 30, 32, 32]
def nll_prior(net, net_interval, step, x):
#    if step%net_interval == 0:
#        global logit
    
    xs = [int(y) for y in x.size()]
    
    x_rot = img_rot90(x)
    x_rotavg = x_rot
    
    x_rotavg = x_rotavg + torch.zeros([4] + xs[1:]).to(device)
    
    for i in range(1,4):
        x_rot = img_rot90(x_rot)
        x_rotavg[i] = x_rot

    #print(x_rotavg.size())
    logit = net(x_rotavg)

            
    logistic = nl_logistic_mixture_continous_test(x_rotavg, logit)/4
    
    return logistic

# negative log likelihood
def nllh(x,y,sigma):
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
def logposterior(x, y, sigma, alpha, net, net_interval, step):
    #x = x.reshape(y.shape)
    return torch.sum(nllh(x,y,sigma)) + torch.sum(alpha*nll_prior(net, net_interval, step, x));
    #return torch.sum(alpha*nll_prior(net, net_interval, step, x));