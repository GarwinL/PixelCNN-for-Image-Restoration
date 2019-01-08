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
from SISR.utils import rescaling

#logit = []

# Prior calculated by PixelCNN - x = [batch_size, 1, 32, 32], logit=[batch_size, 30, 32, 32]
def nll_prior(net, net_interval, step, x):
    if step%net_interval == 0:
        global logit
        logit = net(x) #.detach() TEST
#    else:
#        with torch.no_grad():   
#        #x_nograd = x.detach()
#            logit = net(x)
            
    logistic = nl_logistic_mixture_continous_test(x, logit)
    
    return logistic

# negative log likelihood
def lagrangian(x,y,mu,d):
    #x = x.reshape(y.shape)
    #x = (x+1)*122.5
    #y = (y+1)*122.5
    x_downscaled = rescaling(x, 0.5)
    lagrangian = 0.5*mu*torch.norm(x_downscaled-d,p=2)**2
    
    #print(torch.sum(nllh))
    return lagrangian;

# gradient of negative log likelihood
def grad_nllh(x,y,sigma):
    #x = x.reshape(y.shape)
    grad_nllh = (1/(sigma**2))*(x-y);
    return grad_nllh;

# log-posterior for restoration
def logposterior(x, y, mu, d, alpha, net, net_interval, step):
    #x = x.reshape(y.shape)
    return torch.sum(lagrangian(x,y,mu,d)) + torch.sum(alpha*nll_prior(net, net_interval, step, x));
    #return torch.sum(alpha*nll_prior(net, net_interval, step, x));