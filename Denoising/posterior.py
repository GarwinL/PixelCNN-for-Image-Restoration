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
from Denoising.utils import nl_logistic_mixture_continous_test, discretized_mix_logistic_loss_1d

#logit = []

# Prior calculated by PixelCNN - x = [batch_size, 1, 32, 32], logit=[batch_size, 30, 32, 32]
def nll_prior(net, cont_logistic, net_interval, step, x):
    if step%net_interval == 0:
        global logit
        logit = net(x) #.detach() TEST
#    else:
#        with torch.no_grad():   
#        #x_nograd = x.detach()
#            logit = net(x)
        
    # Continous or discretized logistic   
    if cont_logistic: logistic = nl_logistic_mixture_continous_test(x, logit)
    else: logistic = discretized_mix_logistic_loss_1d(x, logit)
    
    del logit
    
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
def logposterior(x, y, cont_logistic, sigma, alpha, net, net_interval, step):
    #x = x.reshape(y.shape)
    return torch.sum(nllh(x,y,sigma)) + torch.sum(alpha*nll_prior(net, cont_logistic, net_interval, step, x));
    #return torch.sum(alpha*nll_prior(net, net_interval, step, x));