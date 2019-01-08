import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import skimage.measure as measure
import torch.distributions.normal as normal

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs))


'''For Optimizing'''

### Logistic Mixture Continous - derivative of Sigmoid
def logistic_mixture_continous(x,mean,sigma, weights):
    ''' Logistic Mixture Model '''
    # x -> Matrix of size(size(x),size(e.g. mean))
    xs = [int(y) for y in x.size()]
    # x: [batch_size, 32, 32, range_size(x), 1]
    x = x.contiguous().view(xs + [1])
    logistics = sigma*F.sigmoid(sigma*(x-mean))*(1-F.sigmoid(sigma*(x-mean)));
    weighted_logistics = logistics*weights
     
    # Sum over last dim (logistic mixtures) -> value for each intensity (-1-1)
    return torch.sum(weighted_logistics, dim=len(weighted_logistics.size())-1);

### Log of Logistic Mixture 
def log_logistic_mixture(x,mean,sigma, weights):
    ''' Logistic Mixture Model '''
    # x -> Matrix of size(size(x),size(e.g. mean))
    xs = [int(y) for y in x.size()]
    x = x.contiguous().view(xs + [1])
    logistics = (F.sigmoid(sigma*(x+1/255.-mean))-F.sigmoid(sigma*(x-1./255-mean)));
    log_logistic = torch.log(torch.clamp(logistics,min=1e-5)) + weights 
     
    # Sum over last dim (logistic mixtures) -> value for each intensity (-1-1)
    return torch.sum(log_logistic, dim=len(log_logistic.size())-1);

# =============================================================================
# ### Log Logistic Mixture Continous - derivative of Sigmoid
# def logistic_mixture_continous_test(x,means,scales, weights):
#     """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
#     # Pytorch ordering
#     xs = [int(y) for y in x.size()]
#     x = x.contiguous().view(xs + [1])
# 
#     centered_x = x - means
#     sigmoid_fun = F.sigmoid(scales * (centered_x))
#     #print('Sigmoid_Fun: ', torch.sum(sigmoid_fun))
#     logistic_delta = scales*sigmoid_fun*(1-sigmoid_fun)
#     #print('logistic: ', logistic_delta)
#     
#     log_logistic_delta  = torch.log(torch.clamp(logistic_delta, min=1e-12))
#     log_probs        = torch.sum(log_logistic_delta, dim=3) + log_prob_from_logits(weights)
#     
#     return  torch.sum(log_probs, dim=len(log_probs.size())-1);
# =============================================================================

### Log Logistic Mixture Continous - derivative of Sigmoid
def log_logistic_mixture_continous(x,mean,sigma, weights):
    ''' Logistic Mixture Model '''
    # x -> Matrix of size(size(x),size(e.g. mean))
    logistics = sigma*F.sigmoid(sigma*(x-mean))*(1-F.sigmoid(sigma*(x-mean)));
    weighted_logistics = torch.log(torch.clamp(logistics,min=1e-5)) + weights
     
    # Sum over last dim (logistic mixtures) -> value for each intensity (-1-1)
    return torch.sum(weighted_logistics, dim=len(weighted_logistics.size())-1);

### Log Logistic Mixture Continous - derivative of Sigmoid
def nl_logistic_mixture_continous_test(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    #sigmoid_fun = F.sigmoid(inv_stdv * (centered_x))
    #print('Sigmoid_Fun: ', torch.sum(sigmoid_fun))
    #logistic_delta = inv_stdv*sigmoid_fun*(1-sigmoid_fun)
    #print('logistic: ', logistic_delta)
    
    #log_logistic_delta  = torch.log(torch.clamp(logistic_delta, min=1e-12))
    
    mid_in = inv_stdv * centered_x
    log_logistic_delta  = mid_in - log_scales - 2.*F.softplus(mid_in) - np.log(127.5)
    log_probs        = torch.sum(log_logistic_delta, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs))

def add_noise(img, sigma, mean):   
    sigma = sigma*2/255 # 255->[-1,1]: sigma*2/255
    gauss = normal.Normal(mean,sigma)
    noise = gauss.sample(img.size())
    gauss = noise.reshape(img.size())
    img_noisy = torch.tensor(img + gauss);
    
    return img_noisy

def c_psnr(x,y,peak=2):
    '''PSNR of restored image'''
    dim=1
    for i in range(len(x.size())):
        dim *= x.size(i)
        
    mse = 1/(dim)*torch.sum((x-y)**2);
    psnr = 10*torch.log10(peak**2/mse);
    return psnr;

def c_ssim(im1, im2, data_range, gaussian_weights=None):
    return measure.compare_ssim(im1, im2, data_range=data_range, gaussian_weights=gaussian_weights)

def img_rot90(img):
    inv_idx = torch.arange(img.size(-1)-1, -1, -1).long()    
    img_rot = img.permute(0,1,3,2)[...,inv_idx]
    
    return img_rot

# Downsampling of image to desired scale
def rescaling(img, scale, mode='bilinear'):
    img_sz = [int(y) for y in img.size()]
    img_scaled = torch.nn.functional.upsample(img, (img_sz[2]*scale, img_sz[3]*scale), mode=mode)
    
    return img_scaled;