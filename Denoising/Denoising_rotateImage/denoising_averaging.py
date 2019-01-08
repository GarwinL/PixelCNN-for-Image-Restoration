# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:24:18 2018

@author: garwi
"""

from torchvision.utils import make_grid
from Denoising.Denoising_rotateImage.posterior_averaging import logposterior # _withTV
import torch
from Denoising.utils import PSNR, c_ssim


# Class for executing one denoising step
class Denoising(object):
    def __init__(self, optimizer, scheduler, image, y, net, sigma, alpha, net_interval = 1, writer_tensorboard=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer_tensorboard = writer_tensorboard
        self.image = image
        self.y = y
        self.net = net
        self.sigma = sigma
        self.alpha = alpha
        self.net_interval = net_interval # interval of calculating gradient over net (for speed)
    
    def __call__(self, x, y, image, step):
        
        x_plt = (x + 1)/2
        image_grid = make_grid(x_plt, normalize=True, scale_each=True)
        
        if step == 0:
            psnr=PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())
            ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
            
            if self.writer_tensorboard != None: 
                self.writer_tensorboard.add_scalar('PSNR', psnr, step)
                self.writer_tensorboard.add_scalar('SSIM', ssim, step+1)
                self.writer_tensorboard.add_image('Image', image_grid, step)
 
        #self.optimizer.zero_grad();
        #loss = logposterior(x, y, self.sigma, self.alpha, self.net, self.net_interval, step); #loss = logposterior(x, y, sigma, alpha, net);            
        #loss.backward(retain_graph=True); 
                
        # Closure for LBFGS
        def closure():
            x.data.clamp_(min=-1,max=1)
            self.optimizer.zero_grad();
            loss = logposterior(x, y, self.sigma, self.alpha, self.net, self.net_interval, step);            
            loss.backward()#retain_graph=True);
            print(loss)
            return loss;       
            
        self.optimizer.step(closure);
        self.scheduler.step();
        
        psnr=PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())   
        ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)                   
                   
        x_plt = (x + 1)/2
        image_grid = make_grid(x_plt, normalize=True, scale_each=True) 
        
        gradient = x.grad
        #print('gradient: ', torch.sum(torch.abs(gradient)))
        gradient_norm = gradient/torch.max(torch.abs(gradient))
        gradient_plt = (gradient_norm+1)/2
        #gradient_grid = make_grid(gradient_plt, normalize=True, scale_each=True)
        
        if step != None:
            if self.writer_tensorboard != None: 
                self.writer_tensorboard.add_scalar('PSNR', psnr, step+1)
                self.writer_tensorboard.add_scalar('SSIM', ssim, step+1)
                self.writer_tensorboard.add_image('Image', image_grid, step+1)
                #self.writer_tensorboard.add_image('Gradient', gradient_grid, step+1)
                
                #print('Step ', step, ': ', loss);
            print('PSNR: ', step, ' - ', psnr)
        
        
        return x, gradient;
        
