# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:24:18 2018

@author: garwi
"""

import sys
# Add sys path
sys.path.append('../')

from torchvision.utils import make_grid
from Inpainting.posterior import logposterior
import torch
from Denoising.utils import PSNR


# Class for executing one denoising step
class Inpainting(object):
    def __init__(self, optimizer, linesearch, scheduler, cont_logistic, image, y, net, mask, net_interval = 1, writer_tensorboard=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer_tensorboard = writer_tensorboard
        self.image = image
        self.y = y
        self.net = net
        self.mask = mask
        self.net_interval = net_interval # interval of calculating gradient over net (for speed)
        self.loss = 0
        self.grad = 0
        self.linesearch = linesearch
        self.cont_logistic = cont_logistic
    
    def __call__(self, x, y, image, step):
        
        psnr=PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())        
        x_plt = (x + 1)/2
        image_grid = make_grid(x_plt, normalize=True, scale_each=True)
        
        if step == 0:
             
            if self.writer_tensorboard != None: 
                self.writer_tensorboard.add_scalar('Optimize/PSNR', psnr, step)
                self.writer_tensorboard.add_image('Image', image_grid, step)
                
            if self.linesearch: # For LBFGS with linesearch
                x.data.clamp_(min=-1,max=1)
                self.optimizer.zero_grad();
                self.loss = logposterior(x, y, self.cont_logistic, self.net, self.net_interval, step);            
                self.loss.backward()
                    
                gradient = x.grad
                self.grad = self.optimizer._gather_flat_grad()
                
        # Closure for LBFGS
        def closure():
            x.data.clamp_(min=-1,max=1)
            self.optimizer.zero_grad();
            loss = logposterior(x, y, self.cont_logistic, self.net, self.net_interval, step); 
            if self.linesearch == False: loss.backward()
            if x.grad is not None: x.grad.data = x.grad.data*self.mask
            print(loss)
            return loss;       
        
        # Paramaterupdate: Gradient step
        if self.linesearch: 
            # two-loop recursion to compute search direction
            p = self.optimizer.two_loop_recursion(-self.grad)
            
            # perform line search step
            options = {'closure': closure, 'current_loss': self.loss} #
            
            #self.loss, grad, lr, backtracks, clos_evals, grad_evals, desc_dir, fail = self.optimizer.step(p, grad, options=options)
            #self.scheduler.step();
            
            self.loss, self.grad, lr, _, _, _, _, fail = self.optimizer.step(p, self.grad, options=options)
            #self.loss, self.grad, lr, _, _, _, _, fail = self.optimizer.step(options=options)
            print('Fail: ', fail)
            
            
            # compute gradient at new iterate
            #self.loss.backward()
            #self.grad = self.optimizer._gather_flat_grad()
            
            # curvature update
            self.optimizer.curvature_update(self.grad, eps=1e-2, damping=False)           
            
        else:        
            self.loss=self.optimizer.step(closure);
            
#            #Standard Gradient Descent
#            x.data.clamp_(min=-1,max=1)
#            self.loss = logposterior(x, y, self.cont_logistic, self.sigma, self.alpha, self.net, self.net_interval, step); 
#            self.loss.backward()
#            print(self.loss)
#            print(x.grad)
#            
#            x = x - 0.001*x.grad
            
            
        self.scheduler.step()
        
        psnr=PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())                      
                   
        x_plt = (x + 1)/2
        image_grid = make_grid(x_plt, normalize=True, scale_each=True) 
        
        gradient = x.grad
        #print('gradient: ', torch.sum(torch.abs(gradient)))
        #gradient_norm = gradient/torch.max(torch.abs(gradient))
        #gradient_plt = (gradient_norm+1)/2
        #gradient_grid = make_grid(gradient_plt, normalize=True, scale_each=True)
        
        if step != None:
            if self.writer_tensorboard != None: 
                self.writer_tensorboard.add_scalar('Optimize/PSNR', psnr, step+1)
                self.writer_tensorboard.add_image('Image', image_grid, step+1)
                self.writer_tensorboard.add_scalar('Optimize/Loss', self.loss , step+1)
                
                #print('Step ', step, ': ', loss);
            print('PSNR: ', step, ' - ', psnr)
            print('Loss: ', self.loss)
        
        return x, gradient, self.loss;
        
