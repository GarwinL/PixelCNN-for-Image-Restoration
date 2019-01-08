# -*- coding: utf-8 -*-

from torchvision.utils import make_grid
from SISR.posterior import logposterior # _withTV
import torch
from SISR.utils import c_psnr, rescaling


# Class for executing one denoising step
class SISR(object):
    def __init__(self, optimizer, scheduler, image, y, net, mu, alpha, net_interval = 1, writer_tensorboard=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer_tensorboard = writer_tensorboard
        self.image = image
        self.y = y
        self.net = net
        self.mu = mu
        self.alpha = alpha
        self.net_interval = net_interval # interval of calculating gradient over net (for speed)
        self.d = y
    
    def __call__(self, x, y, image, step):
        
        psnr = c_psnr(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())        
        x_plt = (x + 1)/2
        image_grid = make_grid(x_plt, normalize=True, scale_each=True)
        
        if step == 0:
             if self.writer_tensorboard != None: 
                self.writer_tensorboard.add_scalar('Optimize/PSNR', psnr, step)
                self.writer_tensorboard.add_image('Image', image_grid, step)
 
        #self.optimizer.zero_grad();
        #loss = logposterior(x, y, self.sigma, self.alpha, self.net, self.net_interval, step); #loss = logposterior(x, y, sigma, alpha, net);            
        #loss.backward(retain_graph=True); 
                
        # Closure for LBFGS
        def closure():
            x.data.clamp_(min=-1,max=1)
            self.optimizer.zero_grad();
            loss = logposterior(x, y, self.mu, self.d, self.alpha, self.net, self.net_interval, step);            
            loss.backward()#retain_graph=True);
            print(loss)
            return loss;       
            
        self.optimizer.step(closure);
        self.scheduler.step();
        
        x_transform = rescaling(x, 0.5)
        self.d = self.d - (x_transform - y)
        
        psnr=c_psnr(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())                      
                   
        x_plt = (x + 1)/2
        image_grid = make_grid(x_plt, normalize=True, scale_each=True) 
        
        gradient = x.grad
        #print('gradient: ', torch.sum(torch.abs(gradient)))
        gradient_norm = gradient/torch.max(torch.abs(gradient))
        gradient_plt = (gradient_norm+1)/2
        #gradient_grid = make_grid(gradient_plt, normalize=True, scale_each=True)
        
        if step != None:
            if self.writer_tensorboard != None: 
                self.writer_tensorboard.add_scalar('Optimize/PSNR', psnr, step+1)
                self.writer_tensorboard.add_image('Image', image_grid, step+1)
                #self.writer_tensorboard.add_image('Gradient', gradient_grid, step+1)
                
                #print('Step ', step, ': ', loss);
            print('PSNR: ', step, ' - ', psnr)
        
        
        return x, gradient;