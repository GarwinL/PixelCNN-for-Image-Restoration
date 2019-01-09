# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:43:48 2018

@author: garwi
"""
import sys
# Add sys path
sys.path.append('../../')

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
from Denoising.denoising import Denoising
from Denoising.utils import PSNR, c_ssim

#Device for computation (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def optimizeMAP(data_list, scale, net, config):
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    psnr_sum = 0.
    ssim_sum = 0.
    step_sum = 0.
    cnt = 0
    
    description = 'Parametertraining_alpha_' + str(scale)
    
    #writer_tensorboard =  SummaryWriter(comment=description)
    writer_tensorboard =  SummaryWriter(config.directory.joinpath(description))
    writer_tensorboard.add_text('Config parameters', config.config_string)
    
    logfile = open(config.directory.joinpath(description + '.txt'),'w+')
    
    for image, y in data_list: 
        
        cnt += 1
    
        image = torch.tensor(image,dtype=torch.float32)
        
        # Initialization of parameter to optimize
        x = torch.tensor(y.to(device),requires_grad=True);
        
        #PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu()) Optimizing parameters
        sigma = torch.tensor(config.sigma*2/255, dtype=torch.float32).to(device);
        alpha = torch.tensor(scale, dtype=torch.float32).to(device);
        
        y = y.to(device)
        
        params=[x]
        
        optimizer = config.optimizer(params, lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, scheduler, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=None)
        
        conv_cnt=0.
        best_psnr=0.
        best_ssim=0.
        worst_psnr=0.
        step = 0
        optimal_step = 0.
        
        for i in range(2):#config.n_epochs):
    
            x, gradient = denoise(x, y, image, i)
            
            
                
            psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
            ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
            print('SSIM: ', ssim)
                
            # Save best SSIM and PSNR
            if ssim >= best_ssim:
                best_ssim = ssim
                    
            if psnr >= best_psnr:
                best_psnr = psnr
                psnr_ssim = ssim
                x_plt = (x+1)/2
                optimal_step = i+1
            else: 
                conv_cnt += 1
                
            if psnr < worst_psnr:
                worst_psnr = psnr
                
            #if keyboard.is_pressed('s'): break;
            if conv_cnt > 2: break;
        
        psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
        ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)        
       
        # tensorboard
        writer_tensorboard.add_scalar('Optimize/PSNR', best_psnr, cnt)
        writer_tensorboard.add_scalar('Optimize/SSIM', best_ssim, cnt)
        writer_tensorboard.add_scalar('Optimize/SSIM_to_best_PSNR', psnr_ssim, cnt)
        image_grid = make_grid(x_plt, normalize=True, scale_each=True)
        writer_tensorboard.add_image('Image', image_grid, cnt)
            
            
        print('Image ', cnt, ': ', psnr, '-', ssim)
        logfile.write('PSNR_each:  %f - step %f\r\n' %(psnr,step))
        logfile.write('PSNR_best: %f\r\n' %best_psnr)
        logfile.write('SSIM_each:  %f\r\n' %ssim)
        logfile.write('SSIM_best:  %f\r\n' %best_ssim)
        psnr_sum += best_psnr
        ssim_sum += best_ssim
        step_sum += optimal_step
        #if cnt == 1: break;

    
    psnr_avg = psnr_sum/cnt
    ssim_avg = ssim_sum/cnt
    step_avg = step_sum/cnt
    print(psnr_avg)
    print(ssim_avg)
    logfile.write('PSNR_avg:  %f\r\n' %psnr_avg)
    logfile.write('SSIM_avg: %f\r\n' %ssim_avg)
    
    logfile.close()
    writer_tensorboard.close()
    
    return psnr_avg, ssim_avg, step_avg
    