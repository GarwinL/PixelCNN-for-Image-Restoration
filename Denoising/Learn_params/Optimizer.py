# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:16:34 2018

@author: garwi
"""

import sys
# Add sys path
sys.path.append('../../')

import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from Denoising.utils import PSNR, c_ssim, add_noise
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from Denoising.denoising import Denoising
from Denoising.config import BaseConfig

def optimizeMAP(data_list, scale, lr, net, config):
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    best_psnr_sum = 0
    best_ssim_sum = 0
    cnt = 0
    step = 0
    
    description = 'Evaluation_parameter_scale=' + str(scale) + '_lr' + str(lr)
    
    logfile = open(config.directory.joinpath(description + '.txt'),'w+')
    
    #writer_tensorboard =  SummaryWriter(comment=description)
    writer_tensorboard =  SummaryWriter(config.directory.joinpath(description))
    writer_tensorboard.add_text('Config parameters', config.config_string)
    
    #PSNR, SSIM - step size Matrix
    psnr_per_step = np.zeros((len(data_list), config.n_epochs))
    ssim_per_step = np.zeros((len(data_list), config.n_epochs))
    image_list = torch.zeros((len(data_list), config.n_epochs, 1, 1, config.crop_size, config.crop_size))
    
    # Iterate through dataset
    for cnt, (image, y) in enumerate(data_list):
        
        image = torch.tensor(image,dtype=torch.float32)
        
        # Initialization of parameter to optimize
        x = torch.tensor(y.to(device),requires_grad=True);
        
        #PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu()) Optimizing parameters
        sigma = torch.tensor(torch.tensor(config.sigma)*2/255, dtype=torch.float32).to(device);
        alpha = torch.tensor(scale, dtype=torch.float32).to(device); #config.alpha
        
        y = y.to(device)
        
        params=[x]
        
        #Initialize Measurement parameters
        conv_cnt = 0
        best_psnr = 0
        best_ssim = 0
        psnr_ssim = 0
        
        if config.linesearch:
            optimizer = config.optimizer(params, lr=config.lr, history_size=10, line_search='Wolfe', debug=True) 
        else:
            optimizer = config.optimizer(params, lr=lr, betas=[0.9,0.8]) #, momentum=0.88) 
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, config.linesearch, scheduler, config.continuous_logistic,image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=None)
        for i in range(2):#config.n_epochs):
# =============================================================================
#             def closure():
#                 optimizer.zero_grad();
#                 loss = logposterior(x, y, sigma, alpha, logit[0,:,:,:]);            
#                 loss.backward(retain_graph=True);
#                 print(loss)
#                 return loss;
# =============================================================================
            x, gradient, loss = denoise(x, y, image, i)
            
            psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
            ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
            
            #Save psnr in matrix
            psnr_per_step[cnt, i] = psnr.detach().numpy()
            ssim_per_step[cnt, i] = ssim
            
            # tensorboard
            writer_tensorboard.add_scalar('Optimize/PSNR_of_Image'+str(cnt), psnr, i)
            writer_tensorboard.add_scalar('Optimize/SSIM_of_Image'+str(cnt), ssim, i)
            writer_tensorboard.add_scalar('Optimize/Loss_of_Image'+str(cnt), loss, i)
            
            # Save best SSIM and PSNR
            if ssim >= best_ssim:
                best_ssim = ssim
                
            if psnr >= best_psnr:
                best_psnr = psnr
                step = i+1
                psnr_ssim = ssim
                conv_cnt = 0
            else: conv_cnt += 1       
            
            # Save image in list
            image_list[cnt,i] = (x.detach().cpu()+1)/2
    
            #if conv_cnt>config.control_epochs: break;
        
        
        psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
        ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)        
        
        # tensorboard
        writer_tensorboard.add_scalar('Optimize/Best_PSNR', best_psnr, cnt)
        writer_tensorboard.add_scalar('Optimize/Best_SSIM', best_ssim, cnt)
        writer_tensorboard.add_scalar('Optimize/SSIM_to_best_PSNR', psnr_ssim, cnt)        
        
        print('Image ', cnt, ': ', psnr, '-', ssim)
        logfile.write('PSNR_each:  %f - step %f\r\n' %(psnr,step))
        logfile.write('PSNR_best: %f\r\n' %best_psnr)
        logfile.write('SSIM_each:  %f\r\n' %ssim)
        logfile.write('SSIM_best:  %f\r\n' %best_ssim)
        best_psnr_sum += best_psnr
        best_ssim_sum += best_ssim
        #if cnt == 1: break;
    
    psnr_avg = best_psnr_sum/(cnt+1)
    ssim_avg = best_ssim_sum/(cnt+1)
    logfile.write('Best_PSNR_avg:  %f\r\n' %psnr_avg)
    logfile.write('Best_SSIM_avg: %f\r\n' %ssim_avg)
    
    # Logging of average psnr and ssim per step
    log_psnr_per_step = open(config.directory.joinpath(description + '_psnr_per_step.txt'),'w+')
    log_ssim_per_step = open(config.directory.joinpath(description + '_ssim_per_step.txt'),'w+')
    psnr_avg_step = np.mean(psnr_per_step, 0)
    ssim_avg_step = np.mean(ssim_per_step, 0)
    
    for n in range(psnr_avg_step.shape[0]):
        log_psnr_per_step.write('Step %f: %f\r\n' %(n+1, psnr_avg_step[n]))
        log_ssim_per_step.write('Step %f: %f\r\n' %(n+1, ssim_avg_step[n]))
       
    #print(psnr_avg_step.shape)
    #print(psnr_per_step.shape)
    best_step = np.argmax(psnr_avg_step)+1
    
    log_psnr_per_step.write('Best PSNR: %f\r\n' %np.max(psnr_avg_step))
    log_psnr_per_step.write('Step to best PSNR: %f\r\n' %best_step)
    
    logfile.close()
    
    # Save images in tensorboard
    for i in range(len(data_list)):
        image_grid = make_grid(image_list[i, best_step-1], normalize=True, scale_each=True)
        writer_tensorboard.add_image('Image', image_grid, i)
    
    writer_tensorboard.close()
    
    return  np.max(psnr_avg_step), ssim_avg_step[best_step-1], best_step;
    
    
# =============================================================================
#     #Plotting
#     fig, axs = plt.subplots(3,1, figsize=(8,8))    
#     axs[0].imshow(y[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     axs[1].imshow(x[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     axs[2].imshow(image[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     
#     res = x[0,0,:,:].cpu().detach().numpy()
#     orig = image[0,0,:,:].cpu().detach().numpy()
#     
#     
#     #plt.imshow(x[0,0,:,:].cpu().detach().numpy(),cmap='gray')
#     #plt.colorbar()
#     print('Noisy_Image: ', PSNR(y[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
#     print('Denoised_Image: ', PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
#     
#     #save_image(x, 'Denoised.png')
# =============================================================================
    