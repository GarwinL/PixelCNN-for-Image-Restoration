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

def denoise_dataset(dataset, config, net):
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, gray_scale=False, num_workers=0); 
    test_loader = get_loader_denoising('../../../datasets/' + dataset, 1, train=False, gray_scale=True, crop_size=[config.crop_size, config.crop_size])  #256
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    psnr_sum = 0
    ssim_sum = 0
    cnt = 0
    step = 0
    
    description = 'Denoising_dataset_' + dataset
    
    logfile = open(config.directory.joinpath(description + '.txt'),'w+')
    
    #writer_tensorboard =  SummaryWriter(comment=description)
    writer_tensorboard =  SummaryWriter(config.directory.joinpath(description))
    writer_tensorboard.add_text('Config parameters', config.config_string)
    
    # Iterate through dataset
    for image, label in test_loader:
        cnt += 1
        
        image = torch.tensor(image,dtype=torch.float32)
        
        y = add_noise(image, torch.tensor(config.sigma), torch.tensor(0.))
        
        # Initialization of parameter to optimize
        x = torch.tensor(y.to(device),requires_grad=True);
        
        #PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu()) Optimizing parameters
        sigma = torch.tensor(torch.tensor(25.)*2/255, dtype=torch.float32).to(device);
        alpha = torch.tensor(config.alpha, dtype=torch.float32).to(device);
        
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
            optimizer = config.optimizer(params, lr=config.lr, betas=[0.9,0.8]) 
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, config.linesearch, scheduler, config.continuous_logistic,image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=None)
        for i in range(config.n_epochs):
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
            #print('SSIM: ', ssim)
            
            # Save best SSIM and PSNR
            if ssim >= best_ssim:
                best_ssim = ssim
                
            if psnr >= best_psnr:
                best_psnr = psnr
                step = i+1
                psnr_ssim = ssim
                conv_cnt = 0
            else: 
                conv_cnt += 1
    
            #if conv_cnt>config.control_epochs: break;
            
            #if x.grad.sum().abs() < 10 and i > 50: break;
        
        
        psnr = PSNR(x[0,:,:,:].cpu().clamp(min=-1,max=1), image[0,:,:,:].cpu())
        ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=0,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)        
        
        # tensorboard
        x_plt = (x+1)/2
        writer_tensorboard.add_scalar('Optimize/Best_PSNR', best_psnr, cnt)
        writer_tensorboard.add_scalar('Optimize/Best_SSIM', best_ssim, cnt)
        writer_tensorboard.add_scalar('Optimize/SSIM_to_best_PSNR', psnr_ssim, cnt)
        image_grid = make_grid(x_plt, normalize=True, scale_each=True)
        writer_tensorboard.add_image('Image', image_grid, cnt)
        
        
        print('Image ', cnt, ': ', psnr, '-', ssim)
        logfile.write('PSNR_each:  %f - step %f\r\n' %(psnr,step))
        logfile.write('PSNR_best: %f\r\n' %best_psnr)
        logfile.write('SSIM_each:  %f\r\n' %ssim)
        logfile.write('SSIM_best:  %f\r\n' %best_ssim)
        psnr_sum += psnr
        ssim_sum += ssim
        #if cnt == 1: break;
        
    psnr_avg = psnr_sum/cnt
    ssim_avg = ssim_sum/cnt
    print(psnr_avg)
    print(ssim_avg)
    logfile.write('PSNR_avg:  %f\r\n' %psnr_avg)
    logfile.write('SSIM_avg: %f\r\n' %ssim_avg)
    
    logfile.close()
    writer_tensorboard.close()
    
    return psnr_avg, ssim_avg
    
    
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
    