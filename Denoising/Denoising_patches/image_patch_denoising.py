# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:19:25 2018

@author: garwi
"""

import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from PixelCNNpp.network import PixelCNN
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from Denoising.Denoising_patches.utils import patchify, aggregate
from Denoising.utils import PSNR, c_ssim, add_noise
from Denoising.denoising import Denoising
import keyboard

def patch_denoising(dataset, config, net):
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #mat_eng = matlab.engine.start_matlab()
    #mat_eng.cd(r'C:\Users\garwi\Desktop\Uni\Master\3_Semester\Masterthesis\Implementation\DnCNN\DnCNN\utilities')
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../../datasets/' + dataset, 1, train=False, gray_scale=True, crop_size=None)#[140,140])#[config.crop_size, config.crop_size])  #256
    
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
        
        img_size = image.size()  
      
        #Add noise to image
        sigma = torch.tensor(config.sigma)
        mean = torch.tensor(0.)
        noisy_img = add_noise(image, sigma, mean)
        
        # Size of patches
        patch_size = [256,256]
        
        # Cop and create array of patches
        noisy_patches, upper_borders, left_borders = patchify(noisy_img, patch_size)
        image_patches, _, _ = patchify(image, patch_size)
        
        print(image_patches.size())
        
        # Optimizing parameters
        sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);    
        alpha = torch.tensor(config.alpha, dtype=torch.float32).to(device);    
        
        denoised_patches = torch.zeros(noisy_patches.size())
        
        for i in range(noisy_patches.size(0)):
            # Initialization of parameter to optimize
            x = torch.tensor(noisy_patches[i].to(device),requires_grad=True);
            
            img = image_patches[i]
            
            y = noisy_patches[i].to(device)
            
            params=[x]
        
            if config.linesearch:
                optimizer = config.optimizer(params, lr=config.lr, history_size=10, line_search='Wolfe', debug=True) 
            else:
                optimizer = config.optimizer(params, lr=config.lr, betas=[0.9,0.8])
                
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
            denoise = Denoising(optimizer, config.linesearch, scheduler, config.continuous_logistic, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=None)
            
            conv_cnt=0.
            best_psnr=0.
            best_ssim=0.
            psnr_ssim=0.
        
            for j in range(2):#config.n_epochs):
    
                x, gradient, loss = denoise(x, y, img, j)
                
                
                    
                psnr = PSNR(x[0,:,:,:].cpu(), img[0,:,:,:].cpu())
                ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((img.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
                print('SSIM: ', ssim)
                    
                # Save best SSIM and PSNR
                if ssim >= best_ssim:
                    best_ssim = ssim
                    
                if psnr >= best_psnr:
                    best_psnr = psnr
                    step = j+1
                    psnr_ssim = ssim
                    conv_cnt = 0
                else: 
                    conv_cnt += 1
                    
                if keyboard.is_pressed('*'): break;
                
            #x_plt = (x+1)/2    
            denoised_patches[i] = x.detach().cpu()
    
     
        img_denoised = aggregate(denoised_patches, upper_borders, left_borders, img_size)
        psnr = PSNR(img_denoised[0,:,:,:].cpu().clamp(min=-1,max=1), image[0,:,:,:].cpu())
        ssim = c_ssim(((img_denoised.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)        
        
        # tensorboard
        img_denoised_plt = (img_denoised+1)/2
        writer_tensorboard.add_scalar('Optimize/Best_PSNR', psnr, cnt)
        writer_tensorboard.add_scalar('Optimize/Best_SSIM', ssim, cnt)
        image_grid = make_grid(img_denoised_plt, normalize=True, scale_each=True)
        writer_tensorboard.add_image('Image', image_grid, cnt)
        
        print('Image ', cnt, ': ', psnr, '-', ssim)
        logfile.write('PSNR_each:  %f - step %f\r\n' %(psnr,step))
        logfile.write('SSIM_each:  %f\r\n' %ssim)
        psnr_sum += psnr
        ssim_sum += ssim
        
#        test1 = img[0,0].numpy()
#        
#        test2 = ((denoised_patches[0][0,0]+1)/2).numpy()
#        test3 = ((denoised_patches[1][0,0]+1)/2).numpy()
#        test4 = ((denoised_patches[2][0,0]+1)/2).numpy()
#        test5 = ((denoised_patches[3][0,0]+1)/2).numpy()
        
        #Plotting
        fig, axs = plt.subplots(2,1, figsize=(8,8))  
        count=0
        for i in range(0,1):        
            axs[count].imshow(((denoised_patches[i][0,0]+1)/2).cpu().detach().numpy(), cmap='gray')
            count +=1
            #fig.colorbar(im, ax=axs[i])
        
        if cnt>7: break;
    
    psnr_avg = psnr_sum/cnt
    ssim_avg = ssim_sum/cnt
    print('PSNR_Avg: ', psnr_avg)
    print('SSIM_Avg: ', ssim_avg)
    logfile.write('PSNR_avg:  %f\r\n' %psnr_avg)
    logfile.write('SSIM_avg: %f\r\n' %ssim_avg)
    
    logfile.close()
    writer_tensorboard.close()
    
    #print(patches.size())
    print(img.size())
    axs[1].imshow(((img_denoised[0,0,:,:]+1)/2).cpu().detach().numpy(), cmap='gray')
    
    print(PSNR(img_denoised[0,:,:,:].cpu(), image[0,:,:,:].cpu()))
    
    return psnr_avg, ssim_avg
    
    
    
    
   