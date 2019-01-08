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
from tensorboardX import SummaryWriter
from Denoising.Denoising_patches.utils import patchify, aggregate
from Denoising.utils import PSNR, c_ssim, add_noise
from Denoising.config import BaseConfig
from Denoising.denoising import Denoising
import keyboard

if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    torch.backends.cudnn.deterministic = True
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #mat_eng = matlab.engine.start_matlab()
    #mat_eng.cd(r'C:\Users\garwi\Desktop\Uni\Master\3_Semester\Masterthesis\Implementation\DnCNN\DnCNN\utilities')
    
    config = BaseConfig().initialize()
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../../datasets/Set12', 1, train=False, num_workers=0, crop_size=None)
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    for i in range(1):
        image, label = next(data_iter);
    
    image = torch.tensor(image,dtype=torch.float32)
    
    img_size = image.size()  
  
    #Add noise to image
    sigma = torch.tensor(config.sigma)
    mean = torch.tensor(0.)
    noisy_img = add_noise(image, sigma, mean)
    
    # Size of patches
    patch_size = [140,140]
    
    # Cop and create array of patches
    noisy_patches, upper_borders, left_borders = patchify(noisy_img, patch_size)
    image_patches, _, _ = patchify(image, patch_size)
    
    print(image_patches.size())
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=3, nr_filters=100,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
   
#    net = PixelCNN(nr_resnet=3, nr_filters=100,
#            input_channels=1, nr_logistic_mix=10);
    
    # Optimizing parameters
    sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);    
    alpha = torch.tensor(config.alpha, dtype=torch.float32).to(device);
                                         
    # continous - discrete - continous_new
    prior_loss = 'continous_new'
    net.load_state_dict(torch.load('../../Net/' + config.net_name))
 
    net.to(device)

    net.eval()
    
    denoised_patches = torch.zeros(noisy_patches.size())
    
    for i in range(noisy_patches.size(0)):
        # Initialization of parameter to optimize
        x = torch.tensor(noisy_patches[i].clamp(min=-1,max=1).to(device),requires_grad=True);
        
        img = image_patches[i]
        
        y = noisy_patches[i].to(device)
        
        params=[x]
    
        if config.linesearch:
            optimizer = config.optimizer(params, lr=config.lr, history_size=10, line_search='Wolfe', debug=True) 
        else:
            optimizer = config.optimizer(params, lr=config.lr)#, tolerance_grad = 1, tolerance_change=1) 
            
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, config.linesearch, scheduler, config.continuous_logistic, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=None)
        
        conv_cnt=0.
        best_psnr=0.
        best_ssim=0.
        worst_psnr=0.
    
        for j in range(10):#config.n_epochs):

            x, gradient = denoise(x, y, img, j)
            
            
                
            psnr = PSNR(x[0,:,:,:].cpu(), img[0,:,:,:].cpu())
            ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((img.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
            print('SSIM: ', ssim)
                
            # Save best SSIM and PSNR
            if ssim >= best_ssim:
                best_ssim = ssim
                    
            if psnr >= best_psnr:
                best_psnr = psnr
            else: 
                conv_cnt += 1
                
            if psnr < worst_psnr:
                worst_psnr = psnr
                
            if keyboard.is_pressed('s'): break;
            if conv_cnt > 2: break;
            
        denoised_patches[i] = x.detach().cpu()

 
    img_denoised = aggregate(denoised_patches, upper_borders, left_borders, img_size)
    test1 = img[0,0].numpy()
    
    test2 = ((denoised_patches[0][0,0]+1)/2).numpy()
    test3 = ((denoised_patches[1][0,0]+1)/2).numpy()
    test4 = ((denoised_patches[2][0,0]+1)/2).numpy()
    test5 = ((denoised_patches[3][0,0]+1)/2).numpy()
    
    #Plotting
    fig, axs = plt.subplots(2,1, figsize=(8,8))  
    cnt=0
    for i in range(0,1):        
        axs[cnt].imshow(((denoised_patches[i][0,0]+1)/2).cpu().detach().numpy(), cmap='gray')
        cnt +=1
        #fig.colorbar(im, ax=axs[i])
        
    #print(patches.size())
    print(img.size())
    axs[1].imshow(((img_denoised[0,0,:,:]+1)/2).cpu().detach().numpy(), cmap='gray')
    
    print(PSNR(img_denoised[0,:,:,:].cpu(), image[0,:,:,:].cpu()))
    
    
    
    
   