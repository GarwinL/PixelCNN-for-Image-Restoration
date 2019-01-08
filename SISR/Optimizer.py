# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:43:48 2018

@author: garwi
"""

import torch
from Dataloader.dataloader import get_loader_denoising
from SISR.SR import SISR
from PixelCNNpp.network import PixelCNN
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from SISR.utils import rescaling, c_psnr, c_ssim
from scipy import misc
from torchvision.utils import save_image
from SISR.config import BaseConfig
import keyboard
from tensorboardX import SummaryWriter
from sklearn.preprocessing import normalize
from skimage.measure import compare_psnr

# Resize Image by averaging
def imresize_half(img,shape):
    sh = shape[0],img.shape[0]//shape[0],shape[1],img.shape[1]//shape[1]
    return img.reshape(sh).mean(-1).mean(1)

if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1) 
    
    config = BaseConfig().initialize()
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../datasets/Set12', 1, train=False, num_workers=0, crop_size=[config.crop_size,config.crop_size])
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    for i in range(8):
        image, label = next(data_iter);
        
    img_downsampled = rescaling(image, 1/3)
    
    hr_init = rescaling(img_downsampled, 3, mode=config.interpol)
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=2, nr_filters=80,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
   
#    net = PixelCNN(nr_resnet=3, nr_filters=36,
#            input_channels=1, nr_logistic_mix=10);
    
    # continous - discrete - continous_new
    net.load_state_dict(torch.load('../Net/' + config.net_name))
 
    net.to(device)

    net.eval()
    
    description = 'superresolution_' + '_alpha=' + str(config.alpha)
    
    writer_tensorboard =  SummaryWriter(config.directory.joinpath(description)) 

    
    writer_tensorboard.add_text('Config parameters', config.config_string)

    y = torch.tensor(img_downsampled)
    x = torch.tensor(hr_init.to(device),requires_grad=True)
    
    
# =============================================================================
#     # Corrupt image with Gaussian Noise -> noisy image of [32,32]
#     mean =  torch.tensor(0.);
#     #var = torch.tensor(0.04);
#     sigma = 25.*2/255 # 255->[-1,1]: sigma*2/255
#     gauss = normal.Normal(mean,sigma)
#     noise = gauss.sample(image.size())
#     gauss = noise.reshape(image.size())
#     y = torch.tensor(image + gauss);
#     
#     # Initialization of parameter to optimize
#     x = torch.tensor(y.to(device),requires_grad=True);
# =============================================================================
    
    # Optimizing parameters
    sigma = torch.tensor(config.sigma*2/255, dtype=torch.float32).to(device);
    alpha = torch.tensor(config.alpha, dtype=torch.float32).to(device);
    
    y = y.to(device)
    
    params=[x]
    
    if config.linesearch:
        optimizer = config.optimizer(params, lr=config.lr, history_size=10, line_search='Wolfe', dtype=torch.float32,  debug=True) 
    else:
        optimizer = config.optimizer(params, lr=config.lr, betas=[0.9,0.8])#, tolerance_grad = 1, tolerance_change=1) 
        
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
    super_resolution = SISR(optimizer, config.linesearch, scheduler, config.continuous_logistic, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=writer_tensorboard)
    
    conv_cnt=0.
    best_psnr=0.
    best_ssim=0.
    worst_psnr=0.
    
    for i in range(config.n_epochs):        
        x, gradient, loss = super_resolution(x, y, image, i)
        
        psnr = c_psnr(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
        ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
        print('SSIM: ', ssim)
            
        # Save best SSIM and PSNR
        if ssim >= best_ssim:
            best_ssim = ssim
                
        if psnr >= best_psnr:
            best_psnr = psnr
            conv_cnt = 0
        else: 
            conv_cnt += 1
            
        if psnr < worst_psnr:
            worst_psnr = psnr
        
            
        if keyboard.is_pressed('*'): break;
        
    
    print('Original: ', image.size())
    print('LR: ', y.size())
    print('HR: ', x.size())
    
    
    
    hr_init_plt = (hr_init + 1)/2#.clamp(min=-1,max=1) + 1)/2
    x_plt = (x + 1)/2
    image_plt = (image + 1)/2
    
    y_plt = (y+1)/2
    hr_bicubic = imresize(y_plt[0,0].cpu().numpy(), 200, interp='bicubic')
    #hr_bicubic=hr_bicubic.astype(np.float32)
    #hr_bicubic = hr_bicubic/255.
    #hr_bicubic_plt = (hr_bicubic + 1)/2
    
    #Plotting
    fig, axs = plt.subplots(2,2, figsize=(8,8))    
    im00=axs[0,0].imshow(hr_init_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im00, ax=axs[0,0])
    im01=axs[0,1].imshow(hr_bicubic, cmap='gray')
    fig.colorbar(im01, ax=axs[0,1])
    im10=axs[1,0].imshow(x_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im10, ax=axs[1,0])
    im11=axs[1,1].imshow(image_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im11, ax=axs[1,1])
    
    res = x[0,0,:,:].cpu().detach().numpy()
    #orig = image[0,0,:,:].cpu().detach().numpy()
    
    
    #plt.imshow(x[0,0,:,:].cpu().detach().numpy(),cmap='gray')
    #plt.colorbar()
    
    #PSNR of bicubic interpolation
    image_test = imresize(image_plt[0,0].cpu().numpy(), 100, interp='bicubic')
    print('Bicubic_interpolation_PSNR: ', compare_psnr(image_test, hr_bicubic))
    print('Bicubic_interpolation_SSIM: ', c_ssim(image_test, hr_bicubic, data_range=255, gaussian_weights=True))
    #print('Bicubic_interpolation: ', c_psnr(torch.tensor(hr_bicubic), torch.tensor(image[0,0,:,:].cpu())))
    print('Biliniear_interpolation: ', c_psnr(hr_init[0,:,:,:].cpu(), image[0,:,:,:].cpu()))
    print('Optimized : ', c_psnr(x[0,:,:,:].cpu(), image[0,:,:,:].cpu()))
    #print('SSIR: ', PSNR(x[0,0,:,:].cpu(), torch.tensor(image/122.5-1, dtype=torch.float32)))
    
    #save_image(x, 'Denoised.png')
    