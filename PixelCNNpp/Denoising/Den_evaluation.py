# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:16:34 2018

@author: garwi
"""

import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from PixelCNNpp.network import PixelCNN
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from Denoising.utils import PSNR, c_ssim, add_noise
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from Denoising.denoising import Denoising
from Denoising.config import BaseConfig

if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    
    config = BaseConfig().initialize()
    
    logfile = open(config.directory.joinpath('Den_log.txt'),'w+')
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, gray_scale=False, num_workers=0); 
    test_loader = get_loader_denoising('../../../datasets/Set12', 1, train=False, gray_scale=True, crop_size=[80, 80]) 
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    # Load PixelCNN
#    net = PixelCNN(nr_resnet=5, nr_filters=160, 
#            input_channels=1, nr_logistic_mix=10);
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=3, nr_filters=100,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
    
    net.load_state_dict(torch.load('../../Net/net_epoch_5000_standardnet.pt'))
    
    net.to(device)

    net.eval()
    
    psnr_sum = 0
    ssim_sum = 0
    cnt = 0
    step = 0
    
    description = '_waterloo_32x32_gray'
    
    #writer_tensorboard =  SummaryWriter(comment=description)
    writer_tensorboard =  SummaryWriter(config.directory.joinpath(description))
    writer_tensorboard.add_text('Config parameters', config.config_string)
    
    # Iterate through dataset
    for image, label in test_loader:
        cnt += 1
        
        image = torch.tensor(image,dtype=torch.float32)
        
        y = add_noise(image, torch.tensor(25.), torch.tensor(0.))
        
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
            optimizer = config.optimizer(params, lr=config.lr) 
            
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, config.linesearch, scheduler, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=None)
        for i in range(5):
# =============================================================================
#             def closure():
#                 optimizer.zero_grad();
#                 loss = logposterior(x, y, sigma, alpha, logit[0,:,:,:]);            
#                 loss.backward(retain_graph=True);
#                 print(loss)
#                 return loss;
# =============================================================================
            x, gradient = denoise(x, y, image, i)
            
            psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
            ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
            #print('SSIM: ', ssim)
            
            # Save best SSIM and PSNR
            if ssim >= best_ssim:
                best_ssim = ssim
                
            if psnr >= best_psnr:
                best_psnr = psnr
                step = i
                psnr_ssim = ssim
                x_plt = (x+1)/2
                conv_cnt = 0
            else: 
                conv_cnt += 1
    
            if conv_cnt>50: break;
            
            #if x.grad.sum().abs() < 10 and i > 50: break;
        
        
        psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
        ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)        
        
        # tensorboard
        writer_tensorboard.add_scalar('Optimize/Best_PSNR', best_psnr, cnt)
        writer_tensorboard.add_scalar('Optimize/Best_SSIM', best_ssim, cnt)
        image_grid = make_grid(x_plt, normalize=True, scale_each=True)
        writer_tensorboard.add_image('Image', image_grid, cnt)
        
        
        print('Image ', cnt, ': ', psnr, '-', ssim)
        logfile.write('PSNR_each:  %f - step %f\r\n' %(psnr,step))
        logfile.write('PSNR_best: %f\r\n' %best_psnr)
        logfile.write('SSIM_each:  %f\r\n' %ssim)
        logfile.write('SSIM_best:  %f\r\n' %best_ssim)
        psnr_sum += best_psnr
        ssim_sum += best_ssim
        #if cnt == 1: break;
        
    psnr_avg = psnr_sum/cnt
    ssim_avg = ssim_sum/cnt
    print(psnr_avg)
    print(ssim_avg)
    logfile.write('PSNR_avg:  %f\r\n' %psnr_avg)
    logfile.write('SSIM_avg: %f\r\n' %ssim_avg)
    
    logfile.close()
    writer_tensorboard.close()
    
    
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
    