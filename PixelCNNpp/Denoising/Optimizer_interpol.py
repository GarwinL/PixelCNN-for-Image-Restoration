# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:12:03 2018

@author: garwi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:43:48 2018

@author: garwi
"""
import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from PixelCNNpp.network import PixelCNN
import numpy as np
import torch.distributions.normal as normal
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import skimage.measure as measure
from tensorboardX import SummaryWriter
from PixelCNNpp.utils import *
from Denoising.denoising import Denoising
from Denoising.utils import PSNR, img_rot90, add_noise
from Denoising.config import BaseConfig
import time
import keyboard

def c_ssim(im1, im2, data_range):
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    #im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    #im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_ssim(im1, im2, data_range=data_range)


if __name__ == '__main__':
    
    config = BaseConfig().initialize()
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../../datasets/BSDS68', 1, train=False, num_workers=0, crop_size=[80,80])
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    image, label = next(data_iter);
    image, label = next(data_iter);
    image, label = next(data_iter);
    #image, label = next(data_iter);  
    #image, label = next(data_iter);
    #image, label = next(data_iter);
    
    
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
#    net = torch.nn.DataParallel(PixelCNN(nr_resnet=3, nr_filters=100,
#            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
   
    net = PixelCNN(nr_resnet=3, nr_filters=100,
            input_channels=1, nr_logistic_mix=10);
    
    # continous - discrete - continous_new
    prior_loss = 'continous_new'
    net.load_state_dict(torch.load('../' + config.net_name))
 
    net.to(device)

    net.eval()

    image = torch.tensor(image,dtype=torch.float32)
    
    sigma = torch.tensor(config.sigma)
    mean = torch.tensor(0.)
    delta = 0.25
    y = add_noise(image, sigma, mean)
 
    # Initialization of parameter to optimize
    res = torch.tensor(y).to(device);
    interpol_noisy = torch.tensor(y).to(device);
    
    # Optimizing parameters
    sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);    
    alpha = torch.tensor(config.alpha, dtype=torch.float32).to(device); 
    #+ str(config.net_name)
    
    description = '_waterloo_gray_32x32_denoising_' + prior_loss + '_alpha=' + str(config.alpha)
    
    writer_tensorboard =  SummaryWriter(comment=description) 

    
    writer_tensorboard.add_text('Config parameters', config.config_string)
        
    conv_cnt=0
    best_psnr=0
    best_ssim=0
    
    for n in range(2):
        
        interpol_noisy = delta*res.detach() + (1-delta)*interpol_noisy
        print('PSNR Interpolated: ', PSNR(interpol_noisy[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
        
        # Initialization of parameter to optimize
        x = torch.tensor(interpol_noisy.clamp(min=-1,max=1).to(device),requires_grad=True);
        
        params=[x]
        
        optimizer = config.optimizer(params, lr=config.lr) 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, scheduler, image, interpol_noisy, net, sigma, alpha, net_interval=1, writer_tensorboard=writer_tensorboard)
        
        start = time.time()
        for i in range(30*(n+1)):#config.n_epochs):                       
            res, gradient = denoise(x, interpol_noisy, image, i)
            
            psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
            ssim = c_ssim(x.data[0,0,:,:].cpu().detach().clamp(min=-1,max=1).numpy(), image.data[0,0,:,:].cpu().numpy(), data_range=x.cpu().detach().clamp(min=-1,max=1).max().numpy() - x.cpu().detach().clamp(min=-1,max=1).min().numpy())
            print('SSIM: ', ssim)
                
            # Save best SSIM and PSNR
            if ssim >= best_ssim:
                best_ssim = ssim
                    
            if psnr >= best_psnr:
                best_psnr = psnr
            else: 
                conv_cnt += 1
                
            #if conv_cnt>2: break;
            
            if keyboard.is_pressed('s'): break;
            
       

        calc_time = time.time() - start
        gradient = gradient.detach().cpu().numpy()[0,0]
        gradient /= np.max(np.abs(gradient))
        gradient_plt = (interpol_noisy.clamp_(min=-1,max=1)+1)/2
        
        print('Time: ', calc_time) 
    
        check_param = res.cpu().detach().numpy()[0,0]
        #========== Plotting ==========#
        y_plt = (y + 1)/2
        x_plt = (res.cpu() + 1)/2
        image_plt = (image + 1)/2
        fig, axs = plt.subplots(3,2, figsize=(8,8))    
        im00=axs[0,0].imshow(y_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
        fig.colorbar(im00, ax=axs[0,0])
        im01=axs[0,1].imshow((x_plt[0,0,:,:]-y_plt[0,0,:,:].cpu()).detach().numpy(), cmap='gray')
        fig.colorbar(im01, ax=axs[0,1])
        im10=axs[1,0].imshow(x_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
        fig.colorbar(im10, ax=axs[1,0])
        im11=axs[1,1].imshow(gradient_plt[0,0], cmap='gray')
        fig.colorbar(im11, ax=axs[1,1])
        im20=axs[2,0].imshow(image_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
        fig.colorbar(im20, ax=axs[2,0])
        im21=axs[2,1].imshow((x_plt[0,0,:,:].cpu()-image_plt[0,0,:,:]).detach().numpy(), cmap='gray')
        fig.colorbar(im21, ax=axs[2,1])
        
        fig.canvas.flush_events()
        fig.canvas.draw()
    
    result = res[0,0,:,:].cpu().detach().numpy()
    orig = image[0,0,:,:].cpu().detach().numpy()
    offset = y[0,0,:,:].cpu().detach().numpy()-res[0,0,:,:].cpu().detach().numpy()
    
    
    #plt.imshow(y[0,0,:,:].cpu().detach().numpy(),cmap='gray')
    #plt.colorbar()
    
    print('Best PSNR: ', best_psnr)
    print('Noisy_Image: ', PSNR(interpol_noisy[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
    print('Denoised_Image: ', PSNR(x[0,0,:,:].cpu().clamp_(min=-1,max=1), image[0,0,:,:].cpu().clamp_(min=-1,max=1)))
    print('Offset: ', torch.sum(y[0,0,:,:].cpu()-x[0,0,:,:].cpu()))
    
    #psnr = measure.compare_psnr(x_plt.data[0,0,:,:].cpu().numpy(), image_plt.data[0,0,:,:].cpu().numpy())
    
    print('SSIM_noisy: ', c_ssim(image.data[0,0,:,:].cpu().numpy(), y.data[0,0,:,:].cpu().numpy(), data_range=y.cpu().max().numpy() - y.cpu().min().numpy()))
    print('SSIM: ', c_ssim(x.detach().data[0,0,:,:].clamp_(min=-1,max=1).cpu().numpy(), image.data[0,0,:,:].cpu().numpy(), data_range=x.detach().clamp_(min=-1,max=1).cpu().max().numpy() - x.detach().clamp_(min=-1,max=1).cpu().min().numpy()))
    print('Best_SSIM: ', best_ssim)
    
    #save_image(x, 'IdentifieMode.png')
    writer_tensorboard.close()
    