# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:43:48 2018

@author: garwi
"""
import sys
# Add sys path
sys.path.append('../../')


import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from PixelCNNpp_continous.network import PixelCNN
import numpy as np
import torch.distributions.normal as normal
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import skimage.measure as measure
from tensorboardX import SummaryWriter
from PixelCNNpp.utils import *
from Denoising.denoising import Denoising
from Denoising.utils import PSNR, img_rot90, c_ssim, add_noise
from Denoising.config import BaseConfig
import time
import keyboard
import matlab.engine

def sample(model, data):
    model.train(False)
    #data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])#, requires_grad=True)
    data = data.to(device)
    data = torch.tensor(data, requires_grad=False)
    with torch.no_grad():
        out = model(data, sample=True)
        
    out_sample = sample_from_discretized_mix_logistic_1d(out, 10)
    return out_sample


if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    torch.backends.cudnn.deterministic = True
    
    mat_eng = matlab.engine.start_matlab()
    mat_eng.cd(r'C:\Users\garwi\Desktop\Uni\Master\3_Semester\Masterthesis\Implementation\DnCNN\DnCNN\utilities')
    
    config = BaseConfig().initialize()
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../../datasets/BSDS68', 1, train=False, num_workers=0, crop_size=[80,80])
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    for i in range(3):
        image, label = next(data_iter);
    
    
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=3, nr_filters=100,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
   
#    net = PixelCNN(nr_resnet=3, nr_filters=100,
#            input_channels=1, nr_logistic_mix=10);
    
    # continous - discrete - continous_new
    prior_loss = 'continous_new'
    net.load_state_dict(torch.load('../../Net/' + config.net_name))
 
    net.to(device)

    net.eval()

    image = torch.tensor(image,dtype=torch.float32)
    
    sigma = torch.tensor(config.sigma)
    mean = torch.tensor(0.)
    y = add_noise(image, sigma, mean)
    
    #y = torch.tensor(image);
    
    # Initialization of parameter to optimize
    x = torch.tensor(y.clamp(min=-1,max=1));
#    x = torch.tensor(y.clamp(min=-1,max=1).to(device));
 
    
    # Optimizing parameters
    sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);    
    alpha = torch.tensor(config.alpha, dtype=torch.float32).to(device); 
    #+ str(config.net_name)
    
    description = '_waterloo_gray_32x32_denoising_' + prior_loss + '_alpha=' + str(config.alpha)
    
    writer_tensorboard =  SummaryWriter(config.directory.joinpath(description)) 

    
    writer_tensorboard.add_text('Config parameters', config.config_string)
    
    y = y.to(device)   
    
    conv_cnt=0.
    best_psnr=0.
    best_ssim=0.
    worst_psnr=0.
    
    start = time.time()
    for n in range(10):#config.n_epochs):
        #with torch.no_grad():       
        #x.data.clamp_(min=-1,max=1)     
        #torch.set_grad_enabled(True)
#        x_new = torch.tensor(x.detach().to(device),requires_grad=True)
#        
#        params=[x_new]
#    
#        optimizer = config.optimizer(params, lr=config.lr) 
#        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
#        denoise = Denoising(optimizer, scheduler, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=writer_tensorboard)

        ## Turn the images
        if n>0:
            with torch.no_grad():
                x = img_rot90(x)
                y = img_rot90(y)
                image = img_rot90(image)
            
        x = torch.tensor(x.to(device), requires_grad=True)
        
        params=[x]
    
        optimizer = config.optimizer(params, lr=config.lr) #0.05 
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        denoise = Denoising(optimizer, scheduler, image, y, net, sigma, alpha, net_interval=1, writer_tensorboard=writer_tensorboard)

        for i in range(5):
            x, gradient = denoise(x, y, image, i)
            
            psnr = PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
            ssim = c_ssim(((x.data[0,0,:,:]+1)/2).cpu().detach().clamp(min=-1,max=1).numpy(), ((image.data[0,0,:,:]+1)/2).cpu().numpy(), data_range=1, gaussian_weights=True)
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
        

    calc_time = time.time() - start
    gradient = gradient.detach().cpu().numpy()[0,0]
    #gradient_plt = gradient.clamp(min=-1,max=1).detach().cpu().numpy()[0,0]
    gradient_norm = gradient/np.max(np.abs(gradient))
    gradient_plt = (gradient_norm+1)/2
    
    print('Time: ', calc_time)
    
    sample_t = sample(net, x)
    check_param = x.cpu().detach().numpy()[0,0]
    #========== Plotting ==========#
    x = x.cpu().detach().clamp_(min=-1,max=1)
    y_plt = (y + 1)/2
    x_plt = (x + 1)/2
    image_plt = (image + 1)/2
    sample_t_plt = (sample_t +1)/2 
    fig, axs = plt.subplots(3,2, figsize=(8,8))    
    im00=axs[0,0].imshow(y_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im00, ax=axs[0,0])
    im01=axs[0,1].imshow((x_plt[0,0,:,:]-y_plt[0,0,:,:].cpu()).detach().numpy(), cmap='gray')
    fig.colorbar(im01, ax=axs[0,1])
    im10=axs[1,0].imshow(x_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im10, ax=axs[1,0])
    im11=axs[1,1].imshow(gradient_plt, cmap='gray')
    fig.colorbar(im11, ax=axs[1,1])
    im20=axs[2,0].imshow(image_plt[0,0,:,:].cpu().detach().numpy(), cmap='gray')
    fig.colorbar(im20, ax=axs[2,0])
    im21=axs[2,1].imshow((x_plt[0,0,:,:].cpu()-image_plt[0,0,:,:]).detach().numpy(), cmap='gray')
    fig.colorbar(im21, ax=axs[2,1])
    
    res = x[0,0,:,:].cpu().detach().numpy()
    orig = image[0,0,:,:].cpu().detach().numpy()
    offset = y[0,0,:,:].cpu().detach().numpy()-x[0,0,:,:].cpu().detach().numpy()
    
    
    #plt.imshow(y[0,0,:,:].cpu().detach().numpy(),cmap='gray')
    #plt.colorbar()
    
    print('Best PSNR: ', best_psnr)
    print('Noisy_Image: ', PSNR(y[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
    print('Denoised_Image: ', PSNR(x[0,0,:,:].cpu().clamp_(min=-1,max=1), image[0,0,:,:].cpu().clamp_(min=-1,max=1)))
    print('Offset: ', torch.sum(y[0,0,:,:].cpu()-x[0,0,:,:].cpu()))
    
    #psnr = measure.compare_psnr(x_plt.data[0,0,:,:].cpu().numpy(), image_plt.data[0,0,:,:].cpu().numpy())
    
    print('SSIM_noisy: ', c_ssim(image.data[0,0,:,:].cpu().numpy(), y.data[0,0,:,:].cpu().numpy(), data_range=1, gaussian_weights=True))
    print('SSIM: ', c_ssim(x_plt.data[0,0,:,:].cpu().numpy(), image_plt.data[0,0,:,:].cpu().numpy(), data_range=1, gaussian_weights=True))
    print('Best_SSIM: ', best_ssim)
    
# =============================================================================
#     a=matlab.double(x_plt.data[0,0,:,:].cpu().numpy().tolist())
#     b=matlab.double(image_plt.data[0,0,:,:].cpu().numpy().tolist())
#     ssim_mat2 = mat_eng.ssim(a,b)
#     psnr_mat, ssim_mat = mat_eng.Cal_PSNRSSIM(mat_eng.im2uint8(a),mat_eng.im2uint8(b),0,0, nargout=2)
# 
#     print('SSIM_Matlab :', ssim_mat)
#     print('SSIM_Mat2: ', ssim_mat2)
# =============================================================================
    
    #save_image(x, 'IdentifieMode.png')
    writer_tensorboard.close()
    