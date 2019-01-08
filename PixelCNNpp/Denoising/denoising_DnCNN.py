# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:53:00 2018

@author: garwi
"""

import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from PixelCNNpp.network import PixelCNN
import numpy as np
import torch.distributions.normal as normal
import matplotlib.pyplot as plt
import torch.optim as optim
from Denoising.denoising import Denoising
from Denoising.utils import add_noise, c_ssim, PSNR
from scipy import misc
from torchvision.utils import save_image, make_grid
import skimage.measure as measure
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    rescaling_inv = lambda x : .5 * x + .5
    rescaling = lambda x : (x - .5) * 2.
    
    writer_tensorboard =  SummaryWriter(comment='_denoising') 
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dncnn = torch.load("model_DnCNN_sigma=25/model_DnCNN_sigma=25_epoch_37.pth")["model"].to(device)
    dncnn.eval()
    
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=True, crop_size=[32,64]);
    test_loader = get_loader_denoising('../../../datasets/BSDS68', 1, train=False, num_workers=0, crop_size=None)
    
    # Iterate through dataset
    nr=3
    data_iter = iter(test_loader);
    for i in range(nr):
        image, label = next(data_iter);
    
    image = torch.tensor(image,dtype=torch.float32)
    
    sigma = torch.tensor(25.)
    mean = torch.tensor(0.)
    img_noisy = add_noise(image, sigma, mean)
    
    img_noisy = rescaling_inv(img_noisy).to(device)
    image = rescaling_inv(image)
    
    #with torch.no_grad():
    out = dncnn(img_noisy)
    
    result = img_noisy - out
    
    test = result[0,0].detach().cpu().numpy()
    
    psnr_denoised = PSNR(result[0,:,:,:].cpu().clamp(min=0,max=1), image[0,:,:,:].cpu(), peak=1)
    print('Denoised_Image_DnCNN: ', psnr_denoised)
    print('SSIM: ', c_ssim((result.data[0,0,:,:]).clamp(min=-1,max=1).detach().cpu().numpy(), (image.data[0,0,:,:]).cpu().numpy(), data_range=1, gaussian_weights=True))
    
    plt.imshow(result.detach().cpu().numpy()[0,0], cmap='gray')
    
    save_image(result, 'Denoised' + str(nr)+ '.png')
    
# =============================================================================
#     #Initialization of PixelCNN with DnCNN
#     init_pixelcnn = rescaling(result.cpu().detach())
#     y = rescaling(img_noisy)
#     image = rescaling(image)
#     
#     init_pixelcnn = torch.tensor(init_pixelcnn,dtype=torch.float32)
#     
#     # Load PixelCNN
#     net = PixelCNN(nr_resnet=5, nr_filters=160,
#             input_channels=1, nr_logistic_mix=10);
#     
#     net.load_state_dict(torch.load('../net_epoch_15000.pt'))
#  
#     net.to(device)
# 
#     net.eval()
#     
#     # Initialization of parameter to optimize
#     x = torch.tensor(init_pixelcnn.to(device),requires_grad=True);
#     
#     # Optimizing parameters
#     sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);
#     alpha = torch.tensor(0.025, dtype=torch.float32).to(device);
#     
#     params=[x]  
#     
#     optimizer = optim.Adam(params, lr=0.005) #0.05    
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
#     for i in range(100):
#  
# # =============================================================================
# #         def closure():
# #             optimizer.zero_grad();
# #             loss = logposterior(x, y, sigma, alpha, net);            
# #             loss.backward()#retain_graph=True);
# #             print(loss)
# #             return loss;
# # =============================================================================
#         optimizer.zero_grad();
#         loss = logposterior(x, y, sigma, alpha, net); #loss = logposterior(x, y, sigma, alpha, net);            
#         loss.backward();
#         #print('Gradient_sum: ',x.grad.sum())
#         psnr=PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu())
#         writer_tensorboard.add_scalar('Optimize/PSNR', psnr, i)
#         x_plt = (x + 1)/2
#         image_grid = make_grid(x_plt, normalize=True, scale_each=True)
#         writer_tensorboard.add_image('Image', image_grid, i)
#         print('Step ', i, ': ', loss);
#         print(psnr)
#         optimizer.step();
#         #print("Step: ", i)
#         scheduler.step();
#     
#     y = rescaling_inv(y)
#     x = rescaling_inv(x)
#     image = rescaling_inv(image)
#     fig, axs = plt.subplots(3,1, figsize=(8,8))    
#     im1=axs[0].imshow(image[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     fig.colorbar(im1, ax=axs[0])
#     im2=axs[1].imshow(x[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     fig.colorbar(im2, ax=axs[1])
#     im3=axs[2].imshow(result[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     fig.colorbar(im3, ax=axs[2])
#     
#     res = x[0,0,:,:].cpu().detach().numpy()
#     orig = image[0,0,:,:].cpu().detach().numpy()
#     offset = y[0,0,:,:].cpu().detach().numpy()-x[0,0,:,:].cpu().detach().numpy()
#     
#     
#     #plt.imshow(y_plt[0,0,:,:].clamp(min=0,max=1).cpu().detach().numpy(),cmap='gray')
#     #plt.colorbar()
#     psnr_denoised = c_psnr(result.data[0,0,:,:].cpu().numpy(), image.data[0,0,:,:].cpu().numpy())
#     print('Denoised_Image_DnCNN: ', psnr_denoised)
#     psnr_denoised = c_psnr(x.data[0,0,:,:].cpu().numpy(), image.data[0,0,:,:].cpu().numpy())
#     print('Denoised_Image_PixelCNN: ', psnr_denoised)
#     psnr_denoised = c_psnr(img_noisy.data[0,0,:,:].cpu().numpy(), image.data[0,0,:,:].cpu().numpy())
#     print('Image_Noisy: ', psnr_denoised)
# # =============================================================================
# #     print('Noisy_Image: ', PSNR(y[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
# #     print('Denoised_Image: ', PSNR(x[0,0,:,:].cpu(), image[0,0,:,:].cpu()))
# #     print('Offset: ', torch.sum(y[0,0,:,:].cpu()-x[0,0,:,:].cpu()))
# # =============================================================================
#     
#     #psnr = measure.compare_psnr(x_plt.data[0,0,:,:].cpu().numpy(), y_plt.data[0,0,:,:].cpu().numpy())
#     
#     #save_image(x, 'Denoised.png')
#     writer_tensorboard.close()
# =============================================================================
    
    
    