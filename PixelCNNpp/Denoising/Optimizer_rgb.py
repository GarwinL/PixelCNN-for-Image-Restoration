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
from PixelCNNpp.Denoising.Posterior_rgb import logposterior
from Denoising.utils import PSNR
from torchvision.utils import save_image

# Corrupt image with Gaussian Noise
# Intensities: [-1,1]
def add_noise(img, sigma, mean):   
    sigma = sigma*2/255 # 255->[-1,1]: sigma*2/255
    gauss = normal.Normal(mean,sigma)
    noise = gauss.sample(img.size())
    gauss = noise.reshape(img.size())
    img_noisy = torch.tensor(img + gauss);
    
    return img_noisy

if __name__ == '__main__':
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0, gray_scale=False);
    test_loader = get_loader_denoising('../../../datasets/pixelcnn_bsds/Test_Denoising', 1, train=False, gray_scale=False)
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    image, label = next(data_iter);
    image, label = next(data_iter);
    #image, label = next(data_iter); 
    #image, label = next(data_iter); 
    #image, label = next(data_iter); 
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
#    net = PixelCNN(nr_resnet=5, nr_filters=160, 
#            input_channels=3, nr_logistic_mix=10);
                   
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=5, nr_filters=160,
            input_channels=3, nr_logistic_mix=10), device_ids=[0]);
    
    net.load_state_dict(torch.load('../net_epoch_300.pt'))
 
    net.to(device)

    net.eval()

    image = torch.tensor(image,dtype=torch.float32)
    
    sigma = torch.tensor(25.)
    mean = torch.tensor(0.)
    y = add_noise(image, sigma, mean)
    
    # Initialization of parameter to optimize
    x = torch.tensor(y.to(device),requires_grad=True);
    
    # Optimizing parameters
    sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);
    alpha = torch.tensor(0.5, dtype=torch.float32).to(device); #0.5
    
    y = y.to(device)
    
    params=[x]
    
    conv_cnt = 0
    
    optimizer = optim.Adam(params, lr=0.1) # Adam (0,1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    for i in range(100):
 
# =============================================================================
#         def closure():
#             optimizer.zero_grad();
#             loss = logposterior(x, y, sigma, alpha, net);            
#             loss.backward()#retain_graph=True);
#             print(loss)
#             return loss;
# =============================================================================
        optimizer.zero_grad();
        loss = logposterior(x, y, sigma, alpha, net);            
        loss.backward()#retain_graph=True);
        #print('Gradient_sum: ',x.grad.sum())
        print('Step ',i, ': ', loss);
        optimizer.step();
        scheduler.step();
        psnr=PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu())
        print('PSNR: ', psnr)
    
    
    #Plotting
    y_plt = (y.permute(0, 2, 3, 1) + 1)/2
    x_plt = (x.permute(0, 2, 3, 1) + 1)/2
    image_plt = (image.permute(0, 2, 3, 1) + 1)/2
    fig, axs = plt.subplots(3,1, figsize=(8,8))    
    im1=axs[0].imshow(y_plt[0,:,:,:].cpu().detach().numpy())
    fig.colorbar(im1, ax=axs[0])
    im2=axs[1].imshow(x_plt[0,:,:,:].cpu().detach().numpy())
    fig.colorbar(im2, ax=axs[1])
    im3=axs[2].imshow(image_plt[0,:,:,:].cpu().detach().numpy())
    fig.colorbar(im3, ax=axs[2])
    
    #res = x[0,0,:,:].cpu().detach().numpy()
    #orig = image[0,0,:,:].cpu().detach().numpy()

    #plt.imshow(y_plt[0,:,:,:].cpu().detach().numpy())
    #plt.colorbar()
    print('Noisy_Image: ', PSNR(y[0,:,:,:].cpu(), image[0,:,:,:].cpu()))
    print('Denoised_Image: ', PSNR(x[0,:,:,:].cpu(), image[0,:,:,:].cpu()))
    
    #save_image(x, 'Denoised.png')
    