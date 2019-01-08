# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:19:25 2018

@author: garwi
"""

import torch
from dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from network import PixelCNN
import numpy as np
import torch.distributions.normal as normal
import matplotlib.pyplot as plt
import torch.optim as optim
from Posterior import logposterior
from utils import PSNR
from scipy import misc
from torchvision.utils import save_image, make_grid
import skimage.measure as measure
from tensorboardX import SummaryWriter

# Corrupt image with Gaussian Noise
# Intensities: [-1,1]
def add_noise(img, sigma, mean):   
    sigma = sigma*2/255 # 255->[-1,1]: sigma*2/255
    gauss = normal.Normal(mean,sigma)
    noise = gauss.sample(img.size())
    gauss = noise.reshape(img.size())
    img_noisy = torch.tensor(img + gauss);
    
    return img_noisy

def c_psnr(im1, im2):
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return measure.compare_psnr(im1, im2)

### Split image in patches given size of patches
# If image%patch_size = rest: Split the rest over the patches
# Return:
#       Array with patches
#       Array with left_shift or overlap respectively
###    
def patchify(img, patch_size):
    #Number of patches and calculation of shift
    patches_in_x = int(img.size(3)/patch_size[1])
    missing_in_x = img.size(3)%patch_size[1]   
    patches_in_y = int(img.size(2)/patch_size[0])
    missing_in_y = img.size(2)%patch_size[0]
    left_shift = 0
    up_shift = 0
    left_shift_res = 0
    up_shift_res = 0
    
    if missing_in_x: 
        left_shift_gen = int((patch_size[1]-missing_in_x)/patches_in_x)
        left_shift_res = (patch_size[1]-missing_in_x)%patches_in_x
        patches_in_x +=1
        
        
    if missing_in_y:
        up_shift_gen = int((patch_size[0]-missing_in_y)/patches_in_y)
        up_shift_res = (patch_size[1]-missing_in_x)%patches_in_y
        patches_in_y += 1
        
    #number of patches
    nr_patches = patches_in_x*patches_in_y
    
    #Tensor for patches
    patches = torch.zeros(nr_patches, img.size(0), img.size(1), patch_size[0], patch_size[1])
    overlap_y = [] # Overlap in y
    overlap_x = [] # Overlap in x
    upper_borders = []
    left_borders = []
    
    for i in range(patches_in_y):
        if i==0: upper_borders.append(0) 
        else: 
            up_shift = up_shift_gen
            if up_shift_res != 0:
                up_shift = up_shift_gen + 1
                up_shift_res -= 1
                
            overlap_y.append(up_shift)
                    
            upper_borders.append(upper_borders[i-1]+80-up_shift)
    
    #Fill patches and overlap array
    for i in range(patches_in_x):
        if i==0: left_borders.append(0)      
        else: 
            left_shift = left_shift_gen
            if left_shift_res != 0:
                left_shift = left_shift_gen + 1
                left_shift_res -= 1
                
            overlap_x.append(left_shift)
                
            left_borders.append(left_borders[i-1]+80-left_shift)
            
        for j,y in enumerate(upper_borders):
            patches[i*len(upper_borders)+j] = img[:, :, y:y+80, left_borders[i]:left_borders[i]+80]
            
    return patches, upper_borders, left_borders;

###Reconstruct the original image out of the patches -> Averaging over overlapped region
# Input:
#       patches
#       borders (in x,y)
#       original image-size
###      
def unpatchify(patches, upper_borders, left_borders, img_size):
    img = torch.zeros(img_size)
    
    cnt = 0
    for j in left_borders:
        for i in upper_borders:
            for x in range(80):
                for y in range(80):
                    if torch.sum(torch.eq(img[:,:,i+y,j+x], torch.zeros(img_size[0],img_size[1])))==0:
                        img[:,:,i+y,j+x] = (img[:,:,i+y,j+x]+patches[cnt,:,:,y,x])/2
                    else: img[:,:,i+y,j+x] = patches[cnt,:,:,y,x]
                    
            #img[:,:,i:i+80,j:j+80] = patches[cnt] #easy way
            cnt += 1
        
    return img
    


if __name__ == '__main__':
    
    writer_tensorboard =  SummaryWriter(comment='_denoising')     
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../../datasets/pixelcnn_bsds/Test_Denoising', 1, train=False, num_workers=0)
    
    # Iterate through dataset
    data_iter = iter(test_loader);
    image, label = next(data_iter);
    image, label = next(data_iter);
    image, label = next(data_iter);
    image, label = next(data_iter);  
    #image, label = next(data_iter);
    #image, label = next(data_iter);
    
    image = torch.tensor(image,dtype=torch.float32)
    
    img_size = image.size()
    
    print(image.size())
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load and init PixelCNN
# =============================================================================
#     net = PixelCNN(nr_resnet=5, nr_filters=160,
#             input_channels=1, nr_logistic_mix=10);
#     net.load_state_dict(torch.load('../net_epoch_10000.pt'))
#     net.to(device)
#     net.eval()
# =============================================================================
  
    #Add noise to image
    sigma = torch.tensor(25.)
    mean = torch.tensor(0.)
    y = add_noise(image, sigma, mean)
    
    # Initialization of parameter to optimize
    x = torch.tensor(y.to(device),requires_grad=True);
    
    # Optimizing parameters
    sigma = torch.tensor(sigma*2/255, dtype=torch.float32).to(device);
    alpha = torch.tensor(0.025, dtype=torch.float32).to(device); #contionous: 0.025 discrete: 0.45
    
    y = y.to(device)
    
    # Size of patches
    patch_size = [80,80]
    patches, upper_borders, left_borders = patchify(image, patch_size)
    
    print(upper_borders)
    print(left_borders)
    
    img = unpatchify(patches, upper_borders, left_borders, img_size)
    
# =============================================================================
#     #Plotting
#     fig, axs = plt.subplots(4,1, figsize=(8,8))  
#     cnt=0
#     for i in range(0,4):        
#         patch = patches[i].permute(0, 2, 3, 1)
#         axs[cnt].imshow(patch[0,:,:,:].cpu().detach().numpy())
#         cnt +=1
#         #fig.colorbar(im, ax=axs[i])
# =============================================================================
    print(img.size())
    patch = img.permute(0, 2, 3, 1)
    plt.imshow(patch[0,:,:,:].cpu().detach().numpy())
    
    
    
    
   