# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:03:31 2018

@author: garwi
"""

from torchvision import datasets, transforms
from torch.utils import data
from Dataloader import img_dataset, preprocess


# Loader for CIFAR
def get_loader_cifar(directory='../../datasets/CIFAR10', batch_size=128, train=True, num_workers=0,
               pin_memory=True, gray_scale=True):
    
    # Scale images to [-1,1]
    rescaling = lambda x : (x - .5) * 2.
    
    if gray_scale:
        transform = transforms.Compose([
            transforms.Grayscale(), # Convert to grayscale
            transforms.ToTensor(),
            rescaling 
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            rescaling 
        ])
    
    # 32 x 32
    dataset = datasets.CIFAR10(directory,
                               train=train,
                               download=True,
                               transform=transform)
    shuffle = train
    loader = data.DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=False)
    return loader

# Loader for BSDS
img_dir="C:/Users/garwi/Desktop/Uni/Master/3_Semester/Masterthesis/Implementation/datasets/bsds_waterloo"

def get_loader_bsds(directory=img_dir, batch_size=16, train=True, num_workers=0,
               pin_memory=True, gray_scale=True, crop_size=[32,32]):
    
    # 32 x 32 crop
    if train:
        if gray_scale:
            transform = preprocess.scale_random_crop_gray(crop_size[0],crop_size[1])
        else:
            transform = preprocess.scale_random_crop(crop_size[0],crop_size[1])
    else:
        if gray_scale:
            transform = preprocess.central_crop_gray(crop_size[0],crop_size[1])
        else:
            transform = preprocess.scale_crop(crop_size[0],crop_size[1])
    
    dataset = img_dataset.PlainImageFolder(root=directory, transform=transform)
    
    shuffle = train
    loader = data.DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=False)
    return loader

# Loader for Denoising dataset
def get_loader_denoising(directory=img_dir, batch_size=16, train=True, num_workers=0,
               pin_memory=True, gray_scale=True, crop_size=[80,80]):
    
    # 32 x 32 crop
    if gray_scale:
        transform = preprocess.central_crop_gray(crop_size[0],crop_size[1])
    else:
        transform = preprocess.scale_crop(crop_size[0],crop_size[1])
# =============================================================================
#     transform = transforms.Compose([
#             #transforms.Grayscale(), # Convert to grayscale
#             transforms.ToTensor(),
#         ])
#     
# =============================================================================
    dataset = img_dataset.PlainImageFolder(root=directory, transform=transform)
    
    shuffle = train
    loader = data.DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=False)
    return loader

# =============================================================================
# # Test
# loader = get_loader_bsds(train=False, gray_scale=False)
# img_iter = iter(loader)
# img = next(img_iter)
# print(sum(1 for _ in img_iter))
# print("Max: ", img[0][6,:,:,:].max(), "Min: ", img[0][6,:,:,:].min(), "Size:", img[0].size())
# import matplotlib.pyplot as plt
# img[0] = img[0].permute(0, 2, 3, 1) #for RGB
# fig, axs = plt.subplots(4,1, figsize=(8,8)) 
# axs[0].imshow(img[0][0,:,:,:].numpy())
# axs[1].imshow(img[0][1,:,:,:].numpy())
# axs[2].imshow(img[0][2,:,:,:].numpy())
# axs[3].imshow(img[0][3,:,:,:].numpy())
# #plt.imshow(img[0][4,:,:,:].numpy())
# #plt.colorbar()
# =============================================================================

