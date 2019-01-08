# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:31:49 2018

@author: garwi
"""

import torch
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from skimage import color
from PixelCNNpp.network import PixelCNN
from PixelCNNpp.utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic, discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d, nl_logistic_mixture_continous
import numpy as np
from scipy import misc
from PixelCNNpp.utils import *
import matplotlib.pyplot as plt

def sample(net, device):
        """Sampling Images"""

        net.train(False)
        net.eval()
        rescaling_inv = lambda x : .5 * x + .5

        data = torch.zeros(4, 1, 32, 32).to(device) # gray<->color

        with torch.no_grad():
            for i in range(32):
                for j in range(32):
                    out   = net(data, sample=True)
                    sample = sample_from_discretized_mix_logistic_1d(out, 10)
                    data[:, :, i, j] = sample.data[:, :, i, j]
                    
            data = rescaling_inv(data)

        return data;

if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    #torch.backends.cudnn.deterministic = False
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    
    
    # Load datasetloader
    #train_loader = get_loader_cifar('../../datasets/CIFAR10', 8, train=True, num_workers=0);
    #test_loader = get_loader_cifar('../../datasets/CIFAR10', 8, train=False, num_workers=0);
    train_loader = get_loader_bsds('../../datasets/Waterloo_BSDS/Training', 1, train=False, num_workers=0, crop_size=[32,32])
    test_loader = get_loader_bsds('../../datasets/Waterloo_BSDS/Test', 1, train=False, num_workers=0, crop_size=[32,32])
#    train_loader = get_loader_bsds('../../datasets/pixelcnn_bsds/train', 4, train=False, num_workers=0, crop_size=[32,32])
#    test_loader = get_loader_bsds('../../datasets/pixelcnn_bsds/test', 4, train=False, num_workers=0, crop_size=[32,32])
  
  
    # Load PixelCNN
#    net = PixelCNN(nr_resnet=3, nr_filters=100, 
#            input_channels=1, nr_logistic_mix=10);
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=2, nr_filters=40,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
        
    net.load_state_dict(torch.load('../Net/net_epoch_5000_2resnet_40filter.pt'))
     
    net.to(device)
    
    net.eval()
    
    cnt=0
    sum_loss = 0.
    
    ##--Likelihood for test images--##
    for image, label in test_loader:    
        # Form [batch_size, 1, 32, 32]
        cnt += 1
        
        if cnt%100==0: print(cnt)

        image = torch.tensor(image,dtype=torch.float32, requires_grad=False).to(device)
#        image = (image + torch.Tensor(image.size()).uniform_ (-1./255.,1./255.)).to(device)
        
        with torch.no_grad():
            logit = net(image)
    
            loss = discretized_mix_logistic_loss_1d(image, logit)
#        loss = nl_logistic_mixture_continous(image, logit)
        
        sum_loss += float(loss)
        
        #if count>500: break;
    
    # dimension of image
# =============================================================================
#     dim=1
#     for i in range(len(image.size())):
#         dim *= image.size(i)
# =============================================================================
    
    dim = image.size(1)*image.size(2)*image.size(3)

    print('Log_Loss for Testimages: ',sum_loss)
    print('Bits/dim-Loss: ', sum_loss/(len(test_loader.dataset) * dim * torch.log(torch.tensor(2.).to(device))))
    
    
    ##--Likelihood for training images--##
    cnt=0
    sum_loss = 0.
    for image, label in train_loader:    
        # Form [batch_size, 1, 32, 32]
        cnt += 1
        
        if cnt%100==0: print(cnt)

        image = torch.tensor(image,dtype=torch.float32, requires_grad=False).to(device)
#        image = (image + torch.Tensor(image.size()).uniform_(-1./255.,1./255.)).to(device)
        
        logit = net(image)
    
        loss = discretized_mix_logistic_loss_1d(image, logit)
#        loss = nl_logistic_mixture_continous(image, logit)
        
        sum_loss += float(loss)
        
        #if count>500: break;
    
    # dimension of image
# =============================================================================
#     dim=1
#     for i in range(len(image.size())):
#         dim *= image.size(i)
# =============================================================================
    
    dim = image.size(1)*image.size(2)*image.size(3)

    print('Log_Loss for Trainingimages: ',sum_loss)
    print('Bits/dim-Loss: ', sum_loss/(len(train_loader.dataset) * dim * torch.log(torch.tensor(2.).to(device))))

# =============================================================================
#     sample = sample(net, device);
#     
#     #Plotting
#     fig, axs = plt.subplots(2,2, figsize=(8,8))    
#     axs[0,0].imshow(sample[0,0,:,:].cpu().detach().numpy(), cmap='gray')
#     axs[1,0].imshow(sample[1,0,:,:].cpu().detach().numpy(), cmap='gray')
#     axs[0,1].imshow(sample[2,0,:,:].cpu().detach().numpy(), cmap='gray')
#     axs[1,1].imshow(sample[3,0,:,:].cpu().detach().numpy(), cmap='gray')
# =============================================================================
