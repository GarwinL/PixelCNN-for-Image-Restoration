# -*- coding: utf-8 -*-
import sys
# Add sys path
sys.path.append('../../')


from Denoising.config import BaseConfig
import torch
import numpy as np
from Denoising.Denoising_patches.image_patch_denoising import patch_denoising
from PixelCNNpp.network import PixelCNN


if __name__ == '__main__':
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    
    # config for execution
    config = BaseConfig().initialize()
    
    # determine datasets
    datasets = ['Set12', 'BSDS68']
    #datasets = ['BSDS68']
    
    # Load PixelCNN
#    net = PixelCNN(nr_resnet=5, nr_filters=160, 
#            input_channels=1, nr_logistic_mix=10);
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=2, nr_filters=80,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
    
    net.load_state_dict(torch.load('../../Net/' + config.net_name))
    
    net.to(device)

    net.eval()
    
    psnr_results = []
    ssim_results = []
    
    for dataset in datasets:
        psnr, ssim = patch_denoising(dataset, config, net)
        psnr_results.append(psnr)
        ssim_results.append(ssim)
    