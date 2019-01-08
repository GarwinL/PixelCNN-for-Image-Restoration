# -*- coding: utf-8 -*-
from Denoising.config import BaseConfig
import torch
import numpy as np
from Denoising.Parameter_evaluation.denoising_parameter import denoise_parameter
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
    params = [1]
    
    # Load PixelCNN
#    net = PixelCNN(nr_resnet=5, nr_filters=160, 
#            input_channels=1, nr_logistic_mix=10);
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=3, nr_filters=100,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
    
    net.load_state_dict(torch.load('../../Net/' + config.net_name))
    
    net.to(device)

    net.eval()
    
    psnr_results = []
    ssim_results = []
    step_results = []
    
    for par in params:
        psnr, ssim, step = denoise_parameter(par, config, net)
        psnr_results.append(psnr)
        ssim_results.append(ssim)
        step_results.append(step_results)
        print('Parameter= %f: PSNR=%f - SSIM=%f - Best_step=%f\r\n' %(par, psnr, ssim, step))
    