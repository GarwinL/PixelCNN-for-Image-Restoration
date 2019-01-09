# -*- coding: utf-8 -*-
import sys
# Add sys path
sys.path.append('../../')

import torch
import numpy as np
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds, get_loader_denoising
from PixelCNNpp.network import PixelCNN
from Denoising.config import BaseConfig
from Denoising.utils import PSNR, img_rot90, c_ssim, add_noise
from Denoising.Learn_params.optimizer import optimizeMAP
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

def quadratic_spline_roots(spl):
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)


if __name__ == '__main__':
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    config = BaseConfig().initialize()
    
    torch.backends.cudnn.deterministic = True
    
    # Load datasetloader
    #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
    #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
    test_loader = get_loader_denoising('../../../datasets/Parameterevaluation', 1, train=False, num_workers=0, crop_size=[config.crop_size, config.crop_size])    
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=2, nr_filters=80,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
   
#    net = PixelCNN(nr_resnet=3, nr_filters=100,
#            input_channels=1, nr_logistic_mix=10);
    
    net.load_state_dict(torch.load('../../Net/' + config.net_name))
 
    net.to(device)

    net.eval()
    
    logfile = open(config.directory.joinpath('sigma =' + str(config.sigma) + '.txt'),'w+')
    
    data_list = []
    
    # Iterate through dataset
    for image, label in test_loader:
        image = torch.tensor(image,dtype=torch.float32)
    
        sigma = torch.tensor(config.sigma)
        mean = torch.tensor(0.)
        y = add_noise(image, sigma, mean)
        
        data_list.append([image, y])
    
    
    ###############################################
    ###-- Train weight of prior from 0.3 to 1 --###
    ###############################################
    
    # Predefined learning rate
    lr = 0.08
    
    
    psnr_list = []
    ssim_list = []
    step_list = []
    
    scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Execute denoising on test set with different scales
    for scale in scales:
    
        psnr, ssim, step = optimizeMAP(data_list, scale, lr, net, config)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        step_list.append(step)
    
    # Interpolate
    fct_ssim = InterpolatedUnivariateSpline(scales, ssim_list, k=3)
    fct_psnr = InterpolatedUnivariateSpline(scales, psnr_list, k=3)
    
    
    # Calculate Maximum
    roots = quadratic_spline_roots(fct_psnr.derivative())
    roots = np.append(roots, (scales[0], scales[-1]))
    y_ssim = fct_ssim(roots)
    y_psnr = fct_psnr(roots)
    
    # Optimal value
    scale_opti = roots[np.argmax(y_psnr)]
    
    print('Optimal: ', scale_opti)
    
    #-- Refine scaling --#
    scales_refined = [scale_opti-0.06, scale_opti-0.04, scale_opti-0.02, scale_opti, scale_opti+0.02, scale_opti+0.04, scale_opti+0.06]
    
    psnr_list = []
    ssim_list = []
    step_list = []
    
    # Execute denoising on test set with different scales
    for scale in scales_refined:
        psnr, ssim, step = optimizeMAP(data_list, scale, lr, net, config)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        step_list.append(step)
     
    # Interpolate                                    
    fct_ssim = InterpolatedUnivariateSpline(scales_refined, ssim_list, k=3)
    fct_psnr = InterpolatedUnivariateSpline(scales_refined, psnr_list, k=3)
    
    # Calculate Maximum
    roots_refined = quadratic_spline_roots(fct_psnr.derivative())
    roots_refined = np.append(roots_refined, (scales_refined[0], scales_refined[-1]))
    y_ssim = fct_ssim(roots_refined)
    y_psnr = fct_psnr(roots_refined)
    
    # Optimal value
    scale_opti = roots_refined[np.argmax(y_psnr)]
    
    print('Optimal Parameter:', scale_opti)
    
    logfile.write('Optimal weight: %f\r\n' %scale_opti)
    
    ##########################
    ### -- Learning rate --###
    ##########################
    
    psnr_list = []
    ssim_list = []
    step_list = []
    
    lrs = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    
    # Execute denoising on test set with different scales
    for lr in lrs:
    
        psnr, ssim, step = optimizeMAP(data_list, scale_opti, lr, net, config)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        step_list.append(step)
    
    # Interpolate
    fct_ssim = InterpolatedUnivariateSpline(lrs, ssim_list, k=3)
    fct_psnr = InterpolatedUnivariateSpline(lrs, psnr_list, k=3)
    
    
    # Calculate Maximum
    roots = quadratic_spline_roots(fct_psnr.derivative())
    roots = np.append(roots, (lrs[0], lrs[-1]))
    y_ssim = fct_ssim(roots)
    y_psnr = fct_psnr(roots)
    
    # Optimal value
    lr_opti = roots[np.argmax(y_psnr)]
    
    print('Optimal: ', lr_opti)
    
    #-- Refine scaling --#
    lrs_refined = [lr_opti-0.015, lr_opti-0.01, lr_opti-0.005, lr_opti, lr_opti+0.005, lr_opti+0.010, lr_opti+0.015]
    
    psnr_list = []
    ssim_list = []
    step_list = []
    
    # Execute denoising on test set with different scales
    for lr in lrs_refined:
        psnr, ssim, step = optimizeMAP(data_list, scale_opti, lr, net, config)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        step_list.append(step)
     
    # Interpolate                                    
    fct_ssim = InterpolatedUnivariateSpline(lrs_refined, ssim_list, k=3)
    fct_psnr = InterpolatedUnivariateSpline(lrs_refined, psnr_list, k=3)
    
    # Calculate Maximum
    roots_refined = quadratic_spline_roots(fct_psnr.derivative())
    roots_refined = np.append(roots_refined, (lrs_refined[0], lrs_refined[-1]))
    y_ssim = fct_ssim(roots_refined)
    y_psnr = fct_psnr(roots_refined)
    
    # Optimal value
    lr_opti = roots_refined[np.argmax(y_psnr)]
    
    print('Optimal Parameter:', lr_opti)
    
    logfile.write('Optimal learning rate: %f\r\n' %lr_opti)
    
    psnr, ssim, step = optimizeMAP(data_list, scale_opti, lr_opti, net, config)
    
    logfile.write('Optimal gradient steps: %f\r\n' %step)
    
    logfile.close()
    