# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:58:38 2018

@author: garwi
"""

from PixelCNNpp.config import BaseConfig as Config
from Dataloader.dataloader import get_loader_cifar, get_loader_bsds
from PixelCNNpp.trainer import Trainer
import torch
import numpy as np


def main(config, device):
    
    # reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    
    num_workers=0
    
    if config.bsds: # BSDS -> CIFAR - factor 100
        # Loader for BSDS
        train_dir="../../datasets/Waterloo_BSDS/Training"
        test_dir = "../../datasets/Waterloo_BSDS/Test"
        
        train_loader = get_loader_bsds(
            train_dir,
            config.batch_size,
            train=True,
            num_workers=num_workers,
            gray_scale=config.grayscale,
            crop_size=[config.crop_size,config.crop_size])
        test_loader = get_loader_bsds(
            test_dir,
            4,
            train=False,
            num_workers=num_workers,
            gray_scale=config.grayscale,
            crop_size=[config.crop_size,config.crop_size])
    
    if not config.bsds:
        # Loader for CIFAR10
        train_loader = get_loader_cifar(
            config.dataset_dir,
            config.batch_size,
            train=True,
            num_workers=num_workers,
            gray_scale=config.grayscale,
            crop_size=[config.crop_size,config.crop_size])
        test_loader = get_loader_cifar(
            config.dataset_dir,
            4,
            train=False,
            num_workers=num_workers,
            gray_scale=config.grayscale,
            crop_size=[config.crop_size,config.crop_size])

    
    solver = Trainer(config, train_loader=train_loader, test_loader=test_loader, device = device)
    print(config)
    # print(f'\nTotal data size: {solver.total_data_size}\n')

    solver.build()
    solver.train()
    
    torch.save(solver.net.state_dict(),'results/net_result.pt')


if __name__ == '__main__':
    #Check for GPU 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get Configuration
    config = Config().initialize()

    main(config, device)