# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:50:42 2018

@author: garwi
"""

import torch
from torch.autograd import Variable
from PixelCNNpp.network import PixelCNN
from Dataloader.dataloader import get_loader_denoising
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PixelCNNpp.utils import sample_from_discretized_mix_logistic_1d


#Check for GPU 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_from_data(model, data):
    model.train(False)
    #data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])#, requires_grad=True)
    data = data.to(device)
    data = Variable(data, volatile=True)
    out   = model(data, sample=True)
    out_sample = sample_from_discretized_mix_logistic_1d(out, 10)
    data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

def sample(net, batch_size=1, sample_size= [32, 32]):
    """Sampling Images"""      
    net.train(False)
    net.eval() 

    data = torch.zeros(batch_size, 1, sample_size[0] , sample_size[1], requires_grad=False).to(device) 
    with torch.no_grad():            
        for i in range(sample_size[0]):                
            for j in range(sample_size[1]):              
                out   = net(data, sample=True)                    
                sample = sample_from_discretized_mix_logistic_1d(out, 10)                    
                data[:, :, i, j] = sample.data[:, :, i, j]
        
        return data;

if __name__ == '__main__':
    
    cnt = 0
    batch_size = 4
    sample_size = [52, 52]
    
# =============================================================================
#     # Load datasetloader
#     #test_loader = get_loader_cifar('../../../datasets/CIFAR10', 1, train=False, num_workers=0);
#     #test_loader = get_loader_bsds('../../../datasets/BSDS/pixelcnn_data/train', 1, train=False, crop_size=[32,32]);
#     test_loader = get_loader_denoising('../../datasets/BSDS68', 1, train=False, num_workers=0)
#     
#     # Iterate through dataset
#     data_iter = iter(test_loader);
#     image, label = next(data_iter);
#     image, label = next(data_iter);
#     image, label = next(data_iter);
#     #image, label = next(data_iter);  
#     #image, label = next(data_iter);
#     #image, label = next(data_iter);
# =============================================================================
    
    rescaling_inv = lambda x : .5 * x + .5
    
    #Device for computation (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load PixelCNN
    net = torch.nn.DataParallel(PixelCNN(nr_resnet=2, nr_filters=80,
            input_channels=1, nr_logistic_mix=10), device_ids=[0]);
   
#    net = PixelCNN(nr_resnet=3, nr_filters=100,
#            input_channels=1, nr_logistic_mix=10);
    
    # continous - discrete - continous_new
    net.load_state_dict(torch.load('../Net/net_epoch_3000.pt'))
 
    net.to(device)    
    
    print('sampling...')
    
    for i in range(int(40/batch_size)):
        print('Number ', i)
        sample_t = sample(net, batch_size, sample_size)
        sample_t = rescaling_inv(sample_t)
        
        directory = 'PixelCNN_samples/sample_'
        
        #Save samples
        for single_sample in sample_t:
            image_path = directory + str(cnt) + '.png'
            save_image(single_sample, image_path)
            cnt += 1