# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:02:08 2018

@author: garwi
"""
import sys
# Add sys path
sys.path.append('../')

import time
import numpy as np
import torch
from torch.optim import lr_scheduler
from torchvision.utils import save_image, make_grid
from PixelCNNpp.network_with_interpolation import PixelCNN
from PixelCNNpp.utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic, discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, config, train_loader, test_loader, device):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.test_data_size = len(self.test_loader.dataset)
        self.training_data_size = len(self.train_loader.dataset)
        self.is_train = self.config.isTrain
        self.nr_filters = self.config.nr_filters
        self.nr_resnet = self.config.nr_resnet
        self.nr_logistic_mix = self.config.nr_logistic_mix
        self.lr_decay = self.config.lr_decay**(0.1) if self.config.bsds else self.config.lr_decay
        self.data_size = (1, self.config.crop_size, self.config.crop_size) if self.config.grayscale else (3, self.config.crop_size, self.config.crop_size)
        self.rescaling_inv = lambda x : .5 * x + .5
        self.n_epochs = self.config.n_epochs*10 if self.config.bsds else self.config.n_epochs
        self.net_saveinterval = self.config.save_interval*10 if self.config.bsds else self.config.save_interval 
        self.sample_interval = self.config.sample_interval*10 if self.config.bsds else self.config.sample_interval
    def build(self):
        self.net = torch.nn.DataParallel(PixelCNN(nr_resnet=self.nr_resnet, nr_filters=self.nr_filters, 
            input_channels=self.data_size[0], nr_logistic_mix=self.nr_logistic_mix), device_ids=[0])
#        self.net = PixelCNN(nr_resnet=self.nr_resnet, nr_filters=self.nr_filters, 
#            input_channels=self.data_size[0], nr_logistic_mix=self.nr_logistic_mix).to(self.device)
        #self.net.load_state_dict(torch.load('net_epoch_1500_52.pt'))
        self.net.to(self.device)
    
        #print(self.net, '\n')     
        self.loss_fn = discretized_mix_logistic_loss_1d if self.config.grayscale else discretized_mix_logistic_loss
        self.sample_fn = sample_from_discretized_mix_logistic_1d if self.config.grayscale else sample_from_discretized_mix_logistic
        
        self.logfile = open(str(self.config.ckpt_dir.joinpath('Log_File.txt')),'w+')
        self.logfile_reset = open(str(self.config.ckpt_dir.joinpath('Log_File_Reset.txt')),'w+')
        self.writer_tensorboard =  SummaryWriter(str(self.config.ckpt_dir.joinpath('log_files/'))) 

        if self.config.mode == 'train':
            self.optimizer = self.config.optimizer(self.net.parameters(),self.config.lr)
            #self.optimizer.load_state_dict(torch.load("optimizer_epoch_1500.pt"))
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.lr_decay)

    def train(self):
        
        for epoch_i in range(1,self.n_epochs+1):

# =============================================================================
#             # For debugging
#             if epoch_i == 0:
#                 sample = self.sample(epoch_i)
#                 sample = self.rescaling_inv(sample)
#                 save_image(sample, image_path)
# =============================================================================

            self.net.train()
            self.batch_loss_history = []
            
            batch_i=0

            #for param_group in self.optimizer.param_groups: print(param_group['lr'])
            
            start = time.time()
            
            # Iterate through dataset
            for image, label in self.train_loader:             
                

                batch_i += 1
                # [batch_size, 3, 32, 32] - color image
                image = image.to(self.device)

                # [batch_size, 30, 32, 32]
                logit = self.net(image)

                target = torch.tensor(image.data)

                batch_loss = self.loss_fn(target, logit)
                batch_bpd = float(batch_loss.data)/(np.prod(image.size()) * np.log(2.))
                
                # Reset model parameters if huge jump in loss occurs and omit update
                if batch_bpd > 50:
                    self.net.load_state_dict(self.prev_state_dict)
                    self.logfile_reset.write('Epoch %d\r\n' %epoch_i)
                else:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=100000*self.config.batch_size)
                    self.optimizer.step()
                    
                    # Save previous model parameters
                    self.prev_state_dict = self.net.state_dict()
                
# =============================================================================
#                 ## Calculate total norm of gradient 
#                 total_norm=0
#                 parameters = list(filter(lambda p: p.grad is not None, self.net.parameters()))
#                 for p in parameters:
#                     param_norm = p.grad.data.norm(2)
#                     total_norm += param_norm.item() ** 2
#                     total_norm = total_norm ** (1. / 2)
#                 print(total_norm)
# =============================================================================
                  
                batch_loss = float(batch_loss.data)
                self.batch_loss_history.append(batch_loss)
               
            #decrease learning rate    
            self.scheduler.step()
            
            time_train = time.time() - start
            print('Traintime:',time_train)             
            epoch_loss = np.sum(self.batch_loss_history)/(self.training_data_size * np.prod(self.data_size) * np.log(2.))
            print('Epoch ', epoch_i, '-loss:', epoch_loss)
            self.logfile.write('Epoch %d:\r\n' %epoch_i)
            self.logfile.write('Epoch-loss: %f\r\n' %epoch_loss)
            self.writer_tensorboard.add_scalar('Train/Loss', epoch_loss, epoch_i)
            self.test(epoch_i)

            if epoch_i % self.sample_interval == 0: #Sampling
                image_path = str(self.config.ckpt_dir.joinpath(f'epoch_{epoch_i}.png'))                
                sample = self.sample(epoch_i)
                sample = self.rescaling_inv(sample)
                save_image(sample, image_path)
                image_grid = make_grid(sample, normalize=True, scale_each=True)
                self.writer_tensorboard.add_image('Image', image_grid, epoch_i)
            
            if epoch_i % self.net_saveinterval == 0: #Save net
                net_path = str(self.config.ckpt_dir.joinpath(f'net_epoch_{epoch_i}.pt'))
                torch.save(self.net.state_dict(), net_path)
                optimizer_path = str(self.config.ckpt_dir.joinpath(f'optimizer_epoch_{epoch_i}.pt'))
                torch.save(self.optimizer.state_dict(), optimizer_path)
                
        self.writer_tensorboard.export_scalars_to_json(str(self.config.ckpt_dir.joinpath('log_files/scalars.json')))
        self.writer_tensorboard.close()
        self.logfile_reset.close()

    def test(self, epoch_i):
        """Compute error on test set"""

        test_errors = []
        # cuda.synchronize()
        start = time.time()

        self.net.eval()

        for image, label in self.test_loader:

            # [batch_size, channel, height, width]
            image = image.to(self.device)

            # [batch_size, 30, 32, 32]]
            logit = self.net(image)
            
            target = torch.tensor(image.data)

            loss = self.loss_fn(target, logit)

            test_error = float(loss.data)
            test_errors.append(test_error)

        # cuda.synchronize()
        time_test = time.time() - start
        test_loss = np.sum(test_errors)/(self.test_data_size * np.prod(self.data_size) * np.log(2.))
        print('Testtime:',time_test)
        print('Test Loss:', test_loss)
        self.writer_tensorboard.add_scalar('Val/Loss', test_loss, epoch_i)
        self.logfile.write('Testtime: %f\r\n' %time_test)
        self.logfile.write('Test-loss: %f\r\n\r\n' %test_loss)

    def sample(self, epoch_i):
        """Sampling Images"""

        self.net.train(False)
        self.net.eval()

        data = torch.zeros(4, self.data_size[0], self.data_size[1], self.data_size[2], requires_grad=False).to(self.device) 

        with torch.no_grad():
            for i in range(self.data_size[1]):
                for j in range(self.data_size[2]):
                    out   = self.net(data, sample=True)
                    sample = self.sample_fn(out, 10)
                    data[:, :, i, j] = sample.data[:, :, i, j]

        return data;