# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:47:57 2018

@author: garwi
"""

import argparse
from pathlib import Path, PurePath
import pprint
import torch
from datetime import datetime
#from New_LBFGS.LBFGS import LBFGS

def get_optimizer(optimizer_name='Adam'):
    """Get optimizer by name"""
    # optimizer_name = optimizer_name.capitalize()
    return getattr(torch.optim, optimizer_name)


class BaseConfig(object):
    def __init__(self):
        """Base Configuration Class"""

        self.parse_base()

    def parse_base(self):
        """Base configurations for all models"""

        self.parser = argparse.ArgumentParser()

        #================ Train ==============#
        self.parser.add_argument('--n_epochs', type=int, default=300)
        self.parser.add_argument('--optimizer', type=str, default='Adam') #Adam, LBFGS_linesearch, SGD
        self.parser.add_argument('--lr_decay', type=float, default=0.992) #Adam:0.999
        self.parser.add_argument('--lr', type=float, default=0.3)  #Adam:0.001
        self.parser.add_argument('--net_name', type=str, default='net_epoch_3000_2resnet_80filter.pt') 
        self.parser.add_argument('--control_epochs', type=int, default=5)
        self.parser.add_argument('--continuous_logistic', type=float, default=False)
        self.parser.add_argument('--crop_size', type=int, default=256)
        
        

    def parse(self):
        """Update configuration with extra arguments (To be inherited)"""
        pass

    def initialize(self, parse=True, **optional_kwargs):
        """Set kwargs as class attributes with setattr"""

        # Update parser
        self.parse()

        # Parse arguments
        if parse:
            kwargs = self.parser.parse_args()
        else:
            kwargs = self.parser.parse_known_args()[0]

        # namedtuple => dictionary
        kwargs = vars(kwargs)
        kwargs.update(optional_kwargs)
        
        result_dir = Path('runs/')
        
        if not result_dir.exists():
            result_dir.mkdir()
            
        time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        result_dir = result_dir.joinpath(time_now)
        
        if not result_dir.exists():
            result_dir.mkdir()
            
        self.directory = result_dir
        
        self.linesearch = False

        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    if value == 'LBFGS_linesearch':
                        value = None #LBFGS
                        self.linesearch = True
                    else:    
                        value = get_optimizer(value)
                setattr(self, key, value)

            self.config_string = ''
            for k, v in sorted(self.__dict__.items()):
                if str(k) != 'config_string':
                    self.config_string += '%s: %s\n\r' % (str(k), str(v))     
            
        config_file_path = self.directory.joinpath('config.txt')
        with open(config_file_path, 'w') as f:
            f.write('-------- Configurations ----------\n')
            f.write(self.config_string)
            f.write('-------- End -------\n')

        return self

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True):
    """Get configuration class in single step"""
    return BaseConfig().initialize(parse=parse)


if __name__ == '__main__':
    config = get_config()
    #import ipdb
    #ipdb.set_trace()