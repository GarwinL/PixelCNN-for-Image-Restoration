# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:58:18 2018

@author: garwi
"""
import os

for i in range(1,475):
    i = i*10
    file = f'{i:05}'  + '.bmp'
    os.rename("../../datasets/exploration_database_and_code/pristine_images/" + file, "../../datasets/exploration_database_and_code/Testimages/" + file)