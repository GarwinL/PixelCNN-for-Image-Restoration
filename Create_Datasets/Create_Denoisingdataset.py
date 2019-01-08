# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:33:16 2018

@author: garwi
"""

Files = open("../../datasets/pixelcnn_bsds/Test_Denoising/foe_test.txt","r")

txt=Files.read()

array_files = txt.split()

import os

for file in array_files:
    os.rename("../../datasets/pixelcnn_bsds/gesamt/" + file, "../../datasets/pixelcnn_bsds/Test_Denoising/")

