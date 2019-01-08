# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:20:03 2018

@author: garwi
"""

import matplotlib.pyplot as plt
import numpy as np

Files = open("C:/Users/garwi/Desktop/Uni/Master/3_Semester/Masterthesis/Implementation/Results/PixelCNNpp/New_parameter/Cifar_color/2018-09-05_11-24-17/Log_File.txt","r")

txt=Files.read()

array_files = txt.split()

cnt=0
Train_loss=[]
Test_loss=[]

for i in array_files:
    cnt += 1
    
    if cnt == 4: Train_loss.append(float(i))
    if cnt == 8: 
        Test_loss.append(float(i))
        cnt=0

epoch = range(len(Train_loss))

fig, axs = plt.subplots(2,1)#, figsize=(8,16))   
axs[0].plot(epoch, Train_loss)#/(15*32*32*1*np.log(2.)))
axs[1].plot(epoch, Test_loss)#/(15*32*32*1*np.log(2.)))






