# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:55:32 2018

@author: garwi
"""

import numpy
import time
from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = [1, 2, 3]
    plt.ion()
    for loop in range(1, 4):
        y = numpy.dot(loop, x)
        plt.close()
        plt.figure()
        plt.plot(x,y)
        plt.draw()
        time.sleep(2)