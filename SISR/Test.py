# -*- coding: utf-8 -*-

import torch
from SISR.utils import rescaling, psnr

Test = torch.arange(36).reshape(1,1,6,6)

Test_down = rescaling(Test, 0.5)

Test_up = rescaling(Test_down, 2)