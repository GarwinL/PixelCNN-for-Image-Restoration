# -*- coding: utf-8 -*-

from Inpainting.utils import PSNR, add_noise, c_ssim, add_inpainting
from Dataloader.dataloader import get_loader_mask, get_loader_bsds, get_loader_denoising
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from SISR.utils import rescaling, c_psnr, c_ssim

test_loader = get_loader_denoising('../../datasets/Set12', 1, train=False, num_workers=0, crop_size=[180,180])

nr=8
data_iter = iter(test_loader);
for i in range(nr):
    image, label = next(data_iter);
  
    
y = rescaling(image, 0.5)

image_path = 'original_image' + str(nr) + '_.png'
#save_image(image, image_path)
image = image[0,0].numpy()
image = (((image - image.min()) / (image.max() - image.min())) * 255.9).astype(np.uint8)
img = Image.fromarray(image)
img.save(image_path)

image_path = 'downsampled_' + str(nr) + '.png'
y = y[0,0].numpy()
y = (((y - y.min()) / (y.max() - y.min())) * 255.9).astype(np.uint8)
img = Image.fromarray(y)
img.save(image_path)