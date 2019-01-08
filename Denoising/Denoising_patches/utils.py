# -*- coding: utf-8 -*-

import torch

### Split image in patches given size of patches
# If image%patch_size = rest: Split the rest over the patches
# Return:
#       Array with patches
#       Array with left_shift or overlap respectively
###    
def patchify(img, patch_size):
    
    #Number of patches and calculation of shift
    patches_in_x = int(img.size(3)/patch_size[1])
    missing_in_x = img.size(3)%patch_size[1]   
    patches_in_y = int(img.size(2)/patch_size[0])
    missing_in_y = img.size(2)%patch_size[0]
    left_shift = 0
    up_shift = 0
    left_shift_res = 0
    up_shift_res = 0
    up_shift_gen = 0
    left_shift_gen = 0
    
    if patches_in_x == 0:
        patches_in_x = 1
        x_size = img.size(3)
    else:   
        x_size = patch_size[1]
        if missing_in_x: 
            left_shift_gen = int((patch_size[1]-missing_in_x)/patches_in_x)
            left_shift_res = (patch_size[1]-missing_in_x)%patches_in_x
            patches_in_x +=1
            
    if patches_in_y == 0:
        patches_in_y = 1
        y_size = img.size(2)
        
    else:   
        y_size = patch_size[0]           
        if missing_in_y:
            up_shift_gen = int((patch_size[0]-missing_in_y)/patches_in_y)
            up_shift_res = (patch_size[1]-missing_in_x)%patches_in_y
            patches_in_y += 1
        
    #number of patches
    nr_patches = patches_in_x*patches_in_y
    
    #Tensor for patches
    patches = torch.zeros(nr_patches, img.size(0), img.size(1), y_size, x_size)
    overlap_y = [] # Overlap in y
    overlap_x = [] # Overlap in x
    upper_borders = []
    left_borders = []
    
    for i in range(patches_in_y):
        if i==0: upper_borders.append(0) 
        else: 
            up_shift = up_shift_gen
            if up_shift_res != 0:
                up_shift = up_shift_gen + 1
                up_shift_res -= 1
                
            overlap_y.append(up_shift)
                    
            upper_borders.append(upper_borders[i-1]+patch_size[0]-up_shift)
    
    #Fill patches and overlap array
    for i in range(patches_in_x):
        if i==0: left_borders.append(0)      
        else: 
            left_shift = left_shift_gen
            if left_shift_res != 0:
                left_shift = left_shift_gen + 1
                left_shift_res -= 1
                
            overlap_x.append(left_shift)
                
            left_borders.append(left_borders[i-1]+patch_size[1]-left_shift)
            
        for j,y in enumerate(upper_borders):
            patches[i*len(upper_borders)+j] = img[:, :, y:y+patch_size[0], left_borders[i]:left_borders[i]+patch_size[1]]
            
    return patches, upper_borders, left_borders;

###Reconstruct the original image out of the patches -> Averaging over overlapped region
# Input:
#       patches
#       borders (in x,y)
#       original image-size
###      
def aggregate(patches, upper_borders, left_borders, img_size):
    img = torch.zeros(img_size)
    patch_size = patches.size()
    
    cnt = 0
    for j in left_borders:
        for i in upper_borders:
            for x in range(patch_size[4]):
                for y in range(patch_size[3]):
                    if torch.sum(torch.eq(img[:,:,i+y,j+x], torch.zeros(img_size[0],img_size[1])))==0:
                        img[:,:,i+y,j+x] = (img[:,:,i+y,j+x]+patches[cnt,:,:,y,x])/2
                    else: img[:,:,i+y,j+x] = patches[cnt,:,:,y,x]
                    
            #img[:,:,i:i+80,j:j+80] = patches[cnt] #easy way
            cnt += 1
        
    return img