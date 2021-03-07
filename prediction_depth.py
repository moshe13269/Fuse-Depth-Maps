#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:06:36 2021

@author: moshelaufer
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from tifffile import imread
import numpy
import pandas as pd
from new_model import DepthFuse
from model_dense_v_concat import DUNet


TAG_CHAR = 'PIEH'
TAG_FLOAT = 202021.25

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth



path = '/home/moshelaufer/Desktop/DL-course/project2/data2project/'
category = ['City','Headbutt', 'Sitting', 'WallSlam','WuManchu']
in_dir = ['_L']
in_in_dir = ['Depth']    

torch.cuda.empty_cache()

model =  DUNet(1).float()
model.eval()
model.cuda()
checkpoint = torch.load("/home/moshelaufer/Desktop/DL-course/Data2project/running/model15_dense.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

from my_data_loader import DepthDataset
dataset_valid = DepthDataset('/home/moshelaufer/Desktop/DL-course/project2/data_test/',False,dim=1)
dataloader_valid = DataLoader(dataset_valid, batch_size=60, shuffle=False)
data = next(iter(dataloader_valid))
index =10  
arr_loss_styereo = []
arr_loss_out = []
data=data[index:index+1,:,:,:]
mono_image,stereo_image,y = data[:,0:1,:,:].float(), data[:,1:2,:,:].float(),data[:,2:,:,:].float()
output=model(stereo_image.cuda(),mono_image.cuda())

output=output.cpu().detach().numpy()
y = y.numpy()
ster = stereo_image.numpy()
mono = mono_image.numpy()
        
    
plt.subplot(1,4,1).title.set_text('Our model')
plt.imshow(output[0][0])
plt.axis('off')
plt.subplot(1,4,3).title.set_text('Ground Truth')
plt.imshow(y[0][0])
plt.axis('off')
plt.subplot(1,4,2).title.set_text('Stereo model')
plt.imshow(ster[0][0])
plt.axis('off')
plt.subplot(1,4,4).title.set_text('Mono model')
plt.imshow(mono[0][0])
plt.axis('off')

plt.subplots_adjust(hspace=.5)
plt.axis('off')

plt.show()





















