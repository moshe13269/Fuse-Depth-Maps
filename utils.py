#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:57:24 2021

@author: moshelaufer
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
import torch.nn.functional as F
from my_data_loader import DepthDataset
from torch.utils.data import TensorDataset, DataLoader
from model_dense_v_concat import DUNet
from model_att_v_concat3 import AUNet # 13 weight


TAG_CHAR = 'PIEH'
TAG_FLOAT = 202021.25



                             
PATH = '/home/moshelaufer/Desktop/DL-course/project2/data2project/'    

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

def mape_calc(output,y):
    mape_batch=0.0
    y = torch.squeeze(y)
    output = torch.squeeze(output)
    for i in range(output.size()[0]):
        gt = y.cpu().detach().numpy()[i]
        predicted = output.cpu().detach().numpy()[i]
        gt = np.squeeze(gt)
        predicted = np.squeeze(predicted)
        mape_batch = mape_batch + mape(gt, predicted)
    return mape_batch/output.size()[0]

def validation_calc(model,data_loder,device):
    with torch.no_grad():
        MSE_loss = nn.MSELoss()
        L1_loss = nn.L1Loss()
        loss_l1=0.0
        loss_l2=0.0
        mape_valid=0.0
        loss_valid=0.0
        model.eval()
        sample = len(data_loder)
        for batch_ndx, data in enumerate(data_loder):
          
            mono_image,stereo_image,y = data[:,0:1,:,:].float(), data[:,1:2,:,:].float(),data[:,2:,:,:].float()
            stereo_image = stereo_image.to(device)
            mono_image  = mono_image.to(device)
            y = y.to(device)
            output = model(stereo_image,mono_image)
            
            loss_L1 = L1_loss(output,y)
            loss_L2 = MSE_loss(output,y)
            loss = F.smooth_l1_loss(output,y)
            
            loss_l1 += loss_L1.item()
            loss_l2 += loss_L2.item()
            loss_valid += loss.item()
            
            mape_valid = mape_valid + mape_calc(output,y)
        model.train()
    return model, loss_valid/sample , mape_valid/sample , loss_l2/sample, loss_l1/sample


def check_loss_validation(loss_arr):
    if loss_arr[len(loss_arr)-1]<=min(loss_arr[len(loss_arr)-40:len(loss_arr)-2]):
        return True
    return False
 
def check_results(dataset_path='/home/moshelaufer/Desktop/DL-course/project2/data_test/'):
    dataset_valid = DepthDataset(dataset_path,False,dim=1)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False)
    
    device = torch.device('cuda:2')
    torch.cuda.empty_cache()
    
    # model = AUNet(1).float() # model13
    model = DUNet(1).float() # model15_dense
    model.to(device)
    checkpoint = torch.load("/home/moshelaufer/Desktop/DL-course/Data2project/running/model15_dense.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    _, loss_valid, mape_valid,loss_l2 , loss_l1 = validation_calc(model,dataloader_valid,device)
    print("loss_valid = {},  mape_valid={}, loss_l2={}, loss_l1={}".format(loss_valid, mape_valid,loss_l2 , loss_l1))

check_results()    

def check_loss_stereo_mono(state,dataset_path='/home/moshelaufer/Desktop/DL-course/project2/data_test/' ):
    device = torch.device('cuda:2')

    dataset_valid = DepthDataset(dataset_path,False,dim=1)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False) 
    MSE_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()
    loss_l1=0.0
    loss_l2=0.0
    mape_valid=0.0
    loss_valid=0.0
    sample = len(dataloader_valid)
    for batch_ndx, data in enumerate(dataloader_valid):          
            mono_image,stereo_image,y = data[:,0:1,:,:].float(), data[:,1:2,:,:].float(),data[:,2:,:,:].float()
            stereo_image = stereo_image.to(device)
            mono_image  = mono_image.to(device)
            y = y.to(device)
            if state=='mono':
                loss_L1 = L1_loss(mono_image,y)
                loss_L2 = MSE_loss(mono_image,y)
                loss = F.smooth_l1_loss(mono_image,y)
                
                loss_l1 += loss_L1.item()
                loss_l2 += loss_L2.item()
                loss_valid += loss.item()
                
                mape_valid = mape_valid + mape_calc(mono_image,y)
            else:
                loss_L1 = L1_loss(stereo_image,y)
                loss_L2 = MSE_loss(stereo_image,y)
                loss = F.smooth_l1_loss(stereo_image,y)
                
                loss_l1 += loss_L1.item()
                loss_l2 += loss_L2.item()
                loss_valid += loss.item()
                
                mape_valid = mape_valid + mape_calc(mono_image,y)
    return loss_valid/sample , mape_valid/sample , loss_l2/sample, loss_l1/sample           
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    