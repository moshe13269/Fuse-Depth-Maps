#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:40:15 2021

@author: moshelaufer
"""
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy
from new_model import DepthFuse
from model_att_v_concat import AUNet
from my_data_loader import DepthDataset
from utils import validation_calc, check_loss_validation
from model_dense_v_concat import DUNet

def main():
    device = torch.device('cuda:1')
    
    dataset_valid = DepthDataset('/home/moshelaufer/Desktop/DL-course/project2/data_valid/',False,dim=1)
    dataloader_valid = DataLoader(dataset_valid, batch_size=2, shuffle=False)
    
    
    torch.cuda.empty_cache()
    file = open("/home/moshelaufer/Desktop/DL-course/Data2project/running/process_state_decoder15_dense.txt", "a")
    
    # model = AUNet(1).float()
    model = DUNet(1).float()
    model.to(device)
    checkpoint = torch.load("/home/moshelaufer/Desktop/DL-course/Data2project/running/model15_dense.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    
    

    n_epochs = 85
    
    loss_arr = []
    loss_valid_arr = []
    tot_MAPE_arr =[]  
    print('start epoch')
    file.write('start epoch\n')      
    lr =0.001
    dim_arr = [16,32,64,128,256,512]
    for dim in dim_arr:
        dataset_train = DepthDataset('/home/moshelaufer/Desktop/DL-course/project2/data_train/',True,dim=dim)
        dataloader_train = DataLoader(dataset_train, batch_size=20, shuffle=True)
        train_length = len(dataloader_train)
     
        print("dim of pathc = {}".format(dim))
        for epoch in range(1,n_epochs):
            if epoch%40==0:
                lr =max(lr/2, 0.0001)
            model_optimizer = optim.Adam(model.parameters(), lr = lr,weight_decay=1e-5) 
            loss_tot = 0.0
            mape_tot=0.0
            for num, data in enumerate(dataloader_train):
            
                mono_image,stereo_image,y = data[:,0:1,:,:].float(), data[:,1:2,:,:].float(),data[:,2:,:,:].float()
                
                stereo_image = stereo_image.to(device)
                mono_image  = mono_image.to(device)
                y = y.to(device)
                
                model_optimizer.zero_grad()
                output = model(stereo_image,mono_image)
                loss = F.smooth_l1_loss(output,y)
                loss.backward()
                model_optimizer.step()
                loss_tot += loss.item() 
        
            model, loss_valid, mape_valid,_, _ = validation_calc(model,dataloader_valid,device)
                           
            loss_arr.append(loss_tot/train_length)
            
            loss_valid_arr.append(loss_valid)
            tot_MAPE_arr.append(mape_valid)
            
            print("Loss_train = {}, Loss_valid = {},epoch = {}. MAPE validation: {}".format(loss_tot,loss_valid, epoch,mape_valid))
            file.write("Loss = {}, epoch = {}".format(loss_tot,epoch))  
            outfile_epoch = "/home/moshelaufer/Desktop/DL-course/Data2project/running/loss15_dense.npy"
            np.save(outfile_epoch, np.asarray(loss_arr))
            np.save(outfile_epoch.replace('loss', 'loss_valid_arr'), np.asarray(loss_valid_arr))
            np.save(outfile_epoch.replace('loss', 'MAPE_valid_arr'), np.asarray(tot_MAPE_arr))
            if epoch>40 or dim>dim_arr[0]:
                if check_loss_validation(loss_valid_arr):
                    path = "/home/moshelaufer/Desktop/DL-course/Data2project/running/model15_dense.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict(),
                        'loss': loss_valid,
                        }, path)
                    
                    print("Model had been saved")
                    file.write("Model had been saved")
 
                
 
    print("training is over")
    file.write("training is over\n")
    torch.no_grad()
    print("Weight file had saccsufully saved!!\n")
    file.close()
if __name__ == "__main__":
    main()
        
               
    
