#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:35:20 2021

@author: moshelaufer
"""
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from os import listdir
from os.path import isfile, join
import numpy as np



class DepthDataset(Dataset):
    def __init__(self, path2data,transform_flag,dim, mean=(1.8488874, 1.8488874, 2.8670158984950036),std=(2.0259414,2.0259414, 1.9452832986060502)):
        self.path_list = [join(path2data,file) for file in listdir(path2data)]
        self.transform_flag = transform_flag
        self.transform = transform = transforms.Compose([
                # transforms.ToPILImage(), # because the input dtype is numpy.ndarray
                # transforms.Normalize(mean, std),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(size=(dim,dim), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
                # transforms.RandomRotation(degrees=180, fill=100), # because this method is used for PIL Image dtype
                # transforms.ToTensor(), # because inpus dtype is PIL Image
        ])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        data = np.load(self.path_list[index])
        if self.transform_flag:
            data = self.transform(torch.tensor(data)) 
        else:
            torch.tensor(data)
        return data


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader

#     dataset = DepthDataset('/home/moshelaufer/Desktop/DL-course/project2/data_train/')
#     dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
#     print(next(iter(dataloader)).size())