#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:49:55 2023

@author: ksrao
"""

from torch.utils.data.dataset import Dataset
import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from torchvision import transforms, utils
import cv2
import pandas as pd
import pathlib
import csv




class morph_source_target(data.Dataset):
    def __init__(self,to_tensor_256=None,to_tensor_1024=None):
        
        self.to_tensor_256=to_tensor_256
        self.to_tensor_1024=to_tensor_1024
        # self.loc="/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Aravind_data/codes/restyle-encoder-main/face_morpf_new_latents/"
        self.loc = "/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/Testing/test_images/Female/"
        tloc1="/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/morphing_list/Testing/Female/txts/age_1/"
        self.dfImage=pd.read_csv(tloc1 +"morph_list.txt",header=None,sep=',')
        print(self.dfImage.head())
        self.LandmarkLoc = "/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/Testing/test_images/female_landmarks/"
        #self.df_tar=pd.read_csv("/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Aravind_data/database/face_morph/TBIOM_information/morph_list/P1_male_npy.txt",header=None,sep=' ')
        self.img_src=np.array(self.dfImage.loc[:,0])
        self.img_tar=np.array(self.dfImage.loc[:,1])
        self.LandmarkFilePath = "/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/morphing_list/Testing/Female/txts/age_1/morph_list_landmark.txt"
        
        #creating dataframes for landmark file
        self.landmarkDF = pd.read_csv(self.LandmarkFilePath, header=None, sep = ',')
        self.landmarkSrc = np.array(self.landmarkDF.loc[:,0])
        print(self.landmarkSrc)
        self.landmarkTar = np.array(self.landmarkDF.loc[:,1])
        
        
        #creating the dataframes for the latent files
        self.latentFolderLoc = "/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/Testing/test_images/female_latents_npy/"
        self.LatentFile = "/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/morphing_list/Testing/Female/txts/age_1/morph_list_latents.txt"
        self.latentDF = pd.read_csv(self.LatentFile, header=None, sep = ',')
        self.latentSrc = np.array(self.latentDF.loc[:,0])
        self.latentTar = np.array(self.latentDF.loc[:,1])
        
        #creating the dataframe for the mask files
        self.maskFolderLoc = "/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/Testing/test_images/female_masks/"

        self.MaskFile ="/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Source_Dataset/morphing_list/Testing/Female/txts/age_1/morph_list_mask.txt"
        self.maskDF = pd.read_csv(self.MaskFile, header=None, sep = ',')
        self.maskSrc = np.array(self.maskDF.loc[:,0])
        self.maskTar = np.array(self.maskDF.loc[:,1])
        
        
        
        #self.landmarkDF = pd.read_csv(self.LandmarkFilePath, header=None, sep = ' ')
        
        #self.img_src_npy=np.array(self.df_npy.loc[:,0])
        #self.img_tar_npy=np.array(self.df_npy.loc[:,1])
    
    def _cords_to_map_np(self, cords, img_size=(256,256), sigma=6):
        result = np.zeros(img_size + cords.shape[0:1], dtype='uint8')
        # print(cords)
        for i, point in enumerate(cords):
            # print("point:", point)
            if point[0] == -1 or point[1] == -1:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            x = np.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
            result[..., i] = x
        return result
    
    def encode_segmentation_rgb(self, segmentation, no_neck=True):
        parse = segmentation[:,:,0]

        face_part_ids = [1, 6, 7, 4, 5, 3, 2, 11, 12] if no_neck else [1, 6, 7, 4, 5, 3, 2, 11, 12, 17]
        mouth_id = 10
        hair_id = 13


        face_map = np.zeros([parse.shape[0], parse.shape[1]])
        mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
        hair_map = np.zeros([parse.shape[0], parse.shape[1]])

        for valid_id in face_part_ids:
            valid_index = np.where(parse==valid_id)
            face_map[valid_index] = 255
        valid_index = np.where(parse==mouth_id)
        mouth_map[valid_index] = 255
        valid_index = np.where(parse==hair_id)
        hair_map[valid_index] = 255

        return np.stack([face_map, mouth_map, hair_map], axis=2)
    
    
    
    
    def changeToLandmarkName(self, path):
        fileName = path[-1].split('.')[0] + '.npy'
        landmarkFile = []
        landmarkFile.extend([self.maskFolderLoc, fileName])
        landmarkFile = '/'.join(landmarkFile)
        return landmarkFile
        
    def buildLatentTable(self):
        res = []
        for src, tar in zip(self.img_src, self.img_tar):
            
            src = src.split('/')
            tar = tar.split('/')
            
            path = self.changeToLandmarkName(src) + ' ' + self.changeToLandmarkName(tar)
            
            res.append(path)
        
        print(res[0])
        
        filepath = '/'.join(self.maskFolderLoc.split('/')[:-1] + ['landmark_p1_female.txt'])
        with open(filepath , 'w') as fp:
            fp.write('\n'.join(res))
        
        print(filepath)
    
    #func=buildLatentTable(self)

    
    def buildMaskTable(self):
        pass
    
    def __getitem__(self,idx):
            #loading the image
            src_image_new=Image.open(self.loc+self.img_src[idx]).convert('RGB')
            
            tar_image_new=Image.open(self.loc+self.img_tar[idx]).convert('RGB')
           
            source_image=self.to_tensor_256(src_image_new)
            target_image=self.to_tensor_1024(tar_image_new)

            #loading landmark npy files
            srcLandmark = np.load(self.LandmarkLoc+self.landmarkSrc[idx])
            print(type(srcLandmark))
            print(srcLandmark.shape)
            s_map=self._cords_to_map_np(srcLandmark)
            tarLandmark = np.load(self.LandmarkLoc+self.landmarkTar[idx])
            t_map=self._cords_to_map_np(tarLandmark)

            #loading the latent npy files
            srcLatent = np.load(self.latentFolderLoc+self.latentSrc[idx])
            tarLatent = np.load(self.latentFolderLoc+self.latentTar[idx])
            
            
            #loading the mask png files
            src_mask=cv2.imread(self.maskFolderLoc+self.maskSrc[idx])

            tar_mask=cv2.imread(self.maskFolderLoc+self.maskTar[idx])
            t_mask=self.encode_segmentation_rgb(tar_mask)
            t_mask=cv2.resize(t_mask,(1024,1024))
            t_mask = t_mask.transpose((2, 0, 1)).astype(np.float)/255.0
            t_mask = t_mask[0] + t_mask[1]
            t_mask = cv2.dilate(t_mask, np.ones((50,50)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            mysc_pth = self.img_src[idx]
            mytr_pth = self.img_tar[idx]
            #print("here")
            return source_image,srcLatent,s_map,srcLandmark,target_image,tarLatent,t_map,tarLandmark,t_mask,mysc_pth,mytr_pth
        
 
    def __len__(self):
        return len(self.dfImage)
    
   
'''def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

to_tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

to_tensor2 = transforms.Compose([
        transforms.Resize(1024),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


root_dir='/home/ksrao/Aravind_files/face_morph/selected_for_p1_morph'
dataset=morph_source_target(to_tensor,to_tensor2)
train_loader=torch.utils.data.DataLoader(dataset,batch_size=4,
                                         shuffle=False,
                                         drop_last=True)
train_loader = sample_data(train_loader)
#source_image,target_image=next(train_loader)
#print(source_image,target_image)
#train_loader=sample_data(train_loader)
for x,y,w,z,a,b,c,d,e,f,g in train_loader:
    print(x,y,w,z,a,b,c,d,e,f,g)'''


        