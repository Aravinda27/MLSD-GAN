#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:30:44 2023

@author: user1
"""

import argparse
import math
import random
import os,time
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torchvision
from torchvision import transforms, utils
from numpy.random import randint
from tqdm import tqdm

from dataset import *
from torch.autograd import Variable
import matplotlib as mlb


import itertools
from tensorboardX import SummaryWriter

from parser_morph_csv_new import *
from utils import *
from models.encoders.psp_encoders import *
from models.stylegan2.model import *
from models.nets import *


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)



def train(args, train_loader, encoder_lmk, encoder_target, generator, decoder, bald_model,device):

    train_loader = sample_data(train_loader)
    generator.eval()
    encoder_lmk.eval()
    encoder_target.eval()

    zero_latent = torch.zeros((args.batch,18-args.coarse,512)).to(device).detach()

    trans_256 = transforms.Resize(256)
    trans_1024 = transforms.Resize(1024)






    pbar = range(args.sample_number)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    
    with open('mytable.txt','w') as myfl:
        for idx in pbar:
            i = idx + args.start_iter
    
            if i > args.sample_number:
                print("Done!")
                break
    
            time0 = time.time()
    
            s_img,s_code,s_map,s_lmk,t_img,t_code,t_map,t_lmk,t_mask,mysc_pth,mytr_pth = next(train_loader) #256;1024;...
            #pths1 = str(mysc_pth)
            mysc_pth = mysc_pth[0].rsplit('/',1)[-1]
            mytr_pth = mytr_pth[0].rsplit('/',1)[-1]
            #mysc_pth = mysc_pth - '.jpg'
            mysc_pth = mysc_pth.rsplit('.',1)[0]
            mytr_pth = mytr_pth.rsplit('.',1)[0]
            #print(mysc_pth)
            #exit()
            time1 = time.time()
            s_img = s_img.to(device)
            s_map = s_map.to(device).transpose(1,3).float()#[:,33:]
            t_img = t_img.to(device)
            t_map = t_map.to(device).transpose(1,3).float()#[:,33:]
            t_lmk = t_lmk.to(device)
            t_mask = t_mask.to(device)
    
            s_frame_code = s_code.to(device)
            print("S_frame_code",s_frame_code.shape)
            t_frame_code = t_code.to(device)
    
    
    
            input_map = torch.cat([s_map,t_map],dim=1)
            t_mask = t_mask.unsqueeze_(1).float()
    
            t_lmk_code = encoder_lmk(input_map) 
    
    
            t_lmk_code = torch.cat([t_lmk_code,zero_latent],dim=1)
            
            fusion_code = s_frame_code + t_lmk_code
            
            alpha=0.50
            
            #s_fusion_code_linear=s_fusion_code[:,:18-args.coarse]
            #t_fusion_code_linear=t_frame_code[:,:18-args.coarse]
            
            #fusion_linear_interp=(s_frame_code*(1-alpha)+t_lmk_code*alpha)
            
            s_fusion_slerp=s_frame_code[:,:18-args.coarse:]
            print("S_fusion_slerp",s_fusion_slerp.shape)
            t_fusion_slerp=t_frame_code[:,:18-args.coarse]
            print("t_fusion_code",t_fusion_slerp.shape)
    
            fusion_code = torch.cat([fusion_code[:,:18-args.coarse],t_frame_code[:,18-args.coarse:]],dim=1)
            
            theta = torch.acos(torch.sum(s_fusion_slerp *t_fusion_slerp) / (torch.norm(s_fusion_slerp) * torch.norm(t_fusion_slerp)))
            print("Theta",theta.shape)
            fusion_code=torch.sin((1 - alpha) * theta) / torch.sin(theta) * s_fusion_slerp
            + torch.sin(alpha * theta) / torch.sin(theta) * t_fusion_slerp
            print(fusion_code.shape)
            
            #fusion_code=torch.cat([fusion_linear_interp,fusion_slerp],dim=1)
            
            fusion_code = bald_model(fusion_code.view(fusion_code.size(0), -1), 2)
            fusion_code = fusion_code.view(t_frame_code.size())
            #fusion_code=torch.mean(fusion_code,1)
            print("Fusion_code",fusion_code.shape)
            
    
    
            source_feas = generator([fusion_code], input_is_latent=True, randomize_noise=False)
            print("Source_feas",len(source_feas))
            target_feas = encoder_target(t_img)
            
            blend_img = decoder(source_feas,target_feas,t_mask)
    
    
            s_index = randint(0, 3 - 1)
            t_index = randint(0, 3 - 1)
            name = str(int(s_index))+'_'+str(int(t_index))
    
            myfl.write(f'{mysc_pth}.png\t{mytr_pth}.png\t{name}.png')
            with torch.no_grad():
                sample = torch.cat([s_img.detach(), t_img.detach()])
                utils.save_image(
                    s_img.detach(),
                    _dirs[2]+"/"+mysc_pth+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    t_img.detach(),
                    _dirs[4]+"/"+mysc_pth+'_'+mytr_pth+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    s_img.detach(),
                    _dirs[5]+"/"+mysc_pth+'_'+mytr_pth+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    t_img.detach(),
                    _dirs[3]+"/"+mytr_pth+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )
                sample = torch.cat([sample, blend_img.detach()])
                t_mask = torch.stack([t_mask,t_mask,t_mask],dim=1).squeeze(2)
                sample = torch.cat([sample, t_mask.detach()])
    
                utils.save_image(
                    blend_img,
                    _dirs[0]+"/"+name+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    blend_img,
                    _dirs[6]+"/"+mysc_pth+'_'+mytr_pth+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )
    
                utils.save_image(
                    sample,
                    _dirs[1]+"/"+name+".png",
                    nrow=int(args.batch),
                    normalize=True,
                    range=(-1, 1),
                )






if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--e_ckpt", type=str, default='/media/user1/8089c1c4-5bde-4b0d-abd1-ac7131e6ceb4/Aravind_data_other_drive/codes/FSLSD_HiRes-main_new/CELEBA-HQ-1024.pt')


    parser.add_argument("--image_path", type=str, default="/media/user1/38e499de-0c47-4e4b-8da7-cbf7c595d87c/Aravind_data/database/CelebAMask-HQ_new/CelebAMask-HQ/CelebA-HQ-img")
    
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sample_number", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--lpips", type=float, default=5.0)
    parser.add_argument("--hm", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=10.0)

    parser.add_argument("--fd", type=float, default=10.0)



    parser.add_argument("--id", type=float, default=10.0)
    parser.add_argument("--lmk", type=float, default=0.1)
    parser.add_argument("--adv", type=float, default=5.0)  
    parser.add_argument("--tv", type=float, default=10.0)  

    parser.add_argument("--mask", type=float, default=5.0)  


    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--tensorboard", action="store_true",default=True)
    



    
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    synchronize()
    device = args.device

    args.path = ".//"    
    args.start_iter = 0

    args.size = 1024
    args.latent = 512
    args.n_mlp = 8
    args.channel_multiplier = 2
    args.coarse = 7
    args.least_size = 8
    args.largest_size = 512
    args.mapping_layers = 18
    args.mapping_fmaps = 512
    args.mapping_lrmul = 1
    args.mapping_nonlinearity = 'linear'


    encoder_lmk = GradualLandmarkEncoder(106*2).to(device)
    encoder_target = GPENEncoder(args.largest_size).to(device)
    generator = Generator(args.size,args.latent,args.n_mlp).to(device)
    
    decoder = Decoder(args.least_size,args.size).to(device)

    bald_model = F_mapping(mapping_lrmul= args.mapping_lrmul, mapping_layers=args.mapping_layers, mapping_fmaps=args.mapping_fmaps, mapping_nonlinearity = args.mapping_nonlinearity).to(device)
    bald_model.eval()



    _dirs = ['./test/swapped/','./test/fuse/','./test/source/','./test/target/','./test/ctarget','./test/csource/','./test/cswapped/']
    
    for x in _dirs:
        try:
            os.makedirs(x)
        except:
            pass

    to_tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    to_tensor2 = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    #dataset = SourceICTargetICLM(args.image_path,to_tensor_256 = to_tensor2, to_tensor_1024=to_tensor2)
    dataset=morph_source_target(to_tensor_256=to_tensor2,to_tensor_1024=to_tensor2)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            drop_last=True,
        )



    encoder_lmk = nn.parallel.DistributedDataParallel(encoder_lmk,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    encoder_target = nn.parallel.DistributedDataParallel(encoder_target,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    generator = nn.parallel.DistributedDataParallel(generator,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    decoder = nn.parallel.DistributedDataParallel(decoder,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)
    bald_model = nn.parallel.DistributedDataParallel(bald_model,device_ids=[args.local_rank],output_device=args.local_rank,broadcast_buffers=False,find_unused_parameters=True)


    e_ckpt = torch.load(args.e_ckpt,  map_location=torch.device('cpu'))

    encoder_lmk.load_state_dict(e_ckpt["encoder_lmk"])
    encoder_target.load_state_dict(e_ckpt["encoder_target"])
    decoder.load_state_dict(e_ckpt["decoder"])
    generator.load_state_dict(e_ckpt["generator"])
    bald_model.load_state_dict(e_ckpt["bald_model"])






    train(args, train_loader, encoder_lmk, encoder_target, generator, decoder, bald_model,device)
