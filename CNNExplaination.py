#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/4/2 14:57
# @Author   : ReidChen
# Document  ：CNN Explaination

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
from ConvolutionalNN import Classifier

arg = {'ckptpath':'./checkpoint.pth',
       'dataset_dir':'./hw3/food-11/food-11/'}
args = argparse.Namespace(**arg)


modle = Classifier()
chekpoint = torch.load(args.ckptpath)
modle.load_state_dict(chekpoint['model_state_dict'])

class FoodDataset(Dataset):
    def __init__(self):
        
        pass
    
    
    def __len__(self):
        
        pass
    
    def __getitem__(self, item):
        
        pass
    

def normalize(image):
    return (image - image.min() ) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()
    
    x.request_grad()
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()
    
    saliencies = x.grad.abs().detach.cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    
    return saliencies
    
    