#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/4/13 11:20
# @Author   : ReidChen
# Document  ：Attack CNN


import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cpu')

class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []
        
        for i in range(200):
            self.fnames.append("{:03d}".format(i))
            
    def __getitem__(self, item):
        img = Image.open(os.path.join(self.root, self.fnames[item]+'.png'))
        img = self.transforms(img)
        label = self.label[item]
        return img, label
        
    def __len__(self):
        files = os.listdir(self.root)
        return len(files)
    
class Attacker:
    def __iter__(self, img_dir, label):
        self.model = models.vgg16(pretrained=True)
        self.model.cuda()
        self.model.eval()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([transforms.Resize((224, 224), interpolation=3),
                                        transforms.ToTensor(),
                                        self.normalize])
        path = "./hw6/data/images"
        self.dataset = Adverdataset(path, label, transform)
        
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        
    def fgsm_attack(self, image, epsilion, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilion * sign_data_grad
        return perturbed_image
    
    def attack(self, epsilion):
        adv_examples = []
        wrong, fail, success = 0,0,0
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.request_grad = True
