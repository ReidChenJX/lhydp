#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/4/2 14:57
# @Author   : ReidChen
# Document  ：CNN Explaination

import os
import sys
import argparse
import numpy as np
import cv2

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


# 将图片数据存放在numpy array 中
def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        # CV2的imread默认存储的颜色空间顺序是BGR，与matplot显示用的imshow的颜色顺序RGB相反，需转化。
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x[i, :, :] = cv2.resize(img, (128, 128))  # x 为四维np.zero
        
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


workspace_dir = './hw3/food-11/food-11/'
print("Reading data")

train_x, train_y = readfile(os.path.join(workspace_dir, 'training'), True)
print("Size of training data = {} ".format(len(train_x)))

val_x, val_y = readfile(os.path.join(workspace_dir, 'validation'), True)
print("Size of validation data ={} ".format(len(val_x)))

test_x = readfile(os.path.join(workspace_dir, 'validation'), False)
print("Size of Testing data = {} ".format(len(test_x)))

# train 数据进行随机变化，增加训练样本
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),
])

# testing 不需要
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
    
    def get_batch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


# transforms.ToTensor() 将(128, 128, 3)的图片，按照RGB进行分层，转变为(3, 128, 128)的色调图
# 其中(128, 128)中没一个值都是在【0-1】之间


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Conv2d 为二维卷积，参数为：in_channels 输入的维度，我们将RGB图片进行三层分开，为3
        #                        out_channels 输出维度，也是filter 的个数，设置为64
        #                        kernel_size  卷积核的大小，3 代表 3X3 的卷积核
        #                        stride  卷积步长，默认为1
        #                        padding  填充数，卷积后，H和W会少2（边界），设置1 可保证每条边1 的填充。保持样本大小
        # BatchNorm2d 为filter 后的feature map 提供 normalization 标准化，输入为filter 的输出
        # ReLU 激活函数
        # MaxPool2d 为池化层 参数为   kernel_size 池化维度，2 代表 以2X2 的维度，将四个数据保留一个数据
        #                          stride 卷积步长，与kernel_size 一样代表不重合取值
        #                          padding 填充数 不填充
        # 这样的池化处理，相当于将数据的大小缩小了一半
        # input 維度 [3, 128, 128]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 此时输入为 (64, 64, 64)
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 此时输出为（128, 32, 32）
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出为（256, 16, 16)
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出为（512, 8, 8）
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # 输出为（512,4,4）
        )  # Conversation （512,4,4）
        # 卷积 池化 后，连接一个全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)  # 最终的输出为图片类别中其一
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


model = Classifier().cuda()


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


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


img_indices = [83, 4218, 4707, 8598]
imgages, labels = train_set.get_batch(img_indices)
saliencies = compute_saliency_maps(imgages, labels, model)

fig, axs = plt.subplot(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([img_indices, saliencies]):
    for column, img in enumerate(target):
        axs[row][column].imshow(img.permute(1, 2, 0).numpy())

plt.show()
plt.close()


layer_activations = None

def filter_explain(x, model, cnnid, filterid, iteration=100, lr=1):
    
    model.eval()
    def hook(model, input, output):
        global layer_activations
        layer_activations = output
        
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    
    model(x.cuda())
    filter_activations = layer_activations[:,filterid,:,:].detach().cpu()
    x = x.cuda()
    x.request_grad()
    optimizer = Adam([x], lr=lr)
    
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
        
        objective = -layer_activations[:,filterid,:,:].sum()
        objective.backward()
        optimizer.step()
    
    filter_visualization = x.detach().cpu().squeeze()[0]
    
    hook_handle.remove()
    
    return filter_activations, filter_visualization


images, labels = train_set.get_batch(img_indices)
filter_activations, filter_visualization = filter_explain(images, model, cnnid=15, filterid=0, iteration=100, lr=0.1)

plt.imshow(normalize(filter_visualization.permute(1,2,0)))
plt.show()
plt.close()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15,8))
for i, img in enumerate(images):
    axs[0][i].imshow(img.permute(1,2,0))
for i, img in enumerate(filter_activations):
    axs[1][i].imshow(normalize(img))
    
plt.show()
plt.close()

from torch.optim import Adam


def predict(input):
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    
    output = model(input.cuda())
    return output.detach().cpu().numpy()


def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊
    return slic(input, n_segments=100, compactness=1, sigma=1)


img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.get_batch(img_indices)
fig, axs = plt.subplots(1, 4, figsize=(15, 8))
np.random.seed(16)
# 讓實驗 reproducible
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
    
    lime_img, mask = explaination.get_image_and_mask(
        label=label.item(),
        positive_only=False,
        hide_rest=False,
        num_features=11,
        min_weight=0.05
    )
    
    axs[idx].imshow(lime_img)

plt.show()
plt.close()
