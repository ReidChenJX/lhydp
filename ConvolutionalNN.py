#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/3/18 16:32
# @Author   : ReidChen
# Document  ：Homework 3 - Convolutional Neural Network

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time


# 将图片数据存放在numpy array 中
def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
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


"""训练模型"""

model = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 分类问题，损失函数为交叉熵函数
# 优化方法为 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0  # acc 准确率
    train_loss = 0  # 误差
    val_acc = 0
    val_loss = 0
    
    model.train()
    # 训练数据
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度归零
        train_pred = model(data[0].cuda())  # data 包含训练与标签，data[0]为训练数据
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()  # 计算误差的梯度
        optimizer.step()  # 更新优化
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())
            
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))

print("*******************分割线*************")

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    
    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
        
        # 將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
          (epoch + 1, num_epoch, time.time() - epoch_start_time,
           train_acc / train_val_set.__len__(), train_loss / train_val_set.__len__()))
