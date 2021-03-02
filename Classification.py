#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/3/2 14:12
# @Author   : ReidChen
# Document  ：

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
X_train_fpath = './hw2/data/X_train'
Y_train_fpath = './hw2/data/Y_train'
X_test_fpath = './hw2/data/X_test'
output_fpath = './hw2/output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
    
    
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # 数据中心化
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data
    if specified_column == None:
        specified_column = np.arange(X.shape[1])     # 注意此处为np.arange
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1,-1)
        X_std = np.std(X[:, specified_column], 0).reshape(1,-1)
        
    X[:,specified_column] = (X[:,specified_column] - X_mean) / (X_std + 1e-8)
    
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio=0.25):
    # 切分训练集与验证集,安装顺序切分，未打乱
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


def _shuffle(X, Y):
    # 将数组的顺序打乱并返回
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # sigmoid 函数，将数据限制在（0,1）之间
    res = (1 / (1 + np.exp(-z)))
    return np.clip(res, 1e-8, 1-(1e-8))     # np.clip, 将数组限制在一定范围内
    
def _f(X, w, b):
    # LR function，y=wx+b
    return _sigmoid(np.matmul(X, w) + b)    # np.matmul 矩阵的乘积

def _predict(X, w, b):
    # 预测函数
    return np.round(_f(X, w, b)).astype(np.int)

def _accuracy(Y_pred, Y_label):
    # 精确度
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

# 损失函数
def _cross_entropy_loss(Y_pred, Y_label):
    # loss= -yln(y*) - (1-y)ln(1-y*)
    cross_entropy = -np.dot(Y_label, np.log(Y_pred)) - np.dot((1-Y_label), np.log(1-Y_pred))
    return cross_entropy

# 梯度
def _gradient(X,Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    
    return w_grad, b_grad


# 训练
# 初始化w, b
w = np.zeros((data_dim,))
b = np.zeros((1,))

max_iter = 10
batch_size = 8
learning_rate = 0.2

# 记录训练过程中的损失与正常率
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# 初始化步骤
step = 1

for epoch in range(max_iter):
    # 批次，打乱数据
    X_train, Y_train = _shuffle(X_train, Y_train)
    
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]
        
        w_grad, b_grad = _gradient(X,Y,w,b)
        
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad
        # 步长自增，实现前期梯度跨度大，后期梯度跨度小
        step = step + 1
    
    # 每一批次训练完成，需进行预测并记录准确率
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
    
    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
    
print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

import matplotlib.pyplot as plt

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()
