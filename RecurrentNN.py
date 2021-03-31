#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/3/23 10:56
# @Author   : ReidChen
# Document  ：Recurrent Neural Networks

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from gensim.models import word2vec

"""loading data"""
path_prefix = './'


def load_training_data(path):
    # 读取数据，判断是否包含Label
    # 优化1：保留单词间的‘， 如i'm
    
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').replace(' \' ','\'').split() for line in lines]
        
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]

        return x, y
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').replace(' \' ','\'').split()  for line in lines]
        
        return x


def load_testing_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        X = [",".join(line.strip('\n').replace(' \' ','\'').split(',')[1:]).strip() for line in lines]
    return X


def evaluation(outputs, labels):
    # 准确率
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    # item  单元素的tensor，转化为Scalars数值
    return correct


"""Word to vector"""


def train_word2vec(x):
    #  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5,
                              workers=4, iter=1, sg=1)
    return model

#
# # 词频矩阵 word2vec 模型训练
# if __name__ == '__main__':
#     print("loading training data ...")
#     train_x, y = load_training_data('./hw4/training_label.txt')
#     train_x_no_label = load_training_data('./hw4/training_nolabel.txt')
#
#     print("loading testing data ...")
#     test_x = load_testing_data('./hw4/testing_data.txt')
#
#     model = train_word2vec(train_x + train_x_no_label)
#
#     print("saving model ...")
#     model.save(os.path.join(path_prefix, 'model/w2v_all.model'))



class Preprocess:
    def __init__(self, sentences, sen_len, w2v_path='./model/w2v_all.model'):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    
    def get_w2v_model(self):
        self.embedding = word2vec.Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size  # 250 词频长度
    
    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
    def make_embedding(self, load=True):
        # 将训练完成的w2v_model 加载，提出其中的单词和单词对应的vector
        # 建立单词与索引的列表，并按照索引顺序维护 vector，最后加入<PAD>与<UNK>
        print('Get embedding ~~~')
        
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        
        for i, word in enumerate(self.embedding.wv.vocab):
            print('Get words #{}'.format(i + 1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    
    def pad_sequence(self, sentence):
        # 统一句子长度为sen_len，若不够则在句子后面加<PAD>的编码
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    
    def sentence_word2idx(self):
        
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx['<UNK>'])
            
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        
        return torch.LongTensor(sentence_list)
    
    def label_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


"""Data set 自定义 '__init__', '__getitem__', '__len__'"""
from torch.utils import data


class TwitterDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    
    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)


"""训练模型 MODEL LSTM"""


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, 64),
                                        nn.Dropout(dropout),
                                        nn.Linear(64, 32),
                                        nn.Dropout(dropout),
                                        nn.Linear(32, 16),
                                        nn.Dropout(dropout),
                                        nn.Linear(16, 1),
                                        nn.Sigmoid())
    
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        
        return x


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

if __name__ == '__main__':
    # # main
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    # 处理data
    train_with_label = './hw4/training_label.txt'
    train_no_label = './hw4/training_nolabel.txt'
    testing_data = './hw4/testing_data.txt'
    
    w2v_path = './model/w2v_all.model'
    
    sen_len = 20
    fix_embedding = True
    batch_size = 128
    epoch = 5
    lr = 0.001
    
    train_x, y = load_training_data(train_with_label)
    train_x_no_label = load_training_data(train_no_label)
    
    # 数据预处理
    preprocess = Preprocess(train_x, sen_len)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.label_to_tensor(y)
    
    # 制作模型
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, fix_embedding=fix_embedding)
    model = model.to(device)
    
    # 训练数据集切分为train和validation
    X_train, X_val, y_train, y_val = train_x[:18000], train_x[18000:], y[:18000], y[18000:]
    
    # data 转变为DataSet
    train_dataset = TwitterDataset(X_train, y_train)
    val_dataset = TwitterDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=X_val, batch_size=batch_size, shuffle=False, num_workers=4)
    

