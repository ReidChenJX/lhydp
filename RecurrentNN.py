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
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        
        return x, y
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        
        return x


def load_testing_data(path):
    with open(path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        X = [line.strip('\n').split(' ') for line in lines]
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

# # 词频矩阵 word2vec 模型训练
# if __name__ == '__main__':
#     print("loading training data ...")
#     train_x, y = load_training_data('./hw4/training_label.txt')
#     train_x_no_label = load_training_data('./hw4/training_nolabel.txt')
#
#     print("loading testing data ...")
#     test_x = load_testing_data('./hw4/testing_data.txt')
#
#     model = train_word2vec(train_x + train_x_no_label + test_x)
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
        self.embedding_dim = self.embedding.vector_size     # 250
        
    def add_embedding(self, word):
        vector = torch.empty(1,self.embedding_dim)
        torch.nn.init.uniform(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector],0)

    def make_embedding(self, load=True):
        print('Get embedding ~~~')
        
        if load :
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        
        for i, word in enumerate(self.embedding.wv.vocab):
            print('Get words #{}'.format(i+1), end='\r')
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
        
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len  = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    
    def sentence_word2idx(self):
        
        sentence_list =[]
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx['<UNK>'])
                    
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
            
        return torch.LongTensor(sentence_list)
    
    def label_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)