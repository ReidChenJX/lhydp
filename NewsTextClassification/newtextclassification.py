#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/3/12 14:17
# @Author   : ReidChen
# Document  ：

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/train_set.csv', encoding='gbk',sep='\t',nrows=15000)

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))

# print(train_df['text_len'].describe())
# train_df['label'].value_counts().plot(kind='bar')
# plt.title('News class count')
# plt.xlabel("category")
# plt.show()


# 统计每个单词在训练集中出现的频率
# all_lines = ' '.join(list(train_df['text']))
# word_count = Counter(all_lines.split(" "))      # 以单词为字典的key，统计出现的次数，返回为dist，内存消耗大。
# word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)
#
# print(len(word_count))
# # 6869
#
# print(word_count[0])
# # ('3750', 7482224)
#
# print(word_count[-1])

# 池袋模型，Count Vectors 以独热编码转化训练集
# vectorizer = CountVectorizer(max_features=3000)
# train_test = vectorizer.fit_transform(train_df['text'])
#
# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])
#
# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# # 0.65441877581244

# TF_IDF 模型
from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
# train_test = tfidf.fit_transform(train_df['text'])
#
# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])
#
# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
# # # 0.8719098297954606

import fasttext

train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

model = fasttext.train_supervised('train.csv',lr=1.0, wordNgrams=2,
                                  verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
# # 0.82231338464308


