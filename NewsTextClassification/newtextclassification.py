#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/3/12 14:17
# @Author   : ReidChen
# Document  ：

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('./data/train_set.csv', encoding='gbk',sep='\t')

train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.show()


# # 统计每个单词在训练集中出现的频率
# from collections import Counter
#
# nes = ' '.join(list(train_df['text']))
# word_count = Counter(all_lines.split(" "))
# word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)
#
# print(len(word_count))
# # 6869
#
# print(word_count[0])
# # ('3750', 7482224)
#
# print(word_count[-1])
