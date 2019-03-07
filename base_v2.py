# -*- ecoding:utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import datetime
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import utils

train = pd.read_csv('./jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv('./jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
# print(train.head())
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)

good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)
# print(train.columns)
train = train[good_cols]
good_cols.remove(u'收率')
test  = test[good_cols]

# 合并数据集
target = train[u'收率']
del train[u'收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)
# 得到时间

for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    data[f] = data[f].apply(utils.time2second)


for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[f] = data.apply(lambda df: utils.getDuration(df[f]), axis=1)


