# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 9:49
# @Author  : Tianchiyue
# @File    : new_dongzhen_feature_process.py.py
# @Software: PyCharm Community Edition
import pandas as pd
import os
import copy

"""
将new_151,new_152提取的pca交互特征，tfidf交互特征整合到一个cav文件中
"""
raw_train_df = pd.read_csv('../data/train.csv')
raw_test_df = pd.read_csv('../data/test.csv')

raw_all_df = pd.concat([raw_train_df, raw_test_df])
X = copy.deepcopy(raw_all_df)
for fp in os.listdir('../data/interaction/pca/'):
    if fp[0] == 'w':
        feature_df = pd.read_pickle('../data/interaction/pca/' + fp).drop(
            columns=['label', 'q12_w_hamming', 'q12_w_jaccard'])
    else:
        feature_df = pd.read_pickle('../data/interaction/pca/' + fp).drop(
            columns=['label', 'q12_c_hamming', 'q12_c_jaccard'])
    feature_df.drop_duplicates(inplace=True)
    X = pd.merge(X, feature_df, how='left', left_on=['qid1', 'qid2'], right_on=['qid1', 'qid2'])
X.drop(columns=['qid1', 'qid2', 'label']).to_csv('../feature/dongzhen_pca.csv', encoding='utf8', index=False)

Y = copy.deepcopy(raw_all_df)
for fp in os.listdir('../data/interaction/tfidf/'):
    if fp[0] == 'w':
        feature_df = pd.read_pickle('../data/interaction/tfidf/' + fp).drop(columns=['label', 'q12_w_jaccard'])
    else:
        feature_df = pd.read_pickle('../data/interaction/tfidf/' + fp).drop(columns=['label', 'q12_c_jaccard'])
    feature_df.drop_duplicates(inplace=True)
    Y = pd.merge(Y, feature_df, how='left', left_on=['qid1', 'qid2'], right_on=['qid1', 'qid2'])

Y.drop(columns=['qid1', 'qid2', 'label']).to_csv('../feature/dongzhen_tfidf.csv', encoding='utf8', index=False)
