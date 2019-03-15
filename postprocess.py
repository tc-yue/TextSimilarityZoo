# -*- coding: utf-8 -*-
# @Time    : 2018/11/27 14:59
# @Author  : Tianchiyue
# @File    : postprocess.py
# @Software: PyCharm Community Edition
import pickle
import copy
import itertools
from collections import Counter
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import os
import random as rn
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '7'
np.random.seed(7)
rn.seed(7)


def load_data(data_path):
    with open(data_path, 'rb')as f:
        data = pickle.load(f)
    #     logging.info('\t=======Data Loaded=======')
    return data


# 原始数据读取
question_id_df = pd.read_csv('data/question_id.csv')

raw_train_df = pd.read_csv('data/train.csv')
raw_test_df = pd.read_csv('data/test.csv')
raw_all_df = pd.concat([raw_train_df, raw_test_df])
word_padded_sequences, char_padded_sequences, word_embedding_matrix, char_embedding_matrix, y = load_data(
    'data/processed.pkl')

# stacking第一层结果读取
train_carnn_char, test_carnn_char = load_data('cv_result/carnn_char.pkl')
train_carnn_word, test_carnn_word = load_data('cv_result/carnn_word.pkl')
train_carnn_features_word, test_carnn_features_word = load_data('cv_result/carnn_features_word.pkl')
train_carnn_features_char, test_carnn_features_char = load_data('cv_result/carnn_features_char.pkl')

train_cnn_rnn_char, test_cnn_rnn_char = load_data('cv_result/cnn_rnn_char.pkl')
train_cnn_rnn_word, test_cnn_rnn_word = load_data('cv_result/cnn_rnn_word.pkl')
train_cnn_rnn_features_word, test_cnn_rnn_features_word = load_data('cv_result/cnn_rnn_features_word.pkl')
train_cnn_rnn_features_char, test_cnn_rnn_features_char = load_data('cv_result/cnn_rnn_features_char.pkl')

train_cnn_stacked_char, test_cnn_stacked_char = load_data('cv_result/cnn_stacked_char.pkl')
train_cnn_stacked_word, test_cnn_stacked_word = load_data('cv_result/cnn_stacked_word.pkl')
train_cnn_stacked_features_word, test_cnn_stacked_features_word = load_data('cv_result/cnn_stacked_features_word.pkl')
train_cnn_stacked_features_char, test_cnn_stacked_features_char = load_data('cv_result/cnn_stacked_features_char.pkl')

train_decom_highway_char, test_decom_highway_char = load_data('cv_result/decom_highway_char.pkl')
train_decom_highway_word, test_decom_highway_word = load_data('cv_result/decom_highway_word.pkl')
train_decom_highway_features_word, test_decom_highway_features_word = load_data(
    'cv_result/decom_highway_features_word.pkl')
train_decom_highway_features_char, test_decom_highway_features_char = load_data(
    'cv_result/decom_highway_features_char.pkl')

train_esim_char, test_esim_char = load_data('cv_result/esim_char.pkl')
train_esim_word, test_esim_word = load_data('cv_result/esim_word.pkl')
train_esim_word_char, test_esim_word_char = load_data('cv_result/esim_word_char.pkl')
train_esim_features_word, test_esim_features_word = load_data('cv_result/esim_features_all_word.pkl')
train_esim_features_char, test_esim_features_char = load_data('cv_result/esim_features_all_char.pkl')
train_esim_features_word_char, test_esim_features_word_char = load_data('cv_result/esim_features_all_word_char.pkl')

train_rnn_stacked_char, test_rnn_stacked_char = load_data('cv_result/rnn_stacked_char.pkl')
train_rnn_stacked_word, test_rnn_stacked_word = load_data('cv_result/rnn_stacked_word.pkl')
train_rnn_stacked_features_word, test_rnn_stacked_features_word = load_data('cv_result/rnn_stacked_features_word.pkl')
train_rnn_stacked_features_char, test_rnn_stacked_features_char = load_data('cv_result/rnn_stacked_features_char.pkl')

all_train_pred = [
    train_carnn_char, train_carnn_word, train_carnn_features_char, train_carnn_features_word,
    train_cnn_rnn_char, train_cnn_rnn_word, train_cnn_rnn_features_char, train_cnn_rnn_features_word,
    train_cnn_stacked_char, train_cnn_stacked_word, train_cnn_stacked_features_char, train_cnn_stacked_features_word,
    train_decom_highway_char, train_decom_highway_word, train_decom_highway_features_char,
    train_decom_highway_features_word,
    train_esim_char, train_esim_word, train_esim_features_char, train_esim_features_word,
    train_esim_features_word_char, train_esim_word_char,
    train_rnn_stacked_char, train_rnn_stacked_word, train_rnn_stacked_features_char, train_rnn_stacked_features_word
]

all_test_pred = [
    test_carnn_char, test_carnn_word, test_carnn_features_char, test_carnn_features_word,
    test_cnn_rnn_char, test_cnn_rnn_word, test_cnn_rnn_features_char, test_cnn_rnn_features_word,
    test_cnn_stacked_char, test_cnn_stacked_word, test_cnn_stacked_features_char, test_cnn_stacked_features_word,
    test_decom_highway_char, test_decom_highway_word, test_decom_highway_features_char,
    test_decom_highway_features_word,
    test_esim_char, test_esim_word, test_esim_features_char, test_esim_features_word,
    test_esim_features_word_char, test_esim_word_char,
    test_rnn_stacked_char, test_rnn_stacked_word, test_rnn_stacked_features_char, test_rnn_stacked_features_word,
]
train_new_x = np.hstack([i.reshape(-1, 1) for i in all_train_pred])
test_new_x = np.hstack([i.reshape(-1, 1) for i in all_test_pred]) / 5


# 第二层stacking
def bagging_clfs(clf, train_x, train_y, test_new_x, n_folds):
    #     随机抽样，20%做验证集
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
    final_preds_list1 = []
    final_preds_list2 = []
    score_list1 = []
    score_list2 = []
    new_train_y = np.zeros((len(train_y), 2))
    for clf_id, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('第{}个模型---'.format(clf_id))
        skf_x_train = train_new_x[train_index]  # 800,10
        skf_y_train = train_y[train_index]  # 800,3
        skf_x_valid = train_new_x[valid_index]  # 200,10
        skf_y_valid = train_y[valid_index]
        classif = clf
        classif.fit(skf_x_train, skf_y_train)
        valid_subject_preds = classif.predict(skf_x_valid)
        valid_subject_proba = classif.predict_proba(skf_x_valid)
        test_subject_preds = classif.predict(test_new_x)
        test_subject_proba = classif.predict_proba(test_new_x)
        score_list1.append(f1_score(skf_y_valid, valid_subject_preds))
        score_list2.append(roc_auc_score(skf_y_valid, valid_subject_preds))
        final_preds_list1.append(test_subject_preds)
        final_preds_list2.append(test_subject_proba)
        new_train_y[valid_index] = valid_subject_proba
        del classif
    print('平均fscore:{}_____{}'.format(np.average(score_list1), np.average(score_list2)))
    return final_preds_list1, final_preds_list2, new_train_y


lgb_parameters = {
    'learning_rate': 0.05,
    'application': 'binary',
    'max_depth': 6,
    'num_leaves': 30,
    'verbosity': -1,
    'data_random_seed': 7,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.7,
    'n_jobs': -1,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'n_estimators': 60
}
lgb_clf = lgb.LGBMClassifier(**lgb_parameters)
lgb_preds1, lgb_proba1, lgb_new_train_y1 = bagging_clfs(lgb_clf, train_new_x, np.array(y), test_new_x, 5)

# xgb_parameters = {'nthread': -1,
#                   'gamma': 0.1,
#                   'objective': 'binary:logistic',
#                   'learning_rate': 0.05,
#                   'max_depth': 6,
#                   'min_child_weight': 2,
#                   'silent': 1,
#                   'subsample': 0.7,
#                   'colsample_bytree': 0.8,
#                   'n_estimators': 100,
#                   'missing': -999,
#                   'seed': 7}
#
# xgb_clf = xgb.XGBClassifier(**xgb_parameters)
# xgb_preds1, xgb_proba1, xgb_new_train_y1 = bagging_clfs(xgb_clf, train_new_x, np.array(y), test_new_x, 5)

lr_parameters = {
    'solver': 'liblinear',
    'penalty': 'l2',
    'C': 0.1
}
lr_clf = LogisticRegression(**lr_parameters)
lr_preds, lr_proba, lr_new_train_y = bagging_clfs(lr_clf, train_new_x, np.array(y), test_new_x, 5)

svm = LinearSVC(C=1.0)
lsvc_clf = CalibratedClassifierCV(svm)
lsvc_preds, lsvc_proba, lsvc_new_train_y = bagging_clfs(lsvc_clf, train_new_x, np.array(y), test_new_x, 5)


def get_score(y, y_pred):
    print(f1_score(y, y_pred), roc_auc_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred))


sum_res = lgb_new_train_y1 + lr_new_train_y + lsvc_new_train_y
sum_res1 = np.argmax(sum_res, axis=-1)
get_score(y, sum_res1)

xgb_lgb = lgb_proba1 + lsvc_proba + lr_proba
xgb_lgb_res = np.argmax(np.sum(xgb_lgb, axis=0), axis=-1)
raw_test_df['label'] = xgb_lgb_res
raw_test_df.to_csv('result.csv')
