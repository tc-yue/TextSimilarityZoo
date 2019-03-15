# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 12:14
# @Author  : Tianchiyue
# @File    : long_common_string.py
# @Software: PyCharm Community Edition

import pandas as pd

import warnings
import numpy as np

warnings.filterwarnings(action='ignore')


# 编辑距离
def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


# 最长公共子序列
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def word_edit_share(row, sim=False):
    q1words = []
    q2words = []
    for word in str(row['words_x']).lower().split():
        q1words.append(word)
    for word in str(row['words_y']).lower().split():
        q2words.append(word)
    R = edit(q1words, q2words)
    if sim:
        maxleng = max(len(q1words), len(q2words))
        return (maxleng - R) / maxleng
    else:
        return R


def word_edit_share_char(row, sim=False):
    q1words = []
    q2words = []
    for word in str(row['chars_x']).lower().split():
        q1words.append(word)
    for word in str(row['chars_y']).lower().split():
        q2words.append(word)
    R = edit(q1words, q2words)
    if sim:
        maxleng = max(len(q1words), len(q2words))
        return (maxleng - R) / maxleng
    else:
        return R


def word_lcs_share(row, sim=False):
    q1words = []
    q2words = []
    for word in str(row['words_x']).lower().split():
        q1words.append(word)
    for word in str(row['words_y']).lower().split():
        q2words.append(word)
    R = lcs(q1words, q2words)
    if sim:
        return (R / len(q1words) + R / len(q2words)) / 2
    else:
        return R


def word_lcs_share_char(row, sim=False):
    q1words = []
    q2words = []
    for word in str(row['chars_x']).lower().split():
        q1words.append(word)
    for word in str(row['chars_y']).lower().split():
        q2words.append(word)
    R = lcs(q1words, q2words)
    if sim:
        return (R / len(q1words) + R / len(q2words)) / 2
    else:
        return R


if __name__ == '__main__':
    raw_all_df = pd.read_csv('../data/raw_all_df.csv', encoding='utf8')
    raw_all_df.rename(columns={'word1': 'words_x', 'word2': 'words_y','char1': 'chars_x', 'char2': 'chars_y'}, inplace=True)
    print('Building Features')
    X = pd.DataFrame()
    X['edit_dis'] = raw_all_df.apply(word_edit_share, axis=1, raw=True)
    X['edit_dis_char'] = raw_all_df.apply(word_edit_share_char, axis=1, raw=True)
    X['lcs_dis'] = raw_all_df.apply(word_lcs_share, axis=1, raw=True)
    X['lcs_dis_char'] = raw_all_df.apply(word_lcs_share_char, axis=1, raw=True)
    X['sim_edit_dis'] = raw_all_df.apply(lambda x: word_edit_share(x, sim=True), axis=1, raw=True)
    X['sim_edit_dis_char'] = raw_all_df.apply(lambda x: word_edit_share_char(x, sim=True), axis=1, raw=True)
    X['sim_lcs_dis'] = raw_all_df.apply(lambda x: word_lcs_share(x,sim=True), axis=1, raw=True)
    X['sim_lcs_dis_char'] = raw_all_df.apply(lambda x:word_lcs_share_char(x,sim=True), axis=1, raw=True)
    X.to_csv('long_common_string.csv', index=False)
