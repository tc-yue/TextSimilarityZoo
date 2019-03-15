# -*- coding: utf-8 -*-
# @Time    : 2018/11/11 22:44
# @Author  : Tianchiyue
# @File    : token_count.py.py
# @Software: PyCharm Community Edition

import pandas as pd


def unique_token_count(q):
    token_list = q.split()
    token_set = set(token_list)
    return len(token_set)


def token_count(q):
    token_list = q.split()
    return len(token_list)


def get_rate(a, b):
    res1 = float(a / b)
    res2 = float(b / a)
    return min(res1, res2)


if __name__ == '__main__':
    raw_all_df = pd.read_csv('../data/raw_all_df.csv', encoding='utf8')

    raw_all_df['word_unique_count_1'] = raw_all_df['word1'].apply(lambda x: unique_token_count(x))
    raw_all_df['word_unique_count_2'] = raw_all_df['word2'].apply(lambda x: unique_token_count(x))
    raw_all_df['char_unique_count_1'] = raw_all_df['char1'].apply(lambda x: unique_token_count(x))
    raw_all_df['char_unique_count_2'] = raw_all_df['char2'].apply(lambda x: unique_token_count(x))

    raw_all_df['word_count_1'] = raw_all_df['word1'].apply(lambda x: token_count(x))
    raw_all_df['word_count_2'] = raw_all_df['word2'].apply(lambda x: token_count(x))
    raw_all_df['char_count_1'] = raw_all_df['char1'].apply(lambda x: token_count(x))
    raw_all_df['char_count_2'] = raw_all_df['char2'].apply(lambda x: token_count(x))

    raw_all_df['word_count_diff_abs'] = raw_all_df.apply(lambda x: abs(x['word_count_1'] - x['word_count_2']), axis=1)
    raw_all_df['char_count_diff_abs'] = raw_all_df.apply(lambda x: abs(x['char_count_1'] - x['char_count_2']), axis=1)
    raw_all_df['word_unique_count_diff_abs'] = raw_all_df.apply(lambda x:
                                                                abs(x['word_unique_count_1'] - x[
                                                                    'word_unique_count_2']), axis=1)
    raw_all_df['char_unique_count_diff_abs'] = raw_all_df.apply(lambda x:
                                                                abs(x['char_unique_count_1'] - x[
                                                                    'char_unique_count_2']), axis=1)

    raw_all_df['word_count_rate'] = raw_all_df.apply(lambda x: get_rate(x['word_count_1'], x['word_count_2']), axis=1)
    raw_all_df['char_count_rate'] = raw_all_df.apply(lambda x: get_rate(x['char_count_1'], x['char_count_2']), axis=1)
    raw_all_df['word_unique_rate'] = raw_all_df.apply(lambda x:
                                                      get_rate(x['word_unique_count_1'], x[
                                                          'word_unique_count_2']), axis=1)
    raw_all_df['char_unique_rate'] = raw_all_df.apply(lambda x:
                                                      get_rate(x['char_unique_count_1'], x[
                                                          'char_unique_count_2']), axis=1)

    feature_df = raw_all_df.loc[:, ['word_count_1', 'word_count_2', 'char_count_1', 'char_count_2',
                                    'word_unique_count_1', 'word_unique_count_2', 'char_unique_count_1',
                                    'char_unique_count_2',
                                    'word_count_diff_abs', 'char_count_diff_abs', 'word_unique_count_diff_abs',
                                    'char_unique_count_diff_abs',
                                    'word_count_rate', 'char_count_rate', 'word_unique_rate', 'char_unique_rate']]. \
        to_csv('token_count.csv', index=False)
