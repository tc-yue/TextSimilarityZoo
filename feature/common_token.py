# -*- coding: utf-8 -*-
# @Time    : 2018/11/11 21:13
# @Author  : Tianchiyue
# @File    : common_token.py
# @Software: PyCharm Community Edition

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
def token_count(q):
    token_list = q.split()
    token_set = set(token_list)
    return len(token_set)


def common_token_count(q1, q2):
    q1_token_list = q1.split()
    q2_token_list = q2.split()
    q1_token_set = set(q1_token_list)
    q2_token_set = set(q2_token_list)
    common_set = q1_token_set & q2_token_set
    comment_token_counts = len(common_set)
    return comment_token_counts

def jaccard(q1, q2):
    q1_token_list = q1.split()
    q2_token_list = q2.split()
    q1_token_set = set(q1_token_list)
    q2_token_set = set(q2_token_list)
    common_set = q1_token_set & q2_token_set
    comment_token_counts = len(common_set)
    return comment_token_counts/ len(q1_token_set|q2_token_set)


def token_weight(df):
    all_words = df['word1'].tolist()+df['word2'].tolist()
    all_chars = df['char1'].tolist()+df['char2'].tolist()
    word_tfidf = TfidfVectorizer()
    char_tfidf = TfidfVectorizer()
    word_tfidf.fit(all_words)
    char_tfidf.fit(all_chars)
    word2idf = {i:j for i,j in zip(word_tfidf.get_feature_names(),word_tfidf.idf_)}
    char2idf = {i: j for i, j in zip(char_tfidf.get_feature_names(), char_tfidf.idf_)}
    return word2idf, char2idf

def weighted_common_token(q1,q2,weights):
    q1_token_list = q1.split()
    q2_token_list = q2.split()
    q1_token_set = set(q1_token_list)
    q2_token_set = set(q2_token_list)
    common_set = q1_token_set & q2_token_set
    common_token_weighted = [weights.get(i.lower(), 0) for i in q1_token_list+q2_token_list if i in common_set]
    all_token_weighted = [weights.get(i.lower(), 0) for i in q1_token_list+q2_token_list]
    return np.sum(common_token_weighted)/np.sum(all_token_weighted)

# todo 第一个词或最后一个词是否相同


if __name__ == '__main__':
    """
    句子对共享token的数目，句子对token的jaccard系数，句子对共享token idf 加权得分，句子对共享token占原句的比例
    """
    raw_all_df = pd.read_csv('../data/raw_all_df.csv', encoding='utf8')
    word2idf, char2idf = token_weight(raw_all_df)

    raw_all_df['word_count_1'] = raw_all_df['word1'].apply(lambda x: token_count(x))
    raw_all_df['word_count_2'] = raw_all_df['word2'].apply(lambda x: token_count(x))
    raw_all_df['char_count_1'] = raw_all_df['char1'].apply(lambda x: token_count(x))
    raw_all_df['char_count_2'] = raw_all_df['char2'].apply(lambda x: token_count(x))

    raw_all_df['common_word_count'] = raw_all_df.apply(lambda x: common_token_count(x['word1'], x['word2']), axis=1)
    raw_all_df['common_char_count'] = raw_all_df.apply(lambda x: common_token_count(x['char1'], x['char2']), axis=1)

    raw_all_df['common_word_weighted'] = raw_all_df.apply(lambda x: weighted_common_token(x['word1'], x['word2'],
                                                                                           word2idf), axis=1)

    raw_all_df['common_char_weighted'] = raw_all_df.apply(lambda x: weighted_common_token(x['char1'], x['char2'],
                                                                                          char2idf), axis=1)

    raw_all_df['common_word_count_max_ratio'] = raw_all_df.apply(
        lambda x: x['common_word_count'] / min(x['word_count_1'], x['word_count_2']), axis=1)
    raw_all_df['common_word_count_min_ratio'] = raw_all_df.apply(
        lambda x: x['common_word_count'] / max(x['word_count_1'], x['word_count_2']), axis=1)
    raw_all_df['common_char_count_max_ratio'] = raw_all_df.apply(
        lambda x: x['common_char_count'] / min(x['char_count_1'], x['char_count_2']), axis=1)
    raw_all_df['common_char_count_min_ratio'] = raw_all_df.apply(
        lambda x: x['common_char_count'] / max(x['char_count_1'], x['char_count_2']), axis=1)

    raw_all_df['word_jaccard'] = raw_all_df.apply(lambda x: jaccard(x['word1'], x['word2']), axis=1)
    raw_all_df['char_jaccard'] = raw_all_df.apply(lambda x: jaccard(x['char1'], x['char2']), axis=1)

    feature_df = raw_all_df.loc[:, ['word_jaccard','char_jaccard',
                                    'common_word_count', ' common_char_count', 'common_word_count_max_ratio',
                                    'common_word_count_min_ratio', 'common_char_count_max_ratio',
                                    'common_char_count_min_ratio', 'word_count_abs', 'char_count_abs',
                                    'common_word_weighted','common_char_weighted']]. \
        to_csv('common_token.csv', index=False)
