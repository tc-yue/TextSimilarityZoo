# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 20:02
# @Author  : Tianchiyue
# @File    : embedding_dis.py
# @Software: PyCharm Community Edition

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis


"""
wmd 距离char and word
char and word sentence vector 
"""
def wmd(s1, s2, model):
    s1 = s1.split()
    s2 = s2.split()
    return model.wmdistance(s1, s2)


def sent2vec(s, model):
    words = s.split()
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def build_features(data):


    char_model = gensim.models.KeyedVectors.load_word2vec_format('../data/char_w2v.txt')
    word_model = gensim.models.KeyedVectors.load_word2vec_format('../data/word_w2v.txt')
    X = pd.DataFrame()
    X['word_wmd'] = data.apply(lambda x: wmd(x['word1'], x['word2'], word_model), axis=1)
    X['char_wmd'] = data.apply(lambda x: wmd(x['char1'], x['char2'], char_model), axis=1)
    question1_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.word1.values)):
        question1_vectors[i, :] = sent2vec(q, word_model)

    question2_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.word2.values)):
        question2_vectors[i, :] = sent2vec(q, word_model)

    char_question1_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.char1.values)):
        char_question1_vectors[i, :] = sent2vec(q, char_model)

    char_question2_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.char2.values)):
        char_question2_vectors[i, :] = sent2vec(q, char_model)
    #
    X['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

    X['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    X['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    X['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    X['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    X['char_skew_q1vec'] = [skew(x) for x in np.nan_to_num(char_question1_vectors)]
    X['char_skew_q2vec'] = [skew(x) for x in np.nan_to_num(char_question2_vectors)]
    X['char_kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(char_question1_vectors)]
    X['char_kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(char_question2_vectors)]


    return X


if __name__ == '__main__':
    raw_all_df = pd.read_csv('../data/raw_all_df.csv', encoding='utf8')
    feature_df = build_features(raw_all_df)
    feature_df.to_csv('embedding_dis.csv', index=False)


