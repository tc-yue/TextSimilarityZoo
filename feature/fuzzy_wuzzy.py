# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 12:07
# @Author  : Tianchiyue
# @File    : fuzzy_wuzzy.py
# @Software: PyCharm Community Edition

from fuzzywuzzy import fuzz
import pandas as pd


def extract_features(df):
    """
    fuzzy wuzzy feature
    :param df:
    :return:
    """
    X = pd.DataFrame()
    print("fuzzy feature..")
    X["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["word1"], x["word2"]), axis=1)
    X["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["word1"], x["word2"]), axis=1)
    X["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["word1"], x["word2"]), axis=1)
    X["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["word1"], x["word2"]), axis=1)

    print("fuzzy feature..char,")
    X["token_set_ratio_char"] = df.apply(lambda x: fuzz.token_set_ratio(x["char1"], x["char2"]), axis=1)
    X["token_sort_ratio_char"] = df.apply(lambda x: fuzz.token_sort_ratio(x["char1"], x["char2"]), axis=1)
    X["fuzz_ratio_char"] = df.apply(lambda x: fuzz.QRatio(x["char1"], x["char2"]), axis=1)
    X["fuzz_partial_ratio_char"] = df.apply(lambda x: fuzz.partial_ratio(x["char1"], x["char2"]), axis=1)
    return X


if __name__ == '__main__':
    raw_all_df = pd.read_csv('../data/raw_all_df.csv', encoding='utf8')
    feature_df = extract_features(raw_all_df)
    feature_df.to_csv("fuzzy_wuzzy.csv", index=False)