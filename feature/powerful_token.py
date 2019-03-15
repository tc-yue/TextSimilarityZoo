# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 11:32
# @Author  : Tianchiyue
# @File    : powerful_token.py
# @Software: PyCharm Community Edition
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def generate_powerful_token(df, cut='word'):
    """
    计算数据中词语的影响力，格式如下：
        词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
    """
    train_df = df.iloc[:20000, :]
    token2power = {}
    for index, row in train_df.iterrows():
        label = int(row['label'])
        if cut == 'word':
            q1_tokens = row['word1'].split()
            q2_tokens = row['word2'].split()
        else:
            q1_tokens = row['char1'].split()
            q2_tokens = row['char2'].split()
        all_tokens = set(q1_tokens+q2_tokens)
        q1_tokens = set(q1_tokens)
        q2_tokens = set(q2_tokens)
        for token in all_tokens:
            if token not in token2power:
                token2power[token] = [0. for i in range(7)]
            token2power[token][0] += 1.
            token2power[token][1] += 1.
            if ((token in q1_tokens) and (token not in q2_tokens)) or ((token not in q1_tokens) and (token in q2_tokens)):
                # 计算单侧语句数量
                token2power[token][3] += 1.
                if 0 == label:
                    # 计算正确语句对数量
                    token2power[token][2] += 1.
                    # 计算单侧语句正确比例
                    token2power[token][4] += 1.
            if (token in q1_tokens) and (token in q2_tokens):
                # 计算双侧语句数量
                token2power[token][5] += 1.
                if 1 == label:
                    # 计算正确语句对数量
                    token2power[token][2] += 1.
                    # 计算双侧语句正确比例
                    token2power[token][6] += 1.
    for token in token2power:
        # 计算出现语句对比例
        token2power[token][1] /= len(train_df)
        # 计算正确语句对比例
        token2power[token][2] /= token2power[token][0]
        # 计算单侧语句对正确比例
        if token2power[token][3] > 1e-6:
            token2power[token][4] /= token2power[token][3]
        # 计算单侧语句对比例
        token2power[token][3] /= token2power[token][0]
        # 计算双侧语句对正确比例
        if token2power[token][5] > 1e-6:
            token2power[token][6] /= token2power[token][5]
        # 计算双侧语句对比例
        token2power[token][5] /= token2power[token][0]
    return token2power


def generate_powerful_token_double_side(token2power, thresh_num=20, thresh_rate=0.85):
    # 至少在20个句子对中出现，且为正确的比例》=0.85， 高度相关词
    ptoken_dside = [token for token, power in token2power.items() if power[0] * power[5] >= thresh_num
                    and power[6] >= thresh_rate]
    return ptoken_dside


def extract_powerful_token_double_side(token1, token2, ptoken_dside):
    token1_list = token1.split()
    token2_list = token2.split()
    tags = []
    for token in ptoken_dside:
        if token in token1_list + token2_list:
            tags.append(1.0)
        else:
            tags.append(0.0)
    return tags


def generate_powerful_token_one_side(token2power, thresh_num=20, thresh_rate=0.85):
    # 至少在20个单句出现，但错误比例>=0.85 高度相关词
    ptoken_oside = [token for token, power in token2power.items() if power[0] * power[3] >= thresh_num
                    and power[4] >= thresh_rate]
    return ptoken_oside


def extract_powerful_token_one_side(token1, token2, ptoken_oside):
    token1_list = token1.split()
    token2_list = token2.split()
    tags = []
    for token in ptoken_oside:
        if token in token1_list and token not in token2_list:
            tags.append(1.0)
        elif token not in token1_list and token in token2_list:
            tags.append(1.0)
        else:
            tags.append(0.0)
    return tags

def powerful_token_double_side_rate(token1, token2, token2power, num_least = 100):
    # 双侧出现高频词， 双侧出现正确比例相乘，越大相似度越高
    token1_set = set(token1.split())
    token2_set = set(token2.split())
    common_token = token1_set & token2_set
    rate = 1.0
    for token in common_token:
        if token not in token2power:
            continue
        if token2power[token][0] * token2power[token][5] < num_least:
            continue
        rate *= (1.0-token2power[token][6])
    rate = 1-rate
    return rate

def powerful_token_one_side_rate(token1, token2, token2power, num_least = 100):
    # 单侧出现高频词， 单侧出现错误比例相乘，越大相似度越高
    token1_set = set(token1.split())
    token2_set = set(token2.split())
    q1_diff = list(token1_set.difference(set(token2_set)))
    q2_diff = list(token2_set.difference(set(token1_set)))
    all_diff = set(q1_diff + q2_diff)
    rate = 1.0
    for token in all_diff:
        if token not in token2power:
            continue
        if token2power[token][0] * token2power[token][3] < num_least:
            continue
        rate *= (1.0-token2power[token][4])
    rate = 1-rate
    return rate


if __name__ == '__main__':
    # TODO 调参
    raw_all_df = pd.read_csv('../data/raw_all_df.csv', encoding='utf8')
    char2power = generate_powerful_token(raw_all_df,cut='char')
    word2power = generate_powerful_token(raw_all_df, cut='word')

    pword_dside = generate_powerful_token_double_side(word2power, thresh_num=10, thresh_rate=0.95)
    pchar_dside = generate_powerful_token_double_side(char2power, thresh_num=10, thresh_rate=0.95)

    pword_oside = generate_powerful_token_one_side(word2power, thresh_num=10, thresh_rate=0.9)
    pchar_oside = generate_powerful_token_one_side(char2power, thresh_num=10, thresh_rate=0.9)

    powerful_char_one_side = np.array(
        raw_all_df.apply(lambda x: extract_powerful_token_one_side(x['char1'], x['char2'], pchar_oside),
                         axis=1).tolist())
    powerful_word_one_side = np.array(raw_all_df.apply(lambda x:
                                              extract_powerful_token_one_side(x['word1'], x['word2'], pword_oside),axis=1)).tolist()

    powerful_char_double_side = np.array(raw_all_df.apply(lambda x:
                                                 extract_powerful_token_double_side(x['char1'], x['char2'], pchar_dside),axis=1)).tolist()
    powerful_word_double_side = np.array(raw_all_df.apply(lambda x:
                                                 extract_powerful_token_double_side(x['word1'], x['word2'], pword_dside),axis=1)).tolist()

    X = pd.DataFrame(np.hstack([powerful_char_one_side, powerful_word_one_side,powerful_char_double_side,powerful_word_double_side]))
    X['char_double_rate'] = raw_all_df.apply(lambda x: powerful_token_double_side_rate(x['char1'],x['char2'],char2power),axis=1)
    X['word_double_rate'] = raw_all_df.apply(lambda x: powerful_token_double_side_rate(x['word1'],x['word2'],word2power),axis=1)
    X['char_one_rate'] = raw_all_df.apply(
        lambda x: powerful_token_one_side_rate(x['char1'], x['char2'], char2power),axis=1)
    X['word_one_rate'] = raw_all_df.apply(
        lambda x: powerful_token_one_side_rate(x['word1'], x['word2'], word2power),axis=1)

    X.to_csv('powerful_token.csv', index=False)
