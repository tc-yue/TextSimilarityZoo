# -*- coding: utf-8 -*-
# @Time    : 2018/10/24 14:24
# @Author  : Tianchiyue
# @File    : preprocess.py
# @Software: PyCharm Community Edition

import pandas as pd
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import StratifiedKFold

def data2sequence(texts, domain_embedding, max_num_words, max_sequence_length,word2index=None):
    #     从1到MAX_NUM_WORDS-1
    if word2index:
        sequences = np.array([[word2index[token] if token in word2index else word2index['UNK']
                               for token in line]
                              for line in texts])
        word_index = word2index
    else:
        tokenizer = Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index

    # 词向量矩阵第0个留出 便于mask
    num_words = min(max_num_words, len(word_index))

    # 取均值中位数最大数等,500，截取方式
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, truncating='post')

    domain_embedding_dims = 300
    extra_words = []
    domain_embedding_matrix = np.zeros((num_words + 1, domain_embedding_dims))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        if word in domain_embedding:
            domain_embedding_matrix[i] = domain_embedding[word]
        else:
            #         未登录词随机初始化
            extra_words.append(word)
            print(word)
            domain_embedding_matrix[i] = np.random.uniform(low=-0.01, high=0.01, size=(domain_embedding_dims,))
    print('截取后未登录词个数:{}'.format(len(extra_words)))
    print('Found %s unique tokens.' % len(word_index))
    print('词平均长度:', np.average([len(i.split(' ')) for i in texts]))
    print('最大长度:', np.max([len(i.split(' ')) for i in texts]))
    if not word2index:
        print('总词数：', np.sum(list(tokenizer.word_counts.values())))
        print('词频为1的词个数：', len([i for i in tokenizer.word_counts.values() if i == 1]))
        print('词频为1的未登录词', len([i[0] for i in tokenizer.word_counts.items() if i[1] == 1 and i[0] not in domain_embedding]))
        print('未登录词', len([i[0] for i in tokenizer.word_counts.items() if i[0] not in domain_embedding]))
    return padded_sequences, domain_embedding_matrix, tokenizer


def get_padded_sentence(token_lists, token2index, max_sequence_lenth):
    token_array = np.array([[token2index[token] for token in token_list.split() if token in token2index] for token_list
                            in token_lists]).reshape(1, -1)
    #     return token_array
    return pad_sequences(token_array, maxlen=max_sequence_lenth, truncating='post')


def read_embedding(filepath):
    tokenid2emb = {}
    with open(filepath, 'r')as f:
        lines = f.readlines()
        for line in lines:
            token_id, token_emb = line.split('\t')[0].lower(), np.asarray(line.split('\t')[1:])
            tokenid2emb[token_id] = token_emb
    return tokenid2emb


def dump_data(data_path, data):
    with open(data_path, 'wb')as f:
        pickle.dump(data, f)


question_id_df = pd.read_csv('data/question_id.csv')

raw_train_df = pd.read_csv('data/train.csv')
raw_test_df = pd.read_csv('data/test.csv')

raw_all_df = pd.concat([raw_train_df, raw_test_df])
qid2wid = {}
qid2cid = {}
for item in question_id_df.iterrows():
    qid = item[1][0]
    wid = item[1][1]
    cid = item[1][2]
    qid2cid[qid] = cid
    qid2wid[qid] = wid
word_max_length = max([len(i.split()) for i in list(qid2wid.values())])
char_max_length = max([len(i.split()) for i in list(qid2cid.values())])

print('词的最大长度{}'.format(word_max_length))
print('字的最大长度{}'.format(char_max_length))
# 有句子出现多次？
raw_all_df['word1'] = raw_all_df.qid1.apply(lambda x: qid2wid[x])
raw_all_df['word2'] = raw_all_df.qid2.apply(lambda x: qid2wid[x])
raw_all_df['char1'] = raw_all_df.qid1.apply(lambda x: qid2cid[x])
raw_all_df['char2'] = raw_all_df.qid2.apply(lambda x: qid2cid[x])

cid2emb = read_embedding('data/char_embedding.txt')
wid2emb = read_embedding('data/word_embedding.txt')

char_padded_sequences, char_embedding_matrix, char_tokenizer = data2sequence\
    (raw_all_df.char1.tolist()+raw_all_df.char2.tolist(),cid2emb,10000,57)

word_padded_sequences, word_embedding_matrix, word_tokenizer = data2sequence\
    (raw_all_df.word1.tolist()+raw_all_df.word2.tolist(),wid2emb,10000,43)
y = np.asarray(raw_all_df.label.tolist()[:20000],dtype='int32')
dump_data('data/tokenizer.pkl',[word_tokenizer, char_tokenizer])
dump_data('data/processed.pkl',[word_padded_sequences, char_padded_sequences, word_embedding_matrix, char_embedding_matrix,y])

skf = StratifiedKFold(5, shuffle=True, random_state=7)
index_list = []
for fold_id, (train_index, val_index) in enumerate(skf.split(y, y)):
    index_list.append([train_index,val_index])
with open('oof_index.pkl','wb')as f:
    pickle.dump(index_list,f)