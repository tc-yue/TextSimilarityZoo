# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 15:48
# @Author  : Tianchiyue
# @File    : data.py
# @Software: PyCharm Community Edition
import numpy as np
import pickle
from keras.backend.tensorflow_backend import set_session
import os
import random as rn
import tensorflow as tf
import keras.backend as K
import logging
import pandas as pd

def init_env(gpu_id):
    """
    设置gpuid
    :param gpu_id:字符串
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))
    

def rand_set():
    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = '7'
    np.random.seed(7)
    rn.seed(7)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(7)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    # logging.info('\t=======Init Over=======')


def load_data(data_path):
    with open(data_path, 'rb')as f:
        data = pickle.load(f)
    # logging.info('\t=======Data Loaded=======')
    return data

def dump_data(data_path, data):
    with open(data_path, 'wb')as f:
        pickle.dump(data,f)
    # logging.info('\t=======Data Dump=======')



def get_train_features(features):
    X_train=pd.DataFrame()
    for featurename in features:
        fea=pd.read_csv("../feature/"+featurename+'.csv')
        if 'glove' in featurename:
            fea.iloc[27355,0] = np.average(fea.iloc[:20000,0])
            fea.iloc[27355,2] = np.average(fea.iloc[:20000,2])
        X_train = pd.concat((X_train, fea), axis=1)
    return X_train



