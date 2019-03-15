# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 12:57
# @Author  : Tianchiyue
# @File    : rnn_stacked_features_char.py
# @Software: PyCharm Community Edition


from keras.layers import Concatenate, GlobalAvgPool1D, Dropout, GlobalMaxPool1D
from keras.layers import concatenate, BatchNormalization, Activation, PReLU, add, \
    MaxPooling1D, SpatialDropout1D, Dense, Input, Embedding, Flatten, multiply, subtract, Bidirectional, GRU, LSTM, \
    CuDNNLSTM, CuDNNGRU, Lambda
import keras.backend as K
from keras import optimizers, regularizers, callbacks

from keras.models import Model
from utils.data import *
from layers import f1
import sys
import argparse
import logging

rnn_config = {
    'dense_dropout': 0.2,
    'optimizer': 'adam',
    'learning_rate': 0.0005,
    'lr_decay_epoch': 3,
    'n_stop': 5,
    'batch_size': 64,
    'weight_loss': False,
    'epochs': 20,
    'rnn': 'gru',
    'gpu': True,
    'rnn_output_size': 200,
    'dropout_rate': 0.2,
    'spatial_dropout_rate': 0.2,
    'embed_trainable': False,
    'dense_dim': 300
}


def rnn(embedding_matrix, config):
    if config['rnn'] == 'gru' and config['gpu']:
        encode = Bidirectional(CuDNNGRU(config['rnn_output_size'], return_sequences=True))
        encode2 = Bidirectional(CuDNNGRU(config['rnn_output_size'], return_sequences=True))
        encode3 = Bidirectional(CuDNNGRU(config['rnn_output_size'], return_sequences=True))
    else:
        encode = Bidirectional(CuDNNLSTM(config['rnn_output_size'], return_sequences=True))
        encode2 = Bidirectional(CuDNNLSTM(config['rnn_output_size'] * 2, return_sequences=True))
        encode3 = Bidirectional(CuDNNGRU(config['rnn_output_size'] * 4, return_sequences=True))

    q1 = Input(shape=(config['max_length'],),
               dtype='int32',
               name='q1_input')
    q2 = Input((config['max_length'],),
               dtype='int32',
               name='q2_input')
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                trainable=config['embed_trainable'],
                                weights=[embedding_matrix]
                                # mask_zero=True
                                )

    q1_embed = embedding_layer(q1)
    q2_embed = embedding_layer(q2)  # bsz, 1, emb_dims
    q1_embed = BatchNormalization(axis=2)(q1_embed)
    q2_embed = BatchNormalization(axis=2)(q2_embed)
    q1_embed = SpatialDropout1D(config['spatial_dropout_rate'])(q1_embed)
    q2_embed = SpatialDropout1D(config['spatial_dropout_rate'])(q2_embed)

    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)
    q1_encoded = Dropout(0.2)(q1_encoded)
    q2_encoded = Dropout(0.2)(q2_encoded)
    # 双向
    #     q1_encoded = encode2(q1_encoded)
    #     q2_encoded = encode2(q2_encoded)
    # resnet
    rnn_layer2_input1 = concatenate([q1_embed, q1_encoded])
    rnn_layer2_input2 = concatenate([q2_embed, q2_encoded])
    q1_encoded2 = encode2(rnn_layer2_input1)
    q2_encoded2 = encode2(rnn_layer2_input2)

    # add res shortcut
    res_block1 = add([q1_encoded, q1_encoded2])
    res_block2 = add([q2_encoded, q2_encoded2])
    rnn_layer3_input1 = concatenate([q1_embed, res_block1])
    rnn_layer3_input2 = concatenate([q2_embed, res_block2])
    #     rnn_layer3_input1 = concatenate([q1_embed,q1_encoded,q1_encoded2])
    #     rnn_layer3_input2 = concatenate([q2_embed,q2_encoded,q2_encoded2])
    q1_encoded3 = encode3(rnn_layer3_input1)
    q2_encoded3 = encode3(rnn_layer3_input2)
    #     merged1 = GlobalMaxPool1D()(q1_encoded3)
    #     merged2 = GlobalMaxPool1D()(q2_encoded3)
    #     q1_encoded = concatenate([q1_encoded, q1_encoded2], axis=-1)
    #     q2_encoded = concatenate([q2_encoded, q2_encoded2], axis=-1)

    #     merged1 = concatenate([q1_encoded2, q1_embed], axis=-1)
    #     merged2 = concatenate([q2_encoded2, q2_embed], axis=-1)
    #     # TODO add attention rep , maxpooling rep
    q1_encoded3 = concatenate([q1_encoded, q1_encoded2, q1_encoded3])
    q2_encoded3 = concatenate([q2_encoded, q2_encoded2, q2_encoded3])
    merged1 = GlobalMaxPool1D()(q1_encoded3)
    merged2 = GlobalMaxPool1D()(q2_encoded3)
    # avg1 = GlobalAvgPool1D()(q1_encoded3)
    # avg2 = GlobalAvgPool1D()(q2_encoded3)
    # merged1 = concatenate([max1,avg1])
    # merged2 = concatenate([max2,avg2])
    sub_rep = Lambda(lambda x: K.abs(x[0] - x[1]))([merged1, merged2])
    mul_rep = Lambda(lambda x: x[0] * x[1])([merged1, merged2])
    #     jaccard_rep = Lambda(lambda x: x[0]*x[1]/(K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True)-
    #                                               K.sum(K.abs(x[0]*x[1]),axis=1,keepdims=True)))([merged1,merged2])
    #     merged = Concatenate()([merged1, merged2, mul_rep, sub_rep,jaccard_rep])
    feature_input = Input(shape=(config['feature_length'],))
    feature_dense = BatchNormalization()(feature_input)
    feature_dense = Dense(config['dense_dim'], activation='relu')(feature_dense)


    merged = Concatenate()([merged1, merged2, mul_rep, sub_rep, feature_dense])
    # Classifier
    dense = Dropout(config['dense_dropout'])(merged)
    dense = BatchNormalization()(dense)
    dense = Dense(config['dense_dim'], activation='relu')(dense)
    dense = Dropout(config['dense_dropout'])(dense)
    dense = BatchNormalization()(dense)
    predictions = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[q1, q2, feature_input], outputs=predictions)
    opt = optimizers.get(config['optimizer'])
    K.set_value(opt.lr, config['learning_rate'])
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[f1])
    return model


def cross_validation():
    model_name = 'rnn_stacked_features_all_char'
    saved_models_fp = 'saved_models/{}.hdf5'.format(model_name)
    index_list = load_data('../data/oof_index.pkl')
    _, char_padded_sequences, _, char_embedding_matrix, train_y = \
        load_data('../data/processed.pkl')
    word_x1 = char_padded_sequences[:30000]
    word_x2 = char_padded_sequences[30000:]
    train_word_x1 = word_x1[:20000]
    train_word_x2 = word_x2[:20000]
    test_word_x1 = word_x1[20000:]
    test_word_x2 = word_x2[20000:]
    config = rnn_config
    config['max_length'] = train_word_x1.shape[1]
    feature_list = [
        'dongzhen_tfidf',
        'dongzhen_pca',
        "embedding_dis",
        "fuzzy_wuzzy",
        "long_common_string",
        "powerful_token",
        "common_token",
        "token_count",
        'expand_embedding_dis_glove_100',
        'expand_embedding_dis_glove_50',
        'expand_embedding_dis_w2v_100',
        'expand_embedding_dis_w2v_200'
    ]

    features = get_train_features(feature_list).values
    train_x = [train_word_x1, train_word_x2, features[:20000]]
    test_x = [test_word_x1, test_word_x2, features[20000:]]
    config['feature_length'] = features.shape[1]
    train_new = np.zeros((len(train_word_x1), 1))
    test_new = np.zeros((len(test_word_x1), 1))
    scores = []

    for fold_id, (train_index, test_index) in enumerate(index_list):
        rand_set()
        model = rnn(char_embedding_matrix, config)
        skf_x_train = [train_x[0][train_index], train_x[1][train_index], train_x[2][train_index]]
        skf_y_train = train_y[train_index]
        skf_x_test = [train_x[0][test_index], train_x[1][test_index], train_x[2][test_index]]
        skf_y_test = train_y[test_index]
        lr_decay = callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.5, patience=config['lr_decay_epoch'],
                                               min_lr=0.01 * config['learning_rate'], mode='max')
        es = callbacks.EarlyStopping(monitor='val_f1', patience=config['n_stop'], mode='max')
        mc = callbacks.ModelCheckpoint(saved_models_fp, monitor='val_f1', save_best_only=True,
                                       save_weights_only=True,
                                       mode='max')

        hist = model.fit(skf_x_train, skf_y_train, batch_size=config['batch_size'], epochs=config['epochs'],
                         validation_data=(skf_x_test, skf_y_test), callbacks=[lr_decay, es, mc])
        # 每次填充200*3，五次填充1000*3
        model.load_weights(saved_models_fp)
        valid_pred = model.predict(skf_x_test)
        train_new[test_index, 0:1] = valid_pred
        test_new += model.predict(test_x)  # 300*3
        best_score = max(hist.history['val_f1'])
        scores.append(best_score)
        del model
        if K.backend() == 'tensorflow':
            K.clear_session()
    logging.info('\t{}\t{}\t平均得分{}'.format(model_name, '|'.join([str(i) for i in np.around(scores, decimals=4)]),
                                           str(np.average(scores))))
    dump_data('../cv_result/{}.pkl'.format(model_name), [train_new, test_new])


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='no', help='before running watch nvidia-smi')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s%(message)s',
                        filename='cv.log',
                        filemode='a')
    if len(args.gpu) == 1:
        init_env(str(args.gpu))
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cross_validation()


if __name__ == '__main__':
    main(sys.argv)

