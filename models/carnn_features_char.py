# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 20:43
# @Author  : Tianchiyue
# @File    : carnn_features_char.py
# @Software: PyCharm Community Edition

from keras.legacy.layers import Highway
from keras.layers import Bidirectional, CuDNNGRU, CuDNNLSTM, GlobalAvgPool1D, GlobalMaxPool1D
from keras.layers import Concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras import optimizers, regularizers, callbacks
from utils.data import *
from layers import f1, soft_attention_alignment, apply_multiple, submult
import sys
import argparse
import logging
from carnn_layers import *
from keras.layers import Average

carnn_config = {
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
    'rnn_output_size': 300,
    'dropout_rate': 0.2,
    'spatial_dropout_rate': 0.2,
    'embed_trainable': False,
    'dense_dim': 300
}


def carnn(embedding_matrix, config, compare_out_size=CARNN_COMPARE_LAYER_OUTSIZE, rnn_size=CARNN_RNN_SIZE,
          rnn_dropout=CARNN_AGGREATION_DROPOUT):
    q1 = Input(shape=(config['max_length'],),
               dtype='int32',
               name='q1_input')
    q2 = Input((config['max_length'],),
               dtype='int32',
               name='q2_input')
    activation = 'elu'
    compare_dim = 500
    compare_dropout = 0.2
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

    highway_encoder = TimeDistributed(Highway(activation='relu'))
    self_attention = SelfAttention(d_model=embedding_matrix.shape[1])

    q1_encoded = highway_encoder(q1_embed, )
    q2_encoded = highway_encoder(q2_embed, )

    s1_encoded = self_attention(q1, q1_encoded)
    s2_encoded = self_attention(q2, q2_encoded)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined1 = Concatenate()([q1_encoded, q2_aligned, interaction(q1_encoded, q2_aligned), ])
    q1_combined2 = Concatenate()([q2_aligned, q1_encoded, interaction(q1_encoded, q2_aligned), ])

    q2_combined1 = Concatenate()([q2_encoded, q1_aligned, interaction(q2_encoded, q1_aligned), ])
    q2_combined2 = Concatenate()([q1_aligned, q2_encoded, interaction(q2_encoded, q1_aligned), ])

    s1_combined1 = Concatenate()([q1_encoded, s1_encoded, interaction(q1_encoded, s1_encoded), ])
    s1_combined2 = Concatenate()([s1_encoded, q1_encoded, interaction(q1_encoded, s1_encoded), ])

    s2_combined1 = Concatenate()([q2_encoded, s2_encoded, interaction(q2_encoded, s2_encoded), ])
    s2_combined2 = Concatenate()([s2_encoded, q2_encoded, interaction(q2_encoded, s2_encoded), ])

    compare_layers_d = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_out_size, activation=activation),
        Dropout(compare_dropout),
    ]

    compare_layers_g = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_out_size, activation=activation),
        Dropout(compare_dropout),
    ]

    # NOTE these can be optimized
    q1_compare1 = time_distributed(q1_combined1, compare_layers_d)
    q1_compare2 = time_distributed(q1_combined2, compare_layers_d)
    q1_compare = Average()([q1_compare1, q1_compare2])

    q2_compare1 = time_distributed(q2_combined1, compare_layers_d)
    q2_compare2 = time_distributed(q2_combined2, compare_layers_d)
    q2_compare = Average()([q2_compare1, q2_compare2])

    s1_compare1 = time_distributed(s1_combined1, compare_layers_g)
    s1_compare2 = time_distributed(s1_combined2, compare_layers_g)
    s1_compare = Average()([s1_compare1, s1_compare2])

    s2_compare1 = time_distributed(s2_combined1, compare_layers_g)
    s2_compare2 = time_distributed(s2_combined2, compare_layers_g)
    s2_compare = Average()([s2_compare1, s2_compare2])

    # Aggregate
    q1_encoded = Concatenate()([q1_encoded, q1_compare, s1_compare])
    q2_encoded = Concatenate()([q2_encoded, q2_compare, s2_compare])

    aggreate_rnn = CuDNNGRU(rnn_size, return_sequences=True)
    q1_aggreated = aggreate_rnn(q1_encoded)
    q1_aggreated = Dropout(rnn_dropout)(q1_aggreated)
    q2_aggreated = aggreate_rnn(q2_encoded)
    q2_aggreated = Dropout(rnn_dropout)(q2_aggreated)

    # Pooling
    q1_rep = apply_multiple(q1_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(), ])
    q2_rep = apply_multiple(q2_aggreated, [GlobalAvgPool1D(), GlobalMaxPool1D(), ])

    q_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_rep, q2_rep])
    q_multi = Lambda(lambda x: x[0] * x[1])([q1_rep, q2_rep])

    feature_input = Input(shape=(config['feature_length'],))
    feature_dense = BatchNormalization()(feature_input)
    feature_dense = Dense(config['dense_dim'], activation='relu')(feature_dense)
    h_all1 = Concatenate()([q1_rep, q2_rep, q_diff, q_multi, feature_dense])
    h_all2 = Concatenate()([q2_rep, q1_rep, q_diff, q_multi, feature_dense])
    h_all1 = Dropout(0.5)(h_all1)
    h_all2 = Dropout(0.5)(h_all2)

    dense = Dense(256, activation='relu')

    h_all1 = dense(h_all1)
    h_all2 = dense(h_all2)
    h_all = Average()([h_all1, h_all2])
    predictions = Dense(1, activation='sigmoid')(h_all)
    model = Model(inputs=[q1, q2, feature_input], outputs=predictions)
    opt = optimizers.get(config['optimizer'])
    K.set_value(opt.lr, config['learning_rate'])
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[f1])
    return model


def cross_validation():
    model_name = 'carnn_all_features_char'
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
    config = carnn_config
    config['max_length'] = train_word_x1.shape[1]
    config['feature_length'] = features.shape[1]
    train_new = np.zeros((len(train_word_x1), 1))
    test_new = np.zeros((len(test_word_x1), 1))
    scores = []

    for fold_id, (train_index, test_index) in enumerate(index_list):
        rand_set()
        model = carnn(char_embedding_matrix, config)
        skf_x_train = [train_x[0][train_index], train_x[1][train_index], train_x[2][train_index]]
        skf_y_train = train_y[train_index]
        skf_x_test = [train_x[0][test_index], train_x[1][test_index], train_x[2][test_index]]
        skf_y_test = train_y[test_index]
        lr_decay = callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.5, patience=config['lr_decay_epoch'],
                                               min_lr=0.01 * config['learning_rate'], mode='max')
        es = callbacks.EarlyStopping(monitor='val_f1', patience=config['n_stop'], mode='max')
        mc = callbacks.ModelCheckpoint(saved_models_fp, monitor='val_f1', save_best_only=True, save_weights_only=True,
                                       mode='max')

        hist = model.fit(skf_x_train, skf_y_train, batch_size=config['batch_size'], epochs=config['epochs'],
                         validation_data=(skf_x_test, skf_y_test), callbacks=[lr_decay, es, mc])
        model.load_weights(saved_models_fp)
        valid_pred = model.predict(skf_x_test)
        train_new[test_index, 0:1] = valid_pred
        test_new += model.predict(test_x)
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
