# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 12:56
# @Author  : Tianchiyue
# @File    : decom_highway_features_char.py
# @Software: PyCharm Community Edition



from keras.layers import Bidirectional, CuDNNGRU, CuDNNLSTM, GlobalAvgPool1D, GlobalMaxPool1D
from keras.layers import Concatenate, Dropout, BatchNormalization, Activation, PReLU, add, \
    MaxPooling1D, SpatialDropout1D, Dense, Input, Embedding, Flatten, multiply, subtract, Lambda
from keras.models import Model
from keras import optimizers, regularizers, callbacks
import keras.backend as K
from utils.data import *
from layers import f1, soft_attention_alignment, apply_multiple, submult, time_distributed
import sys
import argparse
import logging
from keras.legacy.layers import Highway
from keras.layers import TimeDistributed, Add, Subtract, Multiply

decom_config = {
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


def interaction(input_1, input_2):
    "Get the interaction then concatenate results"
    mult = Multiply()([input_1, input_2])
    add = Add()([input_1, input_2])
    sub = Subtract()([input_1, input_2])
    # distance = el_distance(input_1, input_2)

    out_ = Concatenate()([sub, mult, add, ])
    return out_


def decom(embedding_matrix, config):
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

    projection_hidden = 300
    activation = 'elu'
    projection_dropout = 0.2
    projection_dim = 300
    compare_dim = 500
    compare_dropout = 0.2
    projection_layers = []
    highway_encoder = TimeDistributed(Highway(activation='relu'))

    q1_encoded = highway_encoder(q1_embed, )
    q2_encoded = highway_encoder(q2_embed, )

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, interaction(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, interaction(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    sub_rep = Lambda(lambda x: K.abs(x[0] - x[1]))([q1_rep, q2_rep])
    mul_rep = Lambda(lambda x: x[0] * x[1])([q1_rep, q2_rep])

    # Classifier
    # Classifier
    feature_input = Input(shape=(config['feature_length'],))
    feature_dense = BatchNormalization()(feature_input)
    feature_dense = Dense(config['dense_dim'], activation='relu')(feature_dense)
    # merged = Concatenate()([merged1, merged2, mul_rep, sub_rep, feature_dense])
    merged = Concatenate()([q1_rep, q2_rep, sub_rep, mul_rep, feature_dense])
    # merged = Concatenate()([q1_rep, q2_rep, sub_rep, mul_rep])

    dense = BatchNormalization()(merged)
    dense = Dense(config['dense_dim'], activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(config['dense_dropout'])(dense)
    dense = Dense(config['dense_dim'], activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(config['dense_dropout'])(dense)
    predictions = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[q1, q2, feature_input], outputs=predictions)
    opt = optimizers.get(config['optimizer'])
    K.set_value(opt.lr, config['learning_rate'])
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[f1])
    return model


def cross_validation():
    model_name = 'decom_highway_features_all_char'
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
    config = decom_config
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
        model = decom(char_embedding_matrix, config)
        skf_x_train = [train_x[0][train_index], train_x[1][train_index], train_x[2][train_index]]  # 800,10
        skf_y_train = train_y[train_index]  # 800,3
        skf_x_test = [train_x[0][test_index], train_x[1][test_index], train_x[2][test_index]]  # 200,10
        skf_y_test = train_y[test_index]
        lr_decay = callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.5, patience=config['lr_decay_epoch'],
                                               min_lr=0.01 * config['learning_rate'], mode='max')
        es = callbacks.EarlyStopping(monitor='val_f1', patience=config['n_stop'], mode='max')
        mc = callbacks.ModelCheckpoint(saved_models_fp, monitor='val_f1',
                                       save_best_only=True, save_weights_only=True,
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

