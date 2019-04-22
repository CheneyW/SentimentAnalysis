#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/4/8
"""

from keras.models import model_from_yaml
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import numpy as np
import jieba
import yaml
import os
import re
import csv

DATA_PATH = os.path.join(os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, 'model','lstm')
n_dim = 300
max_len = 100


def load_test_file():
    def get_text(file_path):
        with open(file_path, 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')
        return re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)

    test = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'test.txt'))]
    return test


def input_transform(docs):
    w2v_model = Word2Vec.load(os.path.join(MODEL_PATH, 'w2v_model.pkl'))
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(w2v_model.wv.vocab.keys(), allow_update=True)
    w2idx = {word: idx + 1 for idx, word in gensim_dict.items()}  # mapping: word->index

    docs2idx = []
    for sentence in docs:
        new_txt = []
        for word in sentence:
            new_txt.append(w2idx[word]) if word in w2idx.keys() else new_txt.append(0)
        docs2idx.append(new_txt)
    # 句子长度统一为maxlen
    docs2idx = sequence.pad_sequences(docs2idx, maxlen=max_len)
    # print(docs2idx)
    return docs2idx


def lstm_predict(texts):
    print('loading lstm model......')
    with open(os.path.join(MODEL_PATH, 'lstm.yml'), 'r') as f:
        yaml_string = yaml.load(f, Loader=yaml.FullLoader)
    lstm_model = model_from_yaml(yaml_string)

    print('loading weights......')
    lstm_model.load_weights(os.path.join(MODEL_PATH, 'lstm.h5'))
    lstm_model.compile(loss='categorical_crossentropy',
                       optimizer='adam', metrics=['accuracy'])

    result = lstm_model.predict_classes(input_transform(texts))
    return result


if __name__ == '__main__':

    test = load_test_file()
    result = lstm_predict(test)
    print(result)
    print(len(result[0]))
    with open(os.path.join(DATA_PATH, 'result.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(result)):
            writer.writerow([i, result[i][0]])
