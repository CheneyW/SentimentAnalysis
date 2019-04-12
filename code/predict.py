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

DATA_PATH = os.path.join(os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, 'model', 'lstm')
n_dim = 300
max_len = 100


def load_file():
    def get_text(file_path):
        with open(file_path, 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')
        return re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)

    neg = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'sample.negative.txt'))]
    pos = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'sample.positive.txt'))]
    return neg, pos


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
    print(docs2idx)
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
    texts = []
    texts.append('本来奔着卓越想买这本书的，但是这个折扣，真的太不厚道啦')
    texts.append('真的很不错 我一般买完不会回来评分 但这次我要向大家推荐下 真的很不错 以后我会一直支持里仁的 产品 大家放心买吧 画面超清晰 有中英问字幕 一二季我都买了 质量很好')
    texts.append('这个版本是D版的！而且是英文的！')
    texts.append('就是一个动作接一个动作，没完没了。开始没有预备，结束没有放松。如果是初学者还是跟着班跟着老师做的好。')
    texts.append('我买了人声低音炮1和2，感觉录音很讨巧，第一下容易抓住人的注意力，但是不耐听。设备越好，越不好听，完全失去了均衡。')
    texts.append('还不错，得多跟着练才能跟的上~~')
    texts.append('作为一本指导性质的编程类书籍，这本书将各种设计模式演绎的淋漓尽致，通俗易懂，看后使人获益匪浅！设计模式只是开始，要真正的会用、用好还是需要实际的磨练。')
    texts.append('不知道拨号数字大不大，话筒音量大不大。。。。')
    texts.append('神秘园的曲子总是能打动每一个人心中最真的地方，太好听了~')
    texts.append(
        '上次买五月天的专辑和几本书，我特别说明了要包装牢固，joyo用了一个箱子，我很感动。这次也是cd和几本书，也是有包装的特别说明，结果竟然只有胶带，气死我了！还好，没有破损，要不然对不起布兰妮啊！joyo的服务怎么每况愈下呢？')
    texts = [jieba.lcut(x) for x in texts]
    print(lstm_predict(texts))

    # neg, pos = load_file()
    # result1 = lstm_predict(neg)
    # result2 = lstm_predict(pos)
    # print(result1)
    # print(result2)
    #
    # print(len(neg))
    # print(sum(result1))
    # print(sum(result2))
