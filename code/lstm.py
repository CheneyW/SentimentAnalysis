#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/4/9
"""
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import to_categorical
import yaml
import numpy as np
import jieba
import os
import re

DATA_PATH = os.path.join(os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, 'model', 'lstm')
n_dim = 300
n_exposures = 10
max_len = 100
batch_size = 32
n_epoch = 4


def load_file():
    def get_text(file_path):
        with open(file_path, 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')
        return re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)

    neg = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'sample.negative.txt'))]
    pos = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'sample.positive.txt'))]
    docs = np.concatenate((neg, pos))
    y = np.concatenate((np.zeros(len(neg)), np.ones(len(pos))))  # 0-反例 1-正例
    return docs, y


def word2vec_train(docs):
    model = Word2Vec(size=n_dim,  # 300 词向量的维度
                     min_count=n_exposures,  # 10 所有频数超过10的词语
                     window=7, workers=4)
    model.build_vocab(docs)  # input: list
    model.train(docs, epochs=model.epochs, total_examples=model.corpus_count)
    model.save(os.path.join(MODEL_PATH, 'w2v_model.pkl'))

    n_symbols, idx2vec, docs2idx = create_dictionaries(model, docs)
    return n_symbols, idx2vec, docs2idx


def create_dictionaries(model, docs):
    """
    :return:
    n_symbols   词汇表大小
    idx2vec     mapping: index->vector
    docs2idx    由词语对应的索引组成的句子
    """
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

    # mapping: word->index
    w2idx = {word: idx + 1 for idx, word in gensim_dict.items()}  # 频率小于10的词索引为0,(idx->word)=>(word->idx+1)

    # 将句子中的词替换为索引
    docs2idx = []
    for sentence in docs:
        new_txt = []
        for word in sentence:
            new_txt.append(w2idx[word]) if word in w2idx.keys() else new_txt.append(0)
        docs2idx.append(new_txt)
    # 句子长度统一为maxlen
    docs2idx = sequence.pad_sequences(docs2idx, maxlen=max_len)

    # mapping: index->vector
    n_symbols = len(w2idx) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    idx2vec = np.zeros((n_symbols, n_dim))  # 初始化 索引为0的词语，词向量全为0
    for word, index in w2idx.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        idx2vec[index, :] = model.wv[word]

    return n_symbols, idx2vec, docs2idx


def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test=None, y_test=None):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # Sequential: 多个网络层的线性堆叠
    # Embedding: 索引->固定长度的密集向量
    model.add(Embedding(output_dim=n_dim,  # 词向量的维度
                        input_dim=n_symbols,  # 词汇表大小
                        mask_zero=True,  # 把 0 看作为一个应该被遮蔽的特殊的 "padding" 值, 索引 0 不能被用于词汇表中
                        weights=[embedding_weights],
                        input_length=max_len))  # 输入序列的长度
    model.add(LSTM(activation="sigmoid",  # 选取激活函数
                   units=50))  # 输出维度
    model.add(Dropout(0.5))
    model.add(Dense(1))  # Dense=>全连接层,输出维度=1
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train...")  # batch_size=32
    model.fit(x_train, y_train,
              batch_size=batch_size,  # 每次梯度更新的样本数
              epochs=n_epoch,  # 训练在样本上迭代次数
              verbose=1)  # 显示进度条

    print('Save model...')
    yaml_string = model.to_yaml()
    with open(os.path.join(MODEL_PATH, 'lstm.yml'), 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights(os.path.join(MODEL_PATH, 'lstm.h5'))

    if x_test is not None and y_test is not None:
        print('Evaluate...')
        score = model.evaluate(x_test, y_test, batch_size=batch_size)
        print('Test score:', score)


def main():
    docs, y = load_file()
    n_symbols, idx2vec, docs2idx = word2vec_train(docs)
    print(idx2vec.shape)

    train_lstm(n_symbols, idx2vec, docs2idx, y)

    # x_train, x_test, y_train, y_test = train_test_split(docs2idx, y, test_size=0.2)
    # train_lstm(n_symbols, idx2vec, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
