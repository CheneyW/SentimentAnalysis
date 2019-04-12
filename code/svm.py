#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/4/8
"""
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.svm import SVC
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import os
import re

DATA_PATH = os.path.join(os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, 'model', 'svm')

n_dim = 300


def load_file():
    def get_text(file_path):
        with open(file_path, 'r', encoding='utf-8')as f:
            text = f.read().replace('\n', '')
        return re.findall(r'<review id="\d+">(.+?)</review>', text, flags=re.S)

    neg = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'sample.negative.txt'))]
    pos = [jieba.lcut(x) for x in get_text(os.path.join(DATA_PATH, 'sample.positive.txt'))]

    # 1-正例   0-反例
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((neg, pos)), y, test_size=0.2)

    np.save(os.path.join(DATA_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_PATH, 'y_test.npy'), y_test)
    return x_train, x_test


# 计算词向量
def get_vecs(x_train, x_test):
    w2v = Word2Vec(size=n_dim, min_count=1, window=7, workers=4)

    w2v.build_vocab(x_train)

    w2v.train(x_train, epochs=w2v.epochs, total_examples=w2v.corpus_count)
    train_vecs = np.concatenate([text2vec(text, w2v) for text in x_train])

    w2v.train(x_test, epochs=w2v.epochs, total_examples=w2v.corpus_count)
    test_vecs = np.concatenate([text2vec(text, w2v) for text in x_test])

    w2v.save(os.path.join(MODEL_PATH, 'w2v_model.pkl'))
    np.save(os.path.join(MODEL_PATH, 'train_vecs.npy'), train_vecs)
    np.save(os.path.join(MODEL_PATH, 'test_vecs.npy'), test_vecs)


# 对每个句子的所有词向量取均值
def text2vec(text, w2v):
    vec = np.zeros(n_dim).reshape(1, n_dim)
    count = 0
    for word in text:
        try:
            vec += w2v.wv.__getitem__(word).reshape((1, n_dim))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 训练svm模型
def svm_train():
    # 导入数据
    train_vecs = np.load(os.path.join(MODEL_PATH, 'train_vecs.npy'))
    test_vecs = np.load(os.path.join(MODEL_PATH, 'test_vecs.npy'))
    y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, os.path.join(MODEL_PATH, 'svm_model.pkl'))
    print(clf.score(test_vecs, y_test))


def main():
    x_train, x_test = load_file()
    print("word2vec...")
    get_vecs(x_train, x_test)

    print("svm train...")
    svm_train()  # 训练svm并保存模型


def svm_predict(s):
    words = jieba.lcut(s)
    w2v = Word2Vec.load(os.path.join(SVM_MODEL_PATH, 'w2v_model.pkl'))
    temp = w2v.wv.__getitem__('好')
    print(type(temp))
    print(len(temp))
    print(temp)
    words_vecs = text2vec(words, w2v)

    clf = joblib.load(os.path.join(SVM_MODEL_PATH, 'svm_model.pkl'))
    result = clf.predict(words_vecs)
    if int(result[0]) == 1:
        print(s, ' positive')
    else:
        print(s, ' negative')


if __name__ == '__main__':
    main()

    s = "哈哈哈"
    svm_predict(s)
