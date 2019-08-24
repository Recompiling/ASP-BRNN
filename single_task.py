import os
import sys
import random
import datetime
import numpy as np
import uuid
import time
import json
import copy

from models import Models
#from tools.utils import *
from config import Config

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
from dataset import load_data

def prepro_bow(name, filename, ngram, encoding='utf-8'):
    print('Process {} dataset...'.format(name))
    word_count = {}
    def link_words(samples):
        '''link sentence back
        return x, y
        '''
        x, y = list(), list()
        # line: (sentence, label)
        for line in samples:
            sentence = line[0]
            for w in sentence:
                if w in word_count:
                    word_count[w]+=1
                else:
                    word_count[w]=0
        for line in samples:
            sentence = []
            for w in line[0]:
                if word_count[w] >= 20:
                    sentence.append(w)
            x.append(' '.join(sentence))
            y.append(line[1])
        vectorizer = CountVectorizer(ngram_range=(1,ngram),min_df=20)
        print(sum([1 if word_count[i]>=20 else 0 for i in word_count]))
        x = vectorizer.fit_transform(x).toarray()
        print(len(vectorizer.get_feature_names()))
        return x, y

    # load dataset
    train_set, _ = load_data(os.path.join('dataset', 'raw', name, filename+'_train.txt'), encoding=encoding)
    #dev_set, _ = load_data(os.path.join('dataset', 'raw', 'books', 'books.task.dev'))
    test_set, _ = load_data(os.path.join('dataset', 'raw', name, filename+'_test.txt'), encoding=encoding)
    #train_set.extend(dev_set)
    train_len = len(train_set)
    train_set.extend(test_set)
    x_train, y_train = link_words(train_set)
    x_test, y_test = x_train[train_len:], y_train[train_len:]
    x_train, y_train = x_train[:train_len], y_train[:train_len]
    print('Process {} dataset done'.format(name))
    return x_train, y_train, x_test, y_test


def single_task(task, model_name, ngram=1):
    def score(y_pred, y):
        assert len(y_pred) == len(y), 'y_pred and y have different lenght'
        acc = 0
        for y1, y2 in zip(y_pred, y):
            if y1 == y2:
                acc += 1
        return float(acc) / len(y)
    if model_name == 'Bayes':
        clf = MultinomialNB()
    elif model_name == 'svm':
        clf = svm.SVC(gamma='scale')

    x_train, y_train, x_test, y_test = prepro_bow(task, task,ngram)
    #clf.fit(x_train, y_train)
    #print('fit done')

    ret = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=10)
    print(ret)
    with open('metric.txt', 'a') as f:
        f.write(str(ret)+'\n')

    #print('Accuracy: {}'.format(score(clf.predict(x_test), y_test)))

with open('metric.txt', 'w') as f:
    pass
single_task('imdb', 'Bayes', 1)
exit()
single_task('imdb', 'Bayes', 2)
single_task('imdb', 'Bayes', 3)
