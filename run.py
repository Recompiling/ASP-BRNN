import os
import sys
import random
import datetime
import numpy as np
import uuid
import time
import json
import copy

from keras.utils import plot_model
from models import Models
from tools.utils import *
from config import Config

from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
from dataset import load_data

def collect_data(tasks):
    '''Assemble data for multi-task learning'''
    x_trains, y_trains, x_tests, y_tests = dict(), dict(), dict(), dict()
    for idx, task in enumerate(tasks):
        in_name = idx2inputname(idx)
        out_name = idx2outputname(idx)
        x_trains[in_name] = task.x_train
        y_trains[out_name] = task.y_train
        x_tests[in_name] = task.x_test
        y_tests[out_name] = np.asarray(task.y_test)
    
    return x_trains, y_trains, x_tests, y_tests



def joint_train(tasks, index_embedding, params, resume_path=None):
    start = datetime.datetime.now()

    x_trains, y_trains, x_tests, y_tests = collect_data(tasks)
    model = Models(params, tasks)
    if resume_path is None:
        mtl_model, single_models = model.get_model('aspbrnn')
    else:
        mtl_model = load_models(idx2modelname(-1), resume_path)
        single_models = []
        for idx in range(len(tasks)):
            single_models.append(load_models(idx2modelname(idx), resume_path))
    print(mtl_model.summary())
    itera = 0
    batch_input = {}
    batch_output = {}
    batch_size = params['batch_size']
    iterations = params['iterations']
    dataset_num = len(tasks)
    # iter_per_epoch = int(12500 / batch_size)
    iter_per_epoch = params['iter_per_epoch']
    print('total iterations: {}'.format(iterations))

    acc_list, loss_list, train_acc_list, train_loss_list = list(), list(), list(), list()
    ss = time.time()
    maxacc = np.zeros(dataset_num)
    while (itera < params['iterations']):
        generate_batch_data_all(batch_input, batch_output, batch_size, x_trains, y_trains)
        train_acc_loss = mtl_model.train_on_batch(batch_input, batch_output)
        #print(train_acc_loss)

        itera += 1
        if itera%50 == 0:
            print(time.time()-ss)
            print('current iteration: {}'.format(itera))
            if len(train_acc_loss) != dataset_num*2:
                train_acc_loss = train_acc_loss[1:]
            print_info(train_acc_loss[dataset_num:dataset_num*2], train_acc_loss[0:dataset_num], dataset_num, True)
            ss = time.time()
        if (itera > 0 and itera % iter_per_epoch == 0):
            print('current iteration: {}'.format(itera))
            # evaluate(single_models, x_trains, y_trains, 'train')
            loss, acc = evaluate_all(single_models, x_tests, y_tests)
            for dataset_idx in range(dataset_num):
                if acc[dataset_idx] > maxacc[dataset_idx]:
                    maxacc[dataset_idx] = acc[dataset_idx]
                    save_models(single_models, dataset_idx, params['save_model_path'])
            if itera % 500 == 0:
                save_models({idx2modelname(-1): mtl_model}, -1, params['save_model_path'])
            acc_list.extend(acc)
            loss_list.extend(loss)
            train_loss_list.extend(train_acc_loss[1:1+dataset_num])
            train_acc_list.extend(train_acc_loss[1+dataset_num:2+dataset_num])
            # Save models and metrics
            save_metrics(train_acc_list, train_loss_list, acc_list, loss_list, params)
            # save_predictions(single_models, x_tests, params['prediction_path'])

    end = datetime.datetime.now()
    sys.stdout.write('\nused time: {}\n'.format(end - start))



def fine_tune(processed_datasets, index, params, itera_num, itera_every=100, experiments_path=None):
    model_name = idx2modelname(index)
    model = load_models(model_name, experiments_path)
    input_name = idx2inputname(index)
    output_name = idx2outputname(index)
    processed_datasets[::2] = list(map(lambda x:x[input_name], processed_datasets[::2]))
    processed_datasets[1::2] = list(map(lambda x:x[output_name], processed_datasets[1::2]))
    x_trains, y_trains, x_tests, y_tests = processed_datasets
    itera = 0
    start_time = time.time()
    while(itera < itera_num):
        itera += 1
        batch_input, batch_output = generate_batch_data(x_trains, y_trains, params['batch_size'])
        train_acc_loss = model.train_on_batch(batch_input, batch_output)
        if itera % 10 == 0:
            print('current itera: %s,\tused time: %s\n'%(str(itera), str(time.time()-start_time)))
            print(train_acc_loss)
            print_info([train_acc_loss[1]], [train_acc_loss[0]], 1, True)
            start_time = time.time()

        if itera!=0 and itera % itera_every==0:
            evaluate_single(model, x_tests, y_tests, index)


def generate_batch_data_all(batch_input, batch_output, batch_size, x_trains, y_trains):
    batch_input.clear()
    batch_output.clear()

    for (in_name, x_train), (out_name, y_train) in zip(x_trains.items(), y_trains.items()):
        assert (in_name[-1] == out_name[-1])
        batch_input[in_name], batch_output[out_name] = generate_batch_data(x_train, y_train, batch_size)

def generate_batch_data(x_train, y_train, batch_size):
    indices = np.random.choice(len(x_train), batch_size, replace=False)
    return np.asarray(x_train)[indices], np.asarray(y_train)[indices]

def evaluate_all(single_models, X, Y):
    acclist = list()
    losslist = list()
    for index in range(len(single_models)):
        x = X[idx2inputname(index)]
        y = Y[idx2outputname(index)]
        model = single_models[idx2modelname(index)]

        loss, acc = evaluate_single(model, x, y, index, to_print=False)
        acclist.append(acc)
        losslist.append(loss)
    print_info(acclist, losslist, len(single_models), False)
    return losslist, acclist

def evaluate_single(model, X, Y, index, to_print=True):
    '''already prossessed x and y'''
    loss, acc = model.evaluate(X, Y, verbose=0)
    if to_print:
        print('==================================')
        print('model_{}:'.format(index))
        print('\tloss: {}, accuracy: {}'.format(loss, acc))
        print('==================================')
    return loss, acc

def check_folder(path):
    # mkdir path
    if not os.path.exists(path):
        os.makedirs(path)

def save_config(param):
    params = copy.deepcopy(param)
    file_path = params['home_path']+'params.txt'
    params = json.dumps(params, indent=4)
    with open(file_path, 'w') as f:
        f.write(params)
    print(params)

def prepro_bow(name, filename, encoding='utf-8'):
    print('Process {} dataset...'.format(name))
    def link_words(samples):
        '''link sentence back
        return x, y
        '''
        x, y = list(), list()
        # line: (sentence, label)
        for line in samples:
            x.append(' '.join(line[0]))
            y.append(line[1])
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(x).toarray()
        return x, y

    # load dataset
    train_set, _ = load_data(os.path.join('dataset', 'raw', name, filename+'.task.train'), encoding=encoding)
    #dev_set, _ = load_data(os.path.join('dataset', 'raw', 'books', 'books.task.dev'))
    test_set, _ = load_data(os.path.join('dataset', 'raw', name, filename+'.task.test'), encoding=encoding)
    #train_set.extend(dev_set)
    train_len = len(train_set)
    train_set.extend(test_set)
    x_train, y_train = link_words(train_set)
    x_test, y_test = x_train[train_len:], y_train[train_len:]
    x_train, y_train = x_train[:train_len], y_train[:train_len]
    return x_train, y_train, x_test, y_test


def single_task(task, model_name):
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

    x_train, y_train, x_test, y_test = prepro_bow(task, task)

    clf.fit(x_train, y_train)
    #cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=10)

    print('Accuracy: {}'.format(score(clf.predict(x_test), y_test)))




def main(PARAMS):

    single_task('books', 'Bayes')
    exit(0)
    ################### Group 1 #################################
    # sst1 = Config('sst1', max_len=100, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    # sst2 = Config('sst2', max_len=100, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    # subj = Config('subj', max_len=100, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    # imdb = Config('imdb', max_len=400, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])

    # tasks = [sst1, sst2, subj, imdb]
    
    ################### Group 2 #################################
    # dvds = Config('dvds', max_len = 300, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    # books = Config('books', max_len = 300, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    # electronics = Config('electronics', max_len = 300, loss_weight=0.25, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    # kitchen = Config('kitchen', max_len = 300, loss_weight=0.25, regularizers=PARAMS['regularizers'])

    # tasks = [dvds, books, electronics, kitchen]

    ################### Group 3 #################################
    rn = Config('rn', max_len = 300, loss_weight=0.3333, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    qc = Config('qc', loss_weight=0.3333, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])
    imdb = Config('imdb', max_len = 400, loss_weight=0.3333, regularizers=PARAMS['regularizers'], lstm_in_dropout=PARAMS['lstm_in_dropout'], lstm_dropout=PARAMS['lstm_dropout'], fc_dropout=PARAMS['fc_dropout'])

    tasks = [rn, qc, imdb]

    #plot_image(None, 'experiments/593c0911f68640b78b10de14d02c7242/metrics_0.txt', 'experiments/593c0911f68640b78b10de14d02c7242/1.png')
    #exit(0)
    data_path = './data/'
    #home_path = 'experiment/'
    home_path = 'experiments/'+str(uuid.uuid4().hex)+'/'
    check_folder('experiments')
    check_folder(home_path)


    params = {
                'home_path': home_path,
                'lstm_output_dim': int(PARAMS['lstm_output_dim']),
                'bi_lstm_output_dim': int(PARAMS['bi_lstm_output_dim']),
                'final_output_dim': int(PARAMS['final_output_dim']),
                'dataset_num': len(tasks),

                'lstm_dropout': PARAMS['sh_lstm_in_dropout'],
                'bi_lstm_dropout': PARAMS['sh_lstm_dropout'],
                #25000 all
                'batch_size': PARAMS['batch_size'],
                'lr': PARAMS['lr'],
                'iterations': 6,#2000
                'iter_per_epoch': 2, 
                # how many words
                'num_words': Config.word_num,
                'save_model_path': home_path + 'model_folder/',
                'prediction_path': home_path + 'preds_folder/',
                'plot_image_path': home_path
            }
    print(params)
    check_folder(params['save_model_path'])
    check_folder(params['prediction_path'])
    save_config(params)
    joint_train(tasks, Config.word_emb, params)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'regularizers': 0.0003786999404939367,
        'lstm_in_dropout':0.67,
        'lstm_dropout':0.45,
        'sh_lstm_in_dropout':0.76,
        'sh_lstm_dropout':0.04,
        'fc_dropout':0.23,
        'lstm_output_dim':100,
        'bi_lstm_output_dim':100,
        'final_output_dim':100,
        'lr': 0.001,
        'batch_size': 64

    }


if __name__ == '__main__':
    #plot_image({'dataset_num':4, 'home_path':'experiments/217deec9101041f492c9397f54913935'})
    #exit(0)
    try:
        # get parameters from tuner
        # RECEIVED_PARAMS = nni.get_next_parameter()
        # LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        #PARAMS.update(RECEIVED_PARAMS)
        # train
        main(PARAMS)
    except Exception as e:
        print(e)
        raise