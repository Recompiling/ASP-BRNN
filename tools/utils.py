
import os
import json
import numpy as np
# from keras.utils import plot_model
from keras.models import load_model, model_from_json
from keras.optimizers import RMSprop
def idx2inputname(idx):
    return 'input_' + str(idx)

def idx2outputname(idx):
    return 'output_'+str(idx)

def idx2modelname(idx):
    return 'model_'+str(idx)

def idx2image(idx, params):
    return os.path.join(params['home_path'], 'image_%s.png' % idx)

def idx2metrics(idx, params):
    return os.path.join(params['home_path'], 'metrics_%s.txt' % idx)

def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except IOError:
        raise "ERROR: Unable to locate file {}".format(filename)

def load_embeddings(filename):
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise 'ERROR: Unable to locate file {}.'.format(filename)

# def plot_image(params, file_path=None, save_to=None):
#     def plot_single(acc, loss, val_acc, val_loss, save_to):
#         epochs = range(len(acc))

#         plt.plot(epochs, acc, 'b', label='Training acc')
#         plt.plot(epochs, val_acc, 'r', label='Validation acc')
#         plt.title('Training and validation accuracy')
#         plt.legend()

#         plt.figure()

#         plt.plot(epochs, loss, 'b', label='Training loss')
#         plt.plot(epochs, val_loss, 'r', label='Validation loss')
#         plt.title('Training and validation loss')
#         plt.legend()

#         plt.savefig(save_to)
#     if file_path is not None:
#         tacc, tloss, acc, loss = load_metrics(file_path)
#         plot_single(tacc, tloss, acc, loss, save_to)
#     else:
#         for idx in range(params['dataset_num']):
#             file_path = idx2metrics(idx, params)
#             save_to = idx2image(idx, params)
#             tacc, tloss, acc, loss = load_metrics(file_path)
#             plot_single(tacc, tloss, acc, loss, save_to)

def save_metrics(train_acc_list, train_loss_list, acc_list, loss_list, params):
    for idx in range(params['dataset_num']):
        slice_list = lambda lt:lt[idx::params['dataset_num']]
        file_path = idx2metrics(idx, params)
        file = open(file_path, 'w')
        zp = zip(slice_list(train_acc_list), slice_list(train_loss_list), slice_list(acc_list), slice_list(loss_list))
        for tacc, tloss, acc, loss in zp:
            file.write("{}\t{}\t{}\t{}\n".format(tacc, tloss, acc, loss))
        file.close()

def load_metrics(path):
    ret = [[],[],[],[]]
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line)!=4:
                print('line', line)
                print('len(line): ', len(line))
                raise RuntimeError("illeagal metrics format")
            for idx, val in enumerate(line):
                ret[idx].append(float(val))
    return ret

def print_info(acc_list, loss_list, model_num, train=True):
    if train:
        line = '----------------------------------'
        word = 'Training'
    else:
        line = '=================================='
        word = 'Testing'
    print(line, '\n', word)
    for i in range(model_num):
        print('model_%s:' % i)
        print('\tloss: %f\taccuracy: %f'% (loss_list[i], acc_list[i]))
    print(line)

def save_models(single_models, index, path):
    '''Save a model to a .h5 file
    single_models: a dict {model_name1: model1, model_name2: model2, ...}
    index: which model to save in the dict
    path: save path without model name
    '''
    model_name = idx2modelname(index)
    model = single_models[model_name]
    model.save(os.path.join(path, "{}.h5".format(model_name)))

def load_models(model_name, path):
    '''Load a model from a .h5 file
    model_name: model name without .h suffix
    path: the directory the model file exists
    '''
    file_path = os.path.join(path, "{}.h5".format(model_name))
    model = load_model(file_path)

    return model

# def save_predictions(single_models, x_tests, file_path):
#     predictions = []

#     for index in range(len(single_models)):
#         model = single_models[idx2modelname(index)]
#         x = x_tests[idx2inputname(index)]

#         predict_y = model.predict(x)
#         predictions.append(predict_y)

#     with open(file_path + 'predictions_'+'.dat', 'wb') as f:
#         pickle.dump(predictions, f)