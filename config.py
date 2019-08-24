import os
import numpy as np
from dataset import load_vocab
from tools import load_embeddings, load_json

from keras.preprocessing import text, sequence


class Config(object):
    word_vocab = None
    word_emb = None
    word_num = 0
    def __init__(self, task, max_len=None, loss_weight=0.25, regularizers=0.001, lstm_in_dropout=0.5, lstm_dropout=0.5, fc_dropout=0.5):
        source_dir = os.path.join('.', 'dataset', 'data', task)
        if Config.word_vocab is None:
            Config.word_vocab, _ = load_vocab(os.path.join(source_dir, '..', 'words.vocab'))
            Config.vocab_size = len(Config.word_vocab)
            Config.word_emb = load_embeddings(os.path.join(source_dir, '..', 'glove.filtered.npz'))
            Config.emb_len = len(Config.word_emb[0])
            Config.word_num = len(Config.word_emb)
        self.max_len = max_len
        self.loss_weight = loss_weight
        self.regularizers = regularizers
        self.lstm_in_dropout = lstm_in_dropout
        self.lstm_dropout = lstm_dropout
        self.fc_dropout = fc_dropout
        self.label_size = load_json(os.path.join(source_dir, 'label.json'))["label_size"]
        self.load_data(source_dir)

    def get_train_test_from_file(self, path):
        data = load_json(path)
        x = list()
        y = list()
        for line in data:
            x.append(line['sentence'])
            y.append(line['label'])
        if self.max_len is None:
            self.max_len = max(map(lambda x:len(x), x))
        x = sequence.pad_sequences(x, maxlen=self.max_len,
                                    padding='post', truncating='post')
        return x, y

    def load_data(self, source_dir):
        self.x_train, self.y_train = self.get_train_test_from_file(os.path.join(source_dir, 'train.json'))
        self.x_dev, self.y_dev = self.get_train_test_from_file(os.path.join(source_dir, 'dev.json'))
        self.x_test, self.y_test = self.get_train_test_from_file(os.path.join(source_dir, 'test.json'))
        self.x_train = np.append(self.x_train, self.x_dev, axis=0)
        self.y_train = np.append(self.y_train, self.y_dev, axis=0)



    # log and model file paths
    max_to_keep = 5  # max model to keep while training
    no_imprv_patience = 5

    # word embeddings
    use_word_emb = True
    finetune_emb = False
    word_dim = 200

    # model parameters
    num_layers = 15
    num_units = 13
    num_units_last = 100

    # hyperparameters
    lr = 0.01
    keep_prob = 0.5