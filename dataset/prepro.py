import os
import re
import sys
import copy
import json
import codecs
import operator
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

np.random.seed(1234)

source_dir = os.path.join('.', 'raw')
emb_dir = os.path.join('.', 'glove')
target_dir = os.path.join('.', 'data')

# special tokens
PAD = '__PAD__'
UNK = '__UNK__'

# global variables
datasets_params = list()

def text_to_wordlist(text, remove_stopwords=False, wnled_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if wnled_words:
        text = text.split()
        wnl = WordNetLemmatizer()
        wnled_words = [wnl.lemmatize(word) for word in text]
        text = " ".join(wnled_words)
    
    # Return a list of words
    return text

def load_data(filename, clean=True, encoding='utf-8', split_by=' ', process_label=None):
    """
    Read data from file into list of tuples
    returns: 
    [(sentence, label), ..., (sentence, label)], len()
    """
    dataset = []
    labels = set()
    with codecs.open(filename, 'r', encoding=encoding) as f:
        for line in f:
            if encoding is not 'utf-8':
                line = line.encode('utf-8').decode(encoding)  # convert string to utf-8 version
            line = text_to_wordlist(line, False, True)  # all the tokens and labels are default to be splitted by __BLANKSPACE__
            line = line.split(split_by)
            sentence = line[1:]
            try:
                if process_label is not None:
                    label = process_label(line[0])
                else:
                    label = int(line[0])
            except Exception as e:
                print(line[0])
                print(process_label)
                print(e)
                exit(0)
            labels.add(label)
            dataset.append((sentence, label))
    return dataset, len(labels)


def load_glove_vocab(filename):
    """Read word vocabulary from embeddings"""
    with open(filename, 'r', encoding='utf-8') as f:
        vocab = {line.strip().split()[0] for line in tqdm(f, desc='Loading GloVe vocabularies')}
    print('\t -- totally {} tokens in GloVe embeddings.\n'.format(len(vocab)))
    return vocab


def save_idx2embedding(vocab, glove_path, save_path, word_dim):
    """Prepare pre-trained word embeddings for dataset. Save idx_embedding dict to save_path"""
    embeddings = np.zeros([len(vocab), word_dim])  # embeddings[0] for PAD
    scale = np.sqrt(3.0 / word_dim)
    embeddings[1] = np.random.uniform(-scale, scale, [1, word_dim])  # for UNK
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Filtering GloVe embeddings'):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = np.asarray(embedding)
    sys.stdout.write('Saving filtered embeddings...')
    np.savez_compressed(save_path, embeddings=embeddings)
    sys.stdout.write(' done.\n')


def write_vocab(vocab, filename, threshold=0):
    """write vocabulary to file
    threshold: it is said in a sense of a whole vocabulary
    """
    sys.stdout.write('Writing vocab to {}...'.format(filename))
    global global_word_count
    with open(filename, 'w') as f:
        for i, word in enumerate(vocab):
            if global_word_count[word] > threshold:
                f.write('{}\n'.format(word)) if i < len(vocab) - 1 else f.write(word)
    sys.stdout.write(' done. Totally {} tokens.\n'.format(len(vocab)))


def load_vocab(filename):
    """read vocabulary from file into dict"""
    word_idx = {PAD:0, UNK:1}
    idx_word = {0:PAD, 1:UNK}
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            if word not in [PAD, UNK]:
                word_idx[word] = idx + 2
                idx_word[idx+2] = word
    return word_idx, idx_word


def build_vocab(datasets, threshold=0):
    """
    Build word vocabularies
    returns: word_vocab where count(word)>threshold
    """
    global global_word_count
    word_count = dict()
    for dataset in datasets:
        for words, _ in dataset:
            for word in words:
                word_count[word] = word_count.get(word, 0) + 1  # update word count in word dict
                global_word_count[word] = global_word_count.get(word, 0) + 1
    word_count = reversed(sorted(word_count.items(), key=operator.itemgetter(1)))
    word_vocab = set([w[0] for w in word_count if w[1] >= threshold])
    return word_vocab


def dump_to_json(dataset, filename):
    """Save built dataset into json"""
    if dataset is not None:
        with open(filename, 'w') as f:
            json.dump(dataset, f)
        sys.stdout.write('dump dataset to {}.\n'.format(filename))


def split_train_dev_test(dataset, dev_ratio=0.1, test_ratio=0.1, build_test=True, shuffle=True):
    """Split dataset into train, dev as well as test sets"""
    if shuffle:
        np.random.shuffle(dataset)
    data_size = len(dataset)
    if build_test:
        train_position = int(data_size * (1 - dev_ratio - test_ratio))
        dev_position = int(data_size * (1 - test_ratio))
        train_set = dataset[:train_position]
        dev_set = dataset[train_position:dev_position]
        test_set = dataset[dev_position:]
        return train_set, dev_set, test_set
    else:
        # dev_ratio = dev_ratio + test_ratio
        train_position = int(data_size * (1 - dev_ratio))
        train_set = dataset[:train_position]
        dev_set = dataset[train_position:]
        return train_set, dev_set, None


def build_dataset(raw_dataset, filename, word_vocab, num_labels, one_hot=True):
    """Convert dataset into word index, make labels to be one hot vectors and dump to json file"""
    dataset = []
    label_map = {}
    for sentence, label in raw_dataset:
        words = []
        if label not in label_map:
            label_map[label] = len(label_map)
        label = label_map[label]
        for word in sentence:
            words.append(word_vocab[word] if word in word_vocab else word_vocab[UNK])
        if one_hot:
            label = [1 if i == label else 0 for i in range(num_labels)]
        dataset.append({'sentence': words, 'label': label})
    dump_to_json(dataset, filename=filename)


def prepro_finalized(glove_path):
    global word_vocab
    write_vocab(word_vocab, filename=os.path.join(target_dir, 'words.vocab'), threshold=0)
    # build embeddings
    word_vocab, _ = load_vocab(os.path.join(target_dir, 'words.vocab'))
    print("len(word_vocab): ", len(word_vocab))
    save_idx2embedding(word_vocab, glove_path, os.path.join(target_dir, 'glove.filtered.npz'), word_dim=200)
    def _prepro_finalized(train_set, dev_set, test_set, num_labels, data_folder, glove_path=glove_path):
        '''Finalize preprocess: save metrics data'''

        build_dataset(train_set, os.path.join(data_folder, 'train.json'), word_vocab, num_labels=num_labels,
                    one_hot=True)
        build_dataset(dev_set, os.path.join(data_folder, 'dev.json'), word_vocab, num_labels=num_labels,
                    one_hot=True)
        build_dataset(test_set, os.path.join(data_folder, 'test.json'), word_vocab, num_labels=num_labels,
                    one_hot=True)
        # save number of labels information into json
        with open(os.path.join(data_folder, 'label.json'), 'w') as f:
            json.dump({"label_size": num_labels}, f)
    
    for param in datasets_params:
        _prepro_finalized(*param)


def prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path, threshold=0):
    """Performs to build vocabularies and processed dataset"""
    # build vocabularies
    current_word_vocab = build_vocab([train_set, dev_set], threshold)  # only process train and dev sets
    print(len(current_word_vocab))
    # ----------------------------------------
    global word_vocab
    word_vocab.update(current_word_vocab&glove_vocab)  # distinct vocab and add PAD and UNK tokens
    # Save other parameter to global varibale
    datasets_params.append([train_set, dev_set, test_set, num_labels, data_folder])


def prepro_sst(glove_path, glove_vocab, mode=1):
    print('Process sst{} dataset...'.format(mode))
    data_folder = os.path.join(target_dir, 'sst{}'.format(mode))
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # load dataset
    name = 'fine' if mode == 1 else 'binary'
    train_set, num_labels = load_data(os.path.join(source_dir, 'sst{}'.format(mode), 'stsa.{}.train'.format(name)))
    dev_set, _ = load_data(os.path.join(source_dir, 'sst{}'.format(mode), 'stsa.{}.dev'.format(name)))
    test_set, _ = load_data(os.path.join(source_dir, 'sst{}'.format(mode), 'stsa.{}.test'.format(name)))
    # build general
    prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path)
    print()


def prepro_imdb(glove_path, glove_vocab):
    print('Process imdb dataset...')
    data_folder = os.path.join(target_dir, 'imdb')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # load dataset
    train_set, num_labels = load_data(os.path.join(source_dir, 'imdb', 'imdb_train.txt'))
    test_set, _ = load_data(os.path.join(source_dir, 'imdb', 'imdb_test.txt'))
    train_set, dev_set, _ = split_train_dev_test(train_set, build_test=False)
    # build general
    prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path, 20)
    print()


def prepro_other(name, filename, glove_path, glove_vocab, threshold=0, encoding='utf-8', process_label=None):
    print('Process {} dataset...'.format(name))
    data_folder = os.path.join(target_dir, name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # load dataset
    train_set, num_labels = load_data(os.path.join(source_dir, name, filename + '.task.train'), encoding=encoding, process_label=process_label)
    test_set, _ = load_data(os.path.join(source_dir, name, filename + '.task.test'), encoding=encoding, process_label=process_label)
    train_set, dev_set, _ = split_train_dev_test(train_set, build_test=False)
    # build general
    prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path, threshold)
    print()

def main():
    glove_path = os.path.join(emb_dir, 'glove.6B.200d.txt')
    global word_vocab
    global global_word_count
    word_vocab = set()
    global_word_count = dict()
    glove_vocab = load_glove_vocab(glove_path)
    # # process sst1 dataset
    # prepro_sst(glove_path, glove_vocab, mode=1)
    # # process sst2 dataset
    # prepro_sst(glove_path, glove_vocab, mode=2)
    # # process subj dataset
    # prepro_other('subj', 'subj.all', glove_path, glove_vocab, encoding='windows-1252')
    # # process imdb dataset
    #prepro_imdb(glove_path, glove_vocab)

    threshold = 0
    # # process dvds
    # prepro_other('dvds', 'dvds', glove_path, glove_vocab, threshold, encoding='ISO-8859-1')
    # # process books
    # prepro_other('books', 'books', glove_path, glove_vocab, threshold)
    # # process electronics
    # prepro_other('electronics', 'electronics', glove_path, glove_vocab, threshold)
    # # process kitchen
    # prepro_other('kitchen', 'kitchen', glove_path, glove_vocab, threshold)
    # # process ending

    prepro_other('qc', 'qc', glove_path, glove_vocab, threshold, process_label=lambda x:x.split(':')[0])
    prepro_other('rn', 'rn', glove_path, glove_vocab, threshold)
    # process imdb dataset
    prepro_imdb(glove_path, glove_vocab)
    prepro_finalized(glove_path)
    print('Pre-processing all the datasets finished... data is located at {}'.format(target_dir))


if __name__ == '__main__':
    main()
