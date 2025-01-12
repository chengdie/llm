import hashlib
import os
import random
import re
import torch
from utils.Vocab import Vocab
import requests
from torch.utils.data import TensorDataset, DataLoader
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB = dict()
DATA_HUB['time_machine'] = ( DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

"""   ----    load_corpus_time_machine   ----   """
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(download('time_machine', DATA_HUB, './data'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def read_wangfeng():
    file_path = './data/input.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub(r'[^\u4e00-\u9fa5]+', ' ', line).strip() for line in lines]

def tokenize(lines, token = 'word'):
    if token == 'word':
        return [ line.split() for line in lines ]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def load_corpus_time_machine(max_tokens=-1,token_type = 'word'):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # lines = read_wangfeng()
    tokens = tokenize(lines, token_type)
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
"""  -----      load_corpus_time_machine      -----     """

"""  -----      load_data_time_machine      -----     """
def load_data_time_machine(batch_size, num_steps,token_type,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens, token_type)
    return data_iter, data_iter.vocab
"""  -----      load_data_time_machine      -----     """


def load_array(features, labels, n_train, batch_size, shuffle = False):
    if n_train > len(features):
        raise ValueError("n_train cannot be greater than the number of features")
    n_train = min(n_train, len(features))
    dataset = TensorDataset(features[:n_train], labels[:n_train])
    train_iter = DataLoader(dataset, batch_size, shuffle=shuffle)
    return train_iter


def download(name, DATA_HUB, cache_dir=os.path.join('.', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

"""     ---- SeqDataLoader     -------       """
def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1)//num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_subseqs]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens, token_type ):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens,token_type)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
# my_seq = list(range(35))
# for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)

# for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)
"""     ---- SeqDataLoader     -------       """