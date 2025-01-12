from utils.dataPreProcess import *
from utils.funcdraw import *
from utils.Vocab import Vocab
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB = dict()
DATA_HUB['time_machine'] = ( DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(download('time_machine', DATA_HUB, './data'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token = 'word'):
    if token == 'word':
        return [ line.split() for line in lines ]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


lines = read_time_machine()
tokens = tokenize(lines)
tokens = [token for line in tokens for token in line]
"""构建一元词汇"""
unigram_tokens = tokens
unigram_vocab = Vocab(unigram_tokens)
unigram_freqs = [freq for token, freq in unigram_vocab._token_freqs]
"""构建二元词汇"""
bigram_tokens = [pair for pair in  zip(tokens[:-1], tokens[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_freqs = [freq for token, freq in bigram_vocab._token_freqs]
print(bigram_vocab._token_freqs[:10])
"""构建三元词汇"""
trigram_tokens = [triple for triple in zip(tokens[:-1], tokens[1:], tokens[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_freqs = [freq for token, freq in trigram_vocab._token_freqs]
print(trigram_vocab._token_freqs[:10])

xyAxis_draw_multi([unigram_freqs, bigram_freqs, trigram_freqs],linestyles=['-','--',':'],markers=['o','x','s'],labels=['unigram','bigram','trigram'])

# corpus, vocab = load_corpus_time_machine()
# print(len(corpus), len(vocab.idx_to_token))
# print(vocab._token_freqs[:10])

# lines = read_time_machine()
# tokens = tokenize(lines)
# vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])  # .item是动态视图，返回[ (key,value),()... ]
# print(vocab.token_to_idx)
# for i in [0,10]:
#     print('words:', tokens[i])
#     print('indices:', vocab[tokens[i]])