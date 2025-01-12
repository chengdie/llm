import torch
import os
import random

from util.Vocab import Vocab
from util.data_load import download_extract

DATA_HUB = dict()
DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4


#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    """  将一行多个句子分割，最终表示 -->[  ["  " , "   "]
                                       ["  ", "   "]   ]  """
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        # random.choice(iterable) 的作用是从一个可迭代对象中随机选择一个元素。paragraphs.shape-> [ [[],[]], [[],[]] ]
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    # round(...)：将计算结果四舍五入为整数
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    """ mlm_input_tokens 的shape: ['cls',...,' ']  
        pred_positions_and_labels的shape: [(2,word1),(4,word2)]  """
    # 按预测位置升序排序
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
    """ vocab[mlm_input_tokens] 的shape: [4,30,80,...,90] 
                 pred_positions 的shape: [3,6,9,...] 
          vocab[mlm_pred_labels]的shape: [30,50,...,]        """

#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,is_next) in examples:
        # 加 <pad> 填充
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                                            dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)),
                                            dtype=torch.long))
        # dtype=torch.long 加了这个会将 True or False 变为 1 or 0
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
        """  all_token_ids[i]的shape:   tensor([30,48,58,...,30,49,...,1,1,1,...,1])           LENGTH = max_len 这里的'<pad>'表示为数字1
             all_segments[i]的shape:    tensor([0,0,...,0,1,1,...,1,0,0,...0,0])               LENGTH = max_len 
             valid_lens[i]的shape:      tensor( 句子a和句子b的真实长度 )                           LENGTH = 1(a+b的句子的真实长度)
             all_pred_positions[i]:     tensor([2,4,7,...,0,0,...,0,0])                        LENGTH = max_num_mlm_preds(最大句子的长度的15%)
             all_mlm_weights[i]的shape: tensor([1,1,...,1,0,...,0,0])                          LENGTH = max_num_mlm_preds(最大句子的长度的15%)
             all_mlm_labels[i]的shape:  tensor([40,59,30,...,59,0,0,...,0])                    LENGTH = max_num_mlm_preds(最大句子的长度的15%)
             nsp_labels[i]的shape:      tensor(1 or 0)                                  LENGTH = 1(b是否真是a的下一句)  """
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        """"  处理后的paragraphs :
            [
                [['this', 'is', 'a', 'sentence'], ['another', 'sentence']],
                [['this', 'is', 'a', 'paragraph'], ['it', 'has', 'multiple', 'sentences']]
            ]
        """
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        """ 处理后的sentences:
            [   ['this', 'is', 'a', 'sentence'], 
                ['another', 'sentence'], 
                ['this', 'is', 'a', 'paragraph'], 
                ['it', 'has', 'multiple', 'sentences'] ]
        """
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            # _get_nsp_data_from_paragraph return: [(tokens,segments,isnext),...,]  length = len(paragraph)-1
            #extend 方法用于将一个可迭代对象的所有元素逐个添加到列表的末尾, 与 append 不同，extend 不会将整个对象作为一个元素添加，而是将其展开
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
            """examples的shape: [ (tokens,segments,isnext),(tokens,segments,isnext),...,]
                tokens:['cls',' ' ,...,'sep',' ',...,'sep'] 
              segments:[0,0,0,0,1,1,1,1,1,]
               isnext: True or False                         """
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                     for tokens, segments, is_next in examples]
        """ examples[i] 的shape:(
                                vocab[mlm_input_tokens] 的shape: [4,30,80,...,90] 
                                         pred_positions 的shape: [3,6,9,...] 
                                  vocab[mlm_pred_labels]的shape: [30,50,...,]  
                                                        segments:[0,0,0,0,1,1,1,1,1,]
                                                         isnext: True or False          )"""
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

#@save
def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    data_dir = './data/wikitext-2'
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,shuffle=True)
    return train_iter, train_set.vocab


""" 测试用例"""
# batch_size, max_len = 512, 64
# train_iter, vocab = load_data_wiki(batch_size, max_len)
#
# for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
#      mlm_Y, nsp_y) in train_iter:
#     print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
#           pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
#           nsp_y.shape)
#     break