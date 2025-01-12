import math

from torch import nn
import torch

"""  X 被reshape成m*n的二维矩阵,方便sequence_mask（）处理；而 valid_lens是一维向量"""
# @save
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # 获取张量 X 的第 1 维度（即第二个维度）的大小
    maxlen = X.size(1)
    #torch.arange((maxlen)) 等价于 torch.arange(maxlen)，因为 (maxlen) 会被解包为 maxlen;返回一个[0, ... ,maxlen]的一维张量
    # [None, :]将生成的序列扩展为二维张量，形状为 (1, maxlen)
    #valid_len[:, None]:将 valid_len 扩展为二维张量，形状为 (batch_size, 1)
    """ 简易写法 """
    # mask = torch.arange((maxlen), dtype=torch.float32,
    #                     device=X.device)[None, :] < valid_len[:, None]
    """ 我的写法  """
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device).unsqueeze(0) < valid_len.unsqueeze(1)
    X[~mask] = value
    return X

""" sequence_mask测试例子 """
# X = torch.ones(3, 4)
# print(sequence_mask(X, torch.tensor([1, 2 ,3]), value=-1))


#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 会将 valid_lens 中的每个元素重复 shape[1] 次
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            #reshape(-1) 会将张量展平为一个一维的向量，无论原始张量的形状如何。
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        # X 被reshape成m*n的二维矩阵,方便sequence_mask（）处理；而 valid_lens是一维向量
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

""" masked_softmax测试例子 """
# X = torch.ones(3,3,4)
# valid_lens = torch.tensor([[1,2,3],[2,3,4],[1,2,3]])
# print(masked_softmax(X, valid_lens))

#@save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)# 形状为(batch*head, seq, seq)
        # 返回（batch_size,查询个数，维度）
        return torch.bmm(self.dropout(self.attention_weights), values)