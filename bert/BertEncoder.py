import torch
from torch import nn

from EncoderBlock import EncoderBlock


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super().__init__(**kwargs)
        # 次元嵌入---单词嵌入
        # 将词汇表大小为vocab_size的词索引映射到维度为 num_hiddens 的词嵌入向量,处理后维度加一
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # 段嵌入
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            #add_module(name, module)
            self.blks.add_module(f'{i}',EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数,nn.Parameter 的张量会自动被添加到模型的参数列表中，并在训练过程中进行优化。
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


""" BERTEncoder 测试用例"""
# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
# encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
#                       ffn_num_hiddens, num_heads, num_layers, dropout)
# #生成一个形状为 (2, 9) 的随机整数张量，其中的每个元素都是从 0 到 vocab_size-1 之间的随机整数,这个张量通常用于模拟输入序列的词索引（token indices)
# tokens = torch.randint(0, vocab_size, (2, 9))
# segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1,1], [0, 0, 0, 1, 1, 1, 1, 1,1]])
# encoded_X = encoder(tokens, segments, None)
# print(encoded_X.shape)  --->[2,9,768]
