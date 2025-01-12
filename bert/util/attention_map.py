import torch
import torch.nn.functional as F

def vector_attention_map(v_q, v_k):
    dim = v_q.size(-1)
    print(torch.matmul(v_k, v_q.transpose(-1, -2)/torch.tensor(dim, dtype=torch.float32)))
    # attention_score = F.softmax(torch.matmul(v_k, v_q.transpose(-1, -2)/torch.sqrt(torch.tensor(dim, dtype=torch.float32))), dim=0)
    attention_score = F.softmax(
        torch.matmul(v_k, v_q.transpose(-1, -2))/torch.sqrt(torch.tensor(dim, dtype=torch.float32)),
        dim=0)
    return attention_score


