import torch
from torch import cuda


def try_all_gpus(i=0):
    if cuda.device_count() >= i+1:
        return [torch.device(f'cuda:{i}')]
    else:
        return [torch.device('cpu')]