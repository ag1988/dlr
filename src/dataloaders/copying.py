import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.dataset import IterableDataset
import numpy as np

from src.utils import distributed


def torch_copying_data(L, M, A, variable=False, batch_shape=()):
    """
        L: number of noise tokens
        M: number of memorization tokens
        A: size of dictionary
    """
    # M toks in {1..A-2}.  0: pad    A-1: M rightmost pos
    tokens = torch.randint(low=1, high=A-1, size=batch_shape+(M,))
    # pick M pos among first M+L for the above tokens
    if variable:
        # generator = torch.Generator().manual_seed(0)
        inds = torch.rand(batch_shape+(M+L,)).topk(M, dim=-1)[1].sort(-1)[0]
    else:
        inds = torch.arange(M).repeat(batch_shape+(1,))   # just need to right shift by M+L 
    # place tokens at inds in order
    x_toks = torch.zeros(batch_shape+(M+L,), dtype=torch.long)
    x_toks.scatter_(-1, inds, tokens)
    markers = (A-1) * torch.ones(batch_shape+(M,), dtype=torch.long)
    
    x_toks = torch.cat([x_toks, markers], dim=-1)  # (B M+L+M)
    x = F.one_hot(x_toks, A).float()
    y = tokens
    return x, y   # (B M+L+M A), (B M)


def copying_static_dataset(L, M, A, variable, samples):
    all_x, all_y = torch_copying_data(L, M, A, variable, batch_shape=(samples,))
    print("Constructing Copying dataset of shape", all_x.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds


class CopyingTrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, L, M, A, variable=False, samples_per_epoch=-1):
        super().__init__()
        self.L = L
        self.M = M
        self.A = A
        self.variable = variable
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self):
        if self.samples_per_epoch < 0:
            while True:
                x, y = torch_copying_data(self.L, self.M, self.A, self.variable)
                yield x, y
        else:
            for _ in range(self.samples_per_epoch):
                x, y = torch_copying_data(self.L, self.M, self.A, self.variable)
                yield x, y


class CopyingEvalDataset(torch.utils.data.TensorDataset):
    def __init__(self, L, M, A, variable, samples):
        self.L = L
        self.M = M
        self.A = A
        self.variable = variable
        self.samples = samples
        all_x, all_y = torch_copying_data(self.L, self.M, self.A, self.variable, batch_shape=(self.samples,))
        super().__init__(all_x, all_y)



if __name__ == '__main__':
    # a = torch_copying_data(20, 5, 10, batch_shape=(3,))
    # print(a)

    ds = CopyingTrainDataset(10, 5, 10, samples_per_epoch=5)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    for (x, y) in enumerate(loader):
        print(x, y)

    print("Copying Evaluation Dataset")
    # eval_ds = CopyingEvalDataset(10, 5, 10, samples=5)
    eval_ds = copying_static_dataset(10, 3, 10, variable=True, samples=5)
    loader = torch.utils.data.DataLoader(eval_ds, batch_size=2, num_workers=2)
    for (x, y) in loader:
        print(x, y)

