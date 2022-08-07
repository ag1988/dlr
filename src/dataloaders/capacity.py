import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.dataset import IterableDataset
import numpy as np


def rconv(x, y, dim=-1):
    assert all(map(torch.is_floating_point, (x, y))), 'inputs must be real'
    n = x.shape[dim] + y.shape[dim]
    fx, fy = (torch.fft.rfft(t, n=n, dim=dim) for t in [x, y])
    return torch.fft.irfft(fx * fy, n=n, dim=dim)     # [... n]


def torch_capacity_data(L, H, batch_shape=()):
    assert L >= H > 1
    x = torch.randn(batch_shape+(L,1))                        # (B,L,1)
    F.normalize(x, p=np.inf, dim=-2, out=x)
    rshifts = torch.linspace(L/H, L, H, dtype=torch.long)-1   # (H) 
    kernels = F.one_hot(rshifts, L).float()                   # (H L)
    y = rconv(x.transpose(-1,-2), kernels)[...,:L].transpose(-1,-2)  # (B L H)
    return x, y  # (B,L,1), (B L H)


def capacity_static_dataset(L, H, samples):
    all_x, all_y = torch_capacity_data(L, H, batch_shape=(samples,))
    print("Constructed capacity dataset of shape", all_x.shape, all_y.shape)
    ds = torch.utils.data.TensorDataset(all_x, all_y)
    return ds

