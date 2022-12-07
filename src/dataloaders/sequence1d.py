import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data.dataset import IterableDataset
import numpy as np
from tqdm.auto import tqdm


def concat_pos(x):
    """ append in last dim the position info of second-last dim 
    """
    L = x.shape[-2]                         # (... L d)
    pos = (2*np.pi*torch.arange(L, device=x.device) / L).view(-1,1)
    cos = torch.zeros_like(x[...,:1]) + pos.cos()
    sin = torch.zeros_like(x[...,:1]) + pos.sin()
    return torch.cat((x,cos,sin), dim=-1)   # (... L d+2)


def rconv(x, y, dim=-1):
    assert all(map(torch.is_floating_point, (x, y))), 'inputs must be real'
    n = x.shape[dim] + y.shape[dim]
    fx, fy = (torch.fft.rfft(t, n=n, dim=dim) for t in [x, y])
    return torch.fft.irfft(fx * fy, n=n, dim=dim)     # [... n]


def shift(L=None, num_shifts=None, batch_shape=(), **kwargs):
    H = num_shifts
    assert L >= H > 1
    x = torch.randn(batch_shape+(L,1))                        # (B,L,1)
    F.normalize(x, p=np.inf, dim=-2, out=x)
    rshifts = torch.linspace(0, L-L/H, H, dtype=torch.long)   # (H) 
    kernels = F.one_hot(rshifts, L).float()                   # (H L)
    y = rconv(x.transpose(-1,-2), kernels)[...,:L].transpose(-1,-2)  # (B L H)
    return concat_pos(x), y  # (B L 3), (B L H)


def context_shift(L=None, batch_shape=(), **kwargs):
    """context dependent shift"""
    assert L >= 4
    x = torch.randn(batch_shape + (L-2,))              # (B,L-2)                        
    F.normalize(x, p=np.inf, dim=-1, out=x)
    rshifts = torch.randint(0, L-2, size=batch_shape)  # (B)
    kernels = F.one_hot(rshifts, L).float()            # (B L)
    rshifts = (2*np.pi*rshifts / L).unsqueeze(-1)      # (B,1)
    x = torch.cat((rshifts.cos(), rshifts.sin(), x), dim=-1).unsqueeze(-1)         # (B,L,1)
    y = rconv(x.transpose(-1,-2), kernels.unsqueeze(-2))[...,:L].transpose(-1,-2)  # (B L 1)
    return concat_pos(x), y  # (B L 3), (B L 1)
   
    
def mips(L=None, D=4, causal=True, batch_shape=(), in_device='cuda', out_device='cpu', **kwargs):
    """ L: num of queries/keys/vals (all norm 1)
        D: dim of query
        causal: i'th query is matched to closest key among k0,...,ki
        
        This is very slow on cpu - its faster to form the samples on gpu and then transfer to cpu to avoid pinned memeory errs.
        Must use loader.num_workers=0 if using device='cuda' to avoid errors.
    """
    assert L > 1 and D > 1
    q, k, v = F.normalize(torch.randn(size=batch_shape+(3*L,D), device=in_device), dim=-1).chunk(3, dim=-2)  # (B L D)
    sim = q.matmul(k.transpose(-1,-2))                        # (... L L)
    if causal:
        i = torch.arange(L, device=q.device)
        sim.masked_fill_(i.view(-1,1) < i.view(1,-1), -1e5)
    inds = sim.argmax(-1, keepdim=True)                       # (... L 1)
    y = v.gather(-2, inds.expand_as(v))                       # (B L D)
    x = torch.cat((q,k,v), dim=-1)                            # (B L 3D)
    x = concat_pos(x)                                         # (B L 3D+2)
    return x.to(out_device), y.to(out_device)                 # (B L 3D+2), (B L D)


def masked_select(task=None, L=None, M=None, batch_shape=(), variable=True, consecutive=False, **kwargs):
    """ 
        L: num noise tokens
        M: num memorization tokens
        variable: vary positions with batches else positions are same in all batches
        consecutive: positions are consecutive
    """
    
    if 'fixed' in task: variable = False
    assert L > 1 and M > 1
    # M toks in randn  L: pad  0: M rightmost pos
        
    if consecutive:
        # pick M random consecutive pos among first M+L for M tokens
        if variable:
            inds = torch.randint(0, L, size=batch_shape+(1,)) + torch.arange(M)
        else:
            inds = torch.randint(0, L, size=(1,), generator=torch.Generator().manual_seed(0)) + torch.arange(M)
            inds = inds.expand(batch_shape+(M,))
    else:
        # pick M random pos among first M+L for M tokens
        if variable:
            inds = torch.rand(batch_shape+(M+L,)).topk(M, dim=-1)[1].sort(-1)[0]
        else:
            inds = torch.rand(M+L, generator=torch.Generator().manual_seed(0)).topk(M, dim=-1)[1].sort(-1)[0]
            inds = inds.expand(batch_shape+(M,))
    
    x_toks = F.pad(F.normalize(torch.randn(batch_shape+(M+L,)), p=np.inf, dim=-1), (0,M)) # (B M+L+M)
    tokens = x_toks.gather(-1, inds)
    markers = torch.zeros_like(x_toks)
    markers.scatter_(-1, inds, 1)
    
    x = torch.stack([x_toks, markers], dim=-1)  # (B M+L+M 2)
    x = concat_pos(x)                           # (B M+L+M 4)
    y = tokens.unsqueeze(-1)
    return x, y   # (B M+L+M 4), (B M 1)


def solve(task=None, L=None, batch_shape=(), variable=True, **kwargs):
    N = int(np.max(np.roots([1, 2, -L])))
    assert N**2 + N + N <= L
   
    if 'fixed' in task: variable = False
    # sample orthogonal linear system
    if variable:
        A = torch.randn(batch_shape + (N,N))    # (B N N)
    else:
        A = torch.randn((N,N), generator=torch.Generator().manual_seed(0))     # (N N)
        A = A.view((1,)*len(batch_shape) + (N,N)).expand(batch_shape + (N,N))  # (B N N)
    
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    A = U.matmul(Vh)                                           # A.A^T == I
    
    X = F.normalize(torch.randn(batch_shape + (N,1)), dim=-2)  # (B N 1)
    B = A.matmul(X)  # AX == B                                 # (B N 1)
    
    AB = torch.cat((A, B), dim=-1)                             # (B N N+1)
    x = AB.view(*AB.shape[:-2], -1)                            # (B N**+N)
    x = F.pad(x, (0,L-x.size(-1))).unsqueeze(-1)               # (B L 1)
    y = X
    return concat_pos(x), y                                    # (B L 3), (B N 1)


def atomic(L=None, task=None, batch_shape=(), **kwargs):
    """ output len is 2L
    """
    assert L > 1
    seq = torch.randn(size=batch_shape+(L,))
    F.normalize(seq, p=np.inf, dim=-1, out=seq)
    x = seq
    
    if task == 'cumsum':
        y = seq.cumsum(-1) / torch.arange(1, L+1)**.5 
    elif task == 'cummax':
        y = seq.cummax(-1).values
    elif task == 'reverse':
        x = F.pad(seq, (0,L))
        y = seq.flip(-1)
    elif task == 'sort':
        x = F.pad(seq, (0,L))
        y = seq.gather(-1, (seq-seq[...,:1]).abs().sort(dim=-1, stable=True)[1])
        # y = seq.sort(stable=True)[0]*(seq[...,:1] <= 0).float() + seq.sort(stable=True, descending=True)[0]*(seq[...,:1] > 0).float()
    else:
        assert False, f'{task}'

    x = concat_pos(x.unsqueeze(-1))           # (B L/2L 3)
    y = y.unsqueeze(-1)                       # (B L 1)
    return x, y


def torch_sequence1d_data(task=None, **kwargs):
    if task == 'shift':
        return shift(**kwargs)
    if task == 'context_shift':
        return context_shift(**kwargs)
    if task == 'mips':
        return mips(**kwargs)
    if 'masked_select' in task:
        return masked_select(task=task, **kwargs)
    if 'solve' in task:
        return solve(task=task, **kwargs)
    return atomic(task=task, **kwargs)


class Sequence1dTrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, samples_per_epoch=-1, **kwargs):
        super().__init__()
        self.samples_per_epoch = samples_per_epoch
        self.kwargs = kwargs
        
    def __iter__(self):
        if self.samples_per_epoch < 0:
            while True:
                x, y = torch_sequence1d_data(batch_shape=(), **self.kwargs)
                yield x, y
        else:
            for _ in range(self.samples_per_epoch):
                x, y = torch_sequence1d_data(batch_shape=(), **self.kwargs)
                yield x, y                
                
        
# import nengo

# def white_noise_capacity_data(L, H, B, data_length_factor=2.5):
#     """ adapted from 
#         https://github.com/nengo/keras-lmu/blob/fix-mackey-glass/experiments/capacity.ipynb
#     """
#     T = L / data_length_factor
#     process = nengo.processes.WhiteSignal(data_length_factor, high=10, y0=0)
#     x = np.random.randn(B, L, 1)
#     for i in range(B):
#         x[i] = process.run_steps(L, dt=1/T)    # [L 1]
#     # x[...,0] = process.run_steps(L, B, dt=1/T).T
    
#     x = torch.tensor(x, dtype=torch.float)
#     F.normalize(x, p=np.inf, dim=-2, out=x)
#     rshifts = torch.linspace(0, T, H, dtype=torch.long)     # (H) 
#     kernels = F.one_hot(rshifts, L).float()                 # (H L)
#     y = rconv(x.transpose(-1,-2), kernels)[...,:L].transpose(-1,-2)  # (B L H)
#     return x, y  # (B L 1), (B L H)


