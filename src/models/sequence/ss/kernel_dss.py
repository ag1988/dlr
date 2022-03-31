""" DSS kernel implementation.
"""

if __name__ == "__main__":
    import sys
    import pathlib

    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum

from omegaconf import DictConfig


import src.utils.train
log = src.utils.train.get_logger(__name__)


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


def reciprocal(x, epsilon=1e-7, clamp=False):
    '''bounded 1 / x'''
    x_conj = x.conj()
    norm_sq = (x*x_conj).real.clamp(epsilon) if clamp else (x*x_conj + epsilon)
    return x_conj / norm_sq


def hippo_skew_evals(N):
    '''eigenvalues of (Hippo - Hippo.t()) / 2 (largest imag part first)'''
    i = torch.arange(N, dtype=torch.float)
    x = 2*i + 1
    Hippo = (x.view(-1,1) * x.view(1,-1)).sqrt().tril(diagonal=-1)  # [N,N]
    Skew = (Hippo - Hippo.t()) / 2                                  # [N,N] 
    evals = torch.linalg.eigvals(Skew)                              # [N]
    # decreasing order of imag
    return evals[evals.imag.argsort(descending=True)]               # [N]


class DSSKernel(OptimModule):
    """ DSS kernel based on structured softmax (arxiv.org/abs/2203.14343).  
        OptimModule is for setting learning rates for parameters.
    """
    def __init__(
        self,
        H,
        N=64,
        l_max=None,           # currently unused
        dt_min=1e-3,
        dt_max=1e-1,
        trainable=None,       # Dictionary of options to train various DSS parameters
        lr=None,              # Hook to set LR of DSS parameters differently
        sep_dt_re_im=True,    # use separate deltas for real, imag parts of Lambda
        init='hippo_skew_pos_imag',
    ):
        super().__init__()
        
        self.N, self.H, self.sep_dt_re_im = N, H, sep_dt_re_im
        
        # complex tensors are stored as real with an extra last dim of size 2 
        # to denote real, imag parts as ADAM moments are non-linear  
        log_dt, Lambda, W = self.init(N, H, l_max, dt_min, dt_max, sep_dt_re_im, init)  # [H], [N,2], [H,N,2] 
        
        self.lr = DictConfig({"log_dt": 1e-3, "Lambda": 1e-3, "W": 1e-3})
        if lr is not None:
            self.lr.update(lr)
        
        self.trainable = DictConfig({"log_dt": True, "Lambda": True, "W": True})
        if trainable is not None:
            self.trainable.update(trainable)
        
        self.register("log_dt", log_dt, self.trainable.log_dt, self.lr.log_dt, wd=0.0)  # [H] or [H,2]
        self.register("Lambda", Lambda, self.trainable.Lambda, self.lr.Lambda, wd=0.0)  # [N,2] 
        self.register("W",      W,      self.trainable.W,      self.lr.W,      wd=0.0)  # [H,N]
        

    def init(self, N, H, l_max, dt_min, dt_max, sep_dt_re_im, init):
        if init == 'hippo_skew_pos_imag':
            w = hippo_skew_evals(2*N)[:N] - .5                          # [N]
        Lambda = torch.view_as_real(w.reshape(-1).to(torch.cfloat))     # [N,2]
        
        # log delta
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))   # [H]
        if sep_dt_re_im:
            log_dt = log_dt.view(-1,1).tile(2)                          # [H,2]
        
        W = torch.randn(H, N, 2)                                        # [H,N,2]
        return log_dt, Lambda, W 
    
    
    def forward(self, L, state=None):
        '''TODO: Currently during grad accum, we compute the kernel for each batch which needs to be fixed.
                 We're slower than S4 as for them N=32 due to symmetry.
        '''
        assert state is None, 'currently we dont support state'
        assert L >= 1
        
        Lambda, W = map(torch.view_as_complex, (self.Lambda, self.W))      # [N], [H,N]
        
        if self.log_dt.ndim <= 1:
            dt_Lambda = self.log_dt.exp().unsqueeze(-1) * Lambda           # [H,N]
        else:
            # Lambda.real * dt0  +  1j * Lambda.imag * dt1
            dt_Lambda = torch.view_as_complex(self.log_dt.exp().unsqueeze(1) 
                                              * self.Lambda.unsqueeze(0))  # [H,N]
        
        P = dt_Lambda.unsqueeze(-1) * torch.arange(L, device=W.device)     # [H,N,L]
        
        # fast softmax using structure of P
        Lambda_gt_0 = Lambda.real > 0                                    # [N]
        if Lambda_gt_0.any():
            with torch.no_grad():
                P_max = dt_Lambda * (Lambda_gt_0 * (L-1))                # [H,N]
            P = P - P_max.unsqueeze(-1)                                  # [H,N,L]
        S = P.exp()                                                      # [H,N,L]
        
        dt_Lambda_neg = dt_Lambda * (1 - 2*Lambda_gt_0)                  # [H,N]
        # S.sum(-1) == den / num
        num = dt_Lambda_neg.exp() - 1                                    # [H,N]
        den = (dt_Lambda_neg * L).exp() - 1                              # [H,N]
        W = W * num * reciprocal(den * Lambda)                           # [H,N]
        
        return einsum('hn,hnl->hl', W, S).real.unsqueeze(0), state       # [H,L]
    


    
    
# """ Tests below """

# def benchmark_kernel(device):
#     N = 64
#     L = 4096
#     H = 256

#     kernel = DSSKernel(H, N, L).to(device)
    
#     utils.benchmark_forward(100, kernel, desc="dss kernel forward")
#     utils.benchmark_backward(100, kernel, desc="dss kernel backward")
#     utils.benchmark_memory(kernel, desc="dss kernel memory")


# if __name__ == "__main__":
#     from benchmark import utils

#     device = "cuda"  # 'cpu'
#     device = torch.device(device)

#     benchmark_kernel(device)
    