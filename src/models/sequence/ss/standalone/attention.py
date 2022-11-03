import torch
from torch import nn#, einsum
import torch.nn.functional as F

from opt_einsum import contract as einsum
from einops import rearrange, repeat

try:
    import pykeops
    from pykeops.torch import LazyTensor
    pykeops.set_verbose(False)
    has_pykeops = True
    
except ImportError:
    has_pykeops = False


class RotaryEmbedding(nn.Module):
    """rope = RotaryEmbedding(64)
    q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
    k = torch.randn(1, 8, 1024, 64) # keys

    # apply embeds to queries, keys after the heads have been split out but prior to dot products
    
    q = rope(q)
    k = rope(k)

    # then do attention with queries (q) and keys (k) as usual
    
    q_k = [q_k_0,...,q_k_d-1] is mapped to [...,cos(theta**(2mk/d))q_k_2m - sin(theta**(2mk/d))q_k_2m+1, cos(theta**(2mk/d))q_k_2m+1 + sin(theta**(2mk/d))q_k_2m,...]
    """
    def __init__(self, d, theta=10000):
        super().__init__()
        assert d % 2 == 0
        freqs = theta ** (-torch.arange(0, d, 2) / d)                # (d / 2)
        self.register_buffer('freqs', freqs)
        self.cache = dict()
        
    def get_freqs(self, pos, cache_key=None):
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        freqs = self.freqs                                           # (d/2)
        freqs = pos.to(freqs).view(-1,1) * freqs                     # (L d/2)
        
        cos, sin = freqs.cos(), freqs.sin()
        freqs = torch.stack((cos, -sin, sin, cos), dim=-1)           # (L d/2 4)
        freqs = rearrange(freqs, '... (r c) -> ... r c', c = 2)      # (L d/2 2 2)
        
        if cache_key:
            self.cache[cache_key] = freqs

        return freqs                                                 # (L d/2 2 2)
    
    def forward(self, x, seq_dim = -2):
        # x: (... L d)
        L = x.shape[seq_dim]
        freqs = self.get_freqs(torch.arange(L, device=x.device), L)  # (L d/2 2 2)
        x = rearrange(x, '... (d r) -> ... d r', r = 2)              # (... L d/2 2)
        x = einsum('... r c, ... c -> ... r', freqs, x)              # (L d/2 2 2), (... L d/2 2)
        return rearrange(x, '... d r -> ... (d r)')


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True, transposed=False, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.d, self.n, self.h = d_model, n_heads, d_model // n_heads
        self.causal, self.transposed = causal, transposed 
        self.Q, self.K, self.V, self.O = (nn.Linear(self.d, self.d) for _ in range(4))
        self.scale = nn.Parameter(torch.ones(1) * self.h**-.5)
        
        self.rope = RotaryEmbedding(self.h)   # pos embeds
        self.cache = {}
        
    def forward(self, x, m=None):
        if self.transposed: 
            x = x.transpose(-1, -2)
            if m is not None: m = m.transpose(-1, -2)
        if m is None: m = x
        
        q, k, v = self.Q(x), self.K(m), self.V(m)               # [b l d]
        q, k, v = (rearrange(x, 'b l (n h)-> b n l h', n=self.n) for x in (q, k, v))
        q, k = map(self.rope, (q, k))
        q = q * self.scale
        
        if has_pykeops:
            o = self.pykeops_attn(q, k, v)
        else:
            dots = q.matmul(k.transpose(-1,-2))   # [b n l l]
            if self.causal: 
                i = torch.arange(q.shape[-2], device=q.device)
                j = torch.arange(k.shape[-2], device=k.device)
                dots.masked_fill_(i.view(-1,1) < j.view(1,-1), -1e5)
            o = dots.softmax(-1).matmul(v)        # [b n l h]
        
        o = self.O(rearrange(o, 'b n l h-> b l (n h)', n=self.n))     # [b l d]
        if self.transposed: o = o.transpose(-1, -2)
        return o
    
    def pykeops_attn(self, q, k, v):
        # q, k, v : [b n l h]
        Lq, Lk, device, dtype = q.shape[-2], k.shape[-2], q.device, q.dtype
        q = q.unsqueeze(-2)                       # [b n l 1 h]  
        k, v = k.unsqueeze(-3), v.unsqueeze(-3)   # [b n 1 l h]
        
        q, k, v = map(LazyTensor, (q, k, v))
        dots = (q*k).sum(dim=-1)                  # [b n l l]
        
        if self.causal:
            if (Lq, Lk) in self.cache:
                mask = self.cache[(Lq, Lk)]
            else:
                i = LazyTensor(torch.arange(Lq, dtype=dtype, device=device).view(-1,1,1) + .1)
                j = LazyTensor(torch.arange(Lk, dtype=dtype, device=device).view(1,-1,1))
                mask = (-i + j).step() * -1e5
                self.cache[(Lq, Lk)] = mask
            dots += mask
        return dots.sumsoftmaxweight(v, dim=3)
