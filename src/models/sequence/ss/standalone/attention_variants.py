'''implementations simplified from https://github.com/lucidrains'''

from functools import partial
from itertools import chain
import numpy as np

from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.linalg import qr_multiply
from scipy.stats import chi

#constants

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helper fns

def default(val, default_val):
    return default_val if val is None else val


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def batched_index_select(values, indices):
    '''values: (b,t,d), indices: (b,t')/(b,r,t'), output: (b,t',d)/(b,r,t',d)'''
    b, t, d = values.shape
    in_shape = indices.shape
    out = values.gather(1, indices.view(b, -1, 1).expand(-1, -1, d))  # (b, r*t', d)
    out_shape = in_shape + (d,)
    return out.view(*out_shape)


def bucket(x, buckets, dim):
    shape = list(x.shape)
    dim = dim if dim >= 0 else len(shape) + dim
    shape[dim:dim+1] = [buckets, -1]
    return x.reshape(*shape)


def unbucket(x, dim):
    shape = list(x.shape)
    dim = dim if dim >= 0 else len(shape) + dim
    shape[dim:dim+2] = [-1]
    return x.reshape(*shape)


def expand_dim(x, k, dim):
    x = x.unsqueeze(dim)
    expand_shape = [-1] * len(x.shape)
    expand_shape[dim] = k
    return x.expand(*expand_shape)


def merge_dims(x, dims):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def softmax_topk(x, dim=-1, topk=-1, in_place=False):
    '''can be useful for long x while training'''
    if topk <= 0:
        return nn.Softmax(dim=dim)(x)
    assert in_place
    tp_x, tp_inds = x.topk(topk, dim=dim)
    tp_x = nn.Softmax(dim=dim)(tp_x)
    return x.zero_().scatter_(dim, tp_inds, tp_x)


def softmax_topk_with_logsumexp(x, dim=-1, topk=-1, in_place=False):
    def _softmax(x):
        x_logsumexp = x.logsumexp(dim=dim, keepdim=True)
        return (x - x_logsumexp).exp(), x_logsumexp
    if topk <= 0:
        return _softmax(x)
    assert in_place
    tp_x, tp_inds = x.topk(topk, dim=dim)
    tp_x, tp_x_logsumexp = _softmax(tp_x) 
    return x.zero_().scatter_(dim, tp_inds, tp_x), tp_x_logsumexp


def reduce_mips_to_mcss(x, is_query=False, M=1.0, soundness=True):
    if not soundness:
        return F.normalize(x, dim=-1)
    # make norm of every query/key == M but preserve relative <query, key>
    x = x * M / x.norm(dim=-1).max(-1)[0].unsqueeze(-1).unsqueeze(-1)
    # apply [neyshabur-srebro]
    zero, correction = torch.zeros_like(x[...,:1]), (M**2 - x.norm(dim=-1, keepdim=True)**2)**0.5
    if is_query:
        return torch.cat((x, zero, correction), dim=-1)  # queries
    return torch.cat((x, correction, zero), dim=-1)      # keys


def spherical_kmeans_iters(x, means, n_iters):
    # make sure x and means are unit normalized
    x_centroid_idx_prev = 0
    for i in range(n_iters):
        x_centroid_idx = torch.matmul(x, means.transpose(-1, -2)).argmax(-1)
        means = torch.zeros_like(means).scatter_add_(-2, x_centroid_idx.unsqueeze(-1).expand_as(x), x)
        means = F.normalize(means, dim=-1)
        if (x_centroid_idx == x_centroid_idx_prev).all():
            break
        x_centroid_idx_prev = x_centroid_idx
    return means, x_centroid_idx


class LSHAttn(nn.Module):
    def __init__(self, bucket_size=64, n_hashes=8, topk_for_softmax=-1, allow_duplicate_attention=True, **kwargs):
        super().__init__()
        self.n_hashes = n_hashes             # num rounds of hashing
        self.bucket_size = bucket_size
        self.allow_duplicate_attention = allow_duplicate_attention
        self.topk_for_softmax = topk_for_softmax
        
    def hash_vectors(self, n_colors, x, proj=None):
        # vecs will be query/key : (newb, t, head_dim)
        b, t, d, device = *x.shape, x.device
        # sample a head_dim x (n_colors // 2) matrix of gaussians & use it 
        # to project each vec to get score for each colors at each seq vector
        if proj is None:
            assert (n_colors % 2 == 0) or (n_colors == 1)
            proj = torch.randn(d, self.n_hashes, max(1, n_colors // 2)).to(x)
        proj_x = torch.einsum('btd,drn->brtn', x, proj)                             # (b,t,d),(d,n_rounds,n_colors//2)
        proj_x = torch.cat([proj_x, -proj_x], dim=-1) if n_colors > 1 else proj_x   # (b,n_rounds,t,n_colors)
        colors = proj_x.argmax(dim=-1)                                              # (b,n_rounds,t)
        # bring scores in [0,1) without changing ranking
        return colors, proj
  
    def forward(self, q, k, v, input_mask=None, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        q, k, v: [bsz*n_heads, seq_len, head_size]
        '''
        batch_size, seqlen, dim, device = *q.shape, q.device
        bucket_size = self.bucket_size if (0 < self.bucket_size < seqlen) else seqlen
        assert not (seqlen % bucket_size)
        n_buckets = seqlen // bucket_size # bucket_size is m in the paper, n_colors is the n_buckets in the paper
        n_colors = n_buckets * 2          # as recommended in the paper
        # larger n_colors is worse as it requires larger n_hashes as there is no structure b/w distinct colors
        # this issue can probably be fixed via a learned version
        
        with torch.no_grad():
            # normalize queries, keys but preserve inner prod rankings
            buckets_q, proj = self.hash_vectors(n_colors, q)
            assert buckets_q.shape == (batch_size, self.n_hashes, seqlen) 
            buckets_k, _ = self.hash_vectors(n_colors, k, proj)
            del proj
            # Hash-based sort ("s" at the start of variable names means "sorted")
            # sort by bucket assignment - within same bucket sort by similarity to bucket
            _, sticker_q = buckets_q.sort(dim=-1)   # ties are broken using pos
            _, undo_sort_q = sticker_q.sort(dim=-1)
            # an element of sticker denotes the index of buckets that was mapped here in sbuckets
            _, sticker_k = buckets_k.sort(dim=-1)
            del buckets_q, buckets_k
            # (b, n_rounds, seqlen)
            tup = (sticker_q, undo_sort_q, sticker_k)
            sticker_q, undo_sort_q, sticker_k = (x.detach() for x in tup)

        sq = batched_index_select(q, sticker_q)              # (b, n_rounds, seqlen, dim)
        sk = batched_index_select(k, sticker_k)              # (b, n_rounds, seqlen, dim)
        sv = batched_index_select(v, sticker_k)              # (b, n_rounds, seqlen, dim)
        del sticker_q, sticker_k
        
        # Split off a "bin" axis so that attention only occurs within chunks.
        split_bin = lambda x: torch.reshape(x, (batch_size, self.n_hashes, n_buckets, -1, dim))
        bq, bk, bv = map(split_bin, (sq, sk, sv))

        look_one_back = lambda x: torch.cat((x.roll(1, dims=2), x), dim=3)
        # no need to look back for the query
        bk, bv = map(look_one_back, (bk, bv))                             # (b, n_rounds, n_bins, 2*bin_size, dim)
        
        # Dot-product attention.
        dots = torch.einsum('brnid,brnjd->brnij', bq, bk) * (dim ** -0.5) # (b, n_rounds, n_bins, bin_size, 2*bin_size)
        # each chunk atnds to 2 chunks
        
        # Softmax.
        # for topk > 0 following op is in-place and so original dots will be overwritten
        #(b, n_rounds, n_bins, bin_size, 2*bin_size), #(b, n_rounds, n_bins, bin_size, 1)
        topk = self.topk_for_softmax
        dots, dots_logsumexp = softmax_topk_with_logsumexp(dots, dim=-1, topk=topk, in_place=topk>0)  
        
        bo = torch.einsum('brnij,brnjd->brnid', dots, bv)              #(b, n_rounds, n_bins, bin_size, dim)
        so = torch.reshape(bo, (batch_size, self.n_hashes, -1, dim))   #(b, n_rounds, seqlen, dim)
        slogits = torch.reshape(dots_logsumexp, (batch_size, self.n_hashes, -1))  #(b, n_rounds, seqlen)
        
        o = so.gather(-2, undo_sort_q.unsqueeze(-1).expand_as(so))
        logits = slogits.gather(-1, undo_sort_q)
        assert o.shape == (batch_size, self.n_hashes, seqlen, dim)
        logits = logits.unsqueeze(-1)                                  #(b, n_rounds, seqlen, 1)
        
        if self.n_hashes == 1:
            out = o.squeeze(1)
        else:
            # if query has logit-vec pairs (s_1_1,v11),(s12,v12)....(s_n_rounds_2*binsize,v_n_rounds_2*binsize) 
            # then softmax is applied on all s_i_j's together and vecs are summed.
            probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
            out = torch.sum(o * probs, dim=1)  # (batch_size, seqlen, dim)

        assert out.shape == q.shape
        return out                             # (b, seqlen, dim)

    
class LocalAttn(nn.Module):
    def __init__(self, type='vanilla', num_toks_attended=-1, topk_for_softmax=-1, **kwargs):
        super().__init__()
        self.type = type
        self.num_toks_attended = num_toks_attended
        self.topk_for_softmax = topk_for_softmax
        self.attn_func = {'vanilla': self.vanilla_attn, 'chunked': self.chunked_attn,
                         'chunked_neighbour': self.chunked_neighbour_attn, 
                         'sliding_window': self.sliding_window_attn}[type]

    def vanilla_attn(self, q, k, v, dropout_fn=None, mask=None, **kwargs):
        # Dot-product attention.
        dots = torch.matmul(q, k.transpose(-1,-2)) * (q.shape[-1] ** -0.5)    #(.., seqlen, seqlen)
        if mask is not None:
            dots = dots + mask
        topk = self.topk_for_softmax
        dots = softmax_topk(dots, dim=-1, topk=topk, in_place=topk>0)         #(.., seqlen, seqlen)
        if dropout_fn is not None:
            dots = dropout_fn(dots)
        o = torch.matmul(dots, v)                                             #(.., seqlen, dim)
        assert o.shape == q.shape        
        return o

    def chunked_attn(self, q, k, v, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        q, k, v: [bsz*n_heads, seq_len, head_size]
        '''
        batch_size, seqlen, dim, bucket_size = *q.shape, self.num_toks_attended
        bucket_size = bucket_size if bucket_size > 0 else seqlen
        assert seqlen % bucket_size == 0
        n_bins = n_buckets = seqlen // bucket_size

        # Split off a "bin" axis
        split_bins = lambda x: x.view(batch_size, n_bins, -1, dim)
        bq, bk, bv = map(split_bins, (q, k, v))
        # Dot-product attention.
        dots = torch.einsum('bnid,bnjd->bnij', bq, bk) * (dim ** -0.5)                
        # Softmax.
        dots = softmax_topk(dots, dim=-1, topk=self.topk_for_softmax) #(b, n_bins, bin_size, bin_size)          
        bo = torch.einsum('bnij,bnjd->bnid', dots, bv)                #(b, n_bins, bin_size, dim)
        o = torch.reshape(bo, (batch_size, -1, dim))                  #(b, seqlen, dim)       
        assert o.shape == v.shape == (batch_size, seqlen, dim)
        return o

    def chunked_neighbour_attn(self, q, k, v, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        qk, v: [bsz*n_heads, seq_len, head_size]
        '''
        batch_size, seqlen, dim, bucket_size = *q.shape, self.num_toks_attended
        assert (bucket_size > 0) and (bucket_size % 2 == 0)  and seqlen % (2*bucket_size) == 0
        n_bins = n_buckets = seqlen // (bucket_size // 2) 

        split_half_bins = lambda x: x.view(batch_size, -1, bucket_size//2, dim)
        bq, bk, bv = map(split_half_bins, (q, k, v))

        # each bin attends to itself (bucket_size/2) and half of left bin (bucket_size/4) and half of right bin (bucket_size/4)
        def cat_half_neighbours(x):
            xl = x[..., -bucket_size//4:, :].roll(1, dims=-3)         # right half of left chunk
            xr = x[..., :bucket_size//4, :].roll(-1, dims=-3)         # left half of right chunk
            return torch.cat((xl, x, xr), dim=-2)
        bk, bv = map(cat_half_neighbours, (bk, bv))

        # Dot-product attention.
        dots = torch.einsum('bnid,bnjd->bnij', bq, bk) * (dim ** -0.5)                
        # Softmax.
        dots = softmax_topk(dots, dim=-1, topk=self.topk_for_softmax)  #(b, n_bins, bin_size/2, bin_size)          
        bo = torch.einsum('bnij,bnjd->bnid', dots, bv)                 #(b, n_bins, bin_size/2, dim)
        o = torch.reshape(bo, (batch_size, -1, dim))                   #(b, seqlen, dim)       
        assert o.shape == v.shape == (batch_size, seqlen, dim)
        return o        # (b, seqlen, dim)

    def sliding_window_attn(self, q, k, v, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        qk, v: [bsz*n_heads, seq_len, head_size]
        '''
        batch_size, seqlen, dim, window_size = *q.shape, self.num_toks_attended
        assert (window_size > 0) and (window_size % 2 == 0)  and seqlen % (2*window_size) == 0
        hw = window_size // 2     # half window
        n_bins = n_buckets = seqlen // hw

        split_half_bins = lambda x: x.view(batch_size, -1, hw, dim)
        bq, bk, bv = map(split_half_bins, (q, k, v))

        # each bin attends to itself (window_sz/2), left bin (window_sz/2), right bin (window_sz/2) with sliding mask
        def cat_half_neighbours(x):
            xl = x.roll(1, dims=-3)  # left chunk
            xr = x.roll(-1, dims=-3) # right chunk
            return torch.cat((xl, x, xr), dim=-2)
        bk, bv = map(cat_half_neighbours, (bk, bv))

        # Dot-product attention.
        dots = torch.einsum('bnid,bnjd->bnij', bq, bk) * (dim ** -0.5)  # [bsz, n_bins, wdsz/2, 3*wdsz/2]
        assert dots.shape[-2:] == (hw, 3*hw)
        lmask = torch.ones(hw, hw, device=q.device).tril_(-1).bool()
        mask = torch.cat((lmask, torch.zeros_like(lmask), ~lmask), dim=-1)  # sliding window mask
        dots.masked_fill_(mask, -1.0e4)                                 # True's are masked out
        del mask
        # Softmax.
        dots = softmax_topk(dots, dim=-1, topk=self.topk_for_softmax)                             
        bo = torch.einsum('bnij,bnjd->bnid', dots, bv)                  #(b, n_bins, wdsz/2, dim)
        o = torch.reshape(bo, (batch_size, -1, dim))                    #(b, seqlen, dim)       
        assert o.shape == v.shape == (batch_size, seqlen, dim)
        return o         # (b, seqlen, dim)

    def forward(self, *args, **kwargs):
        return self.attn_func(*args, **kwargs)


class SinkhornAttn(nn.Module):       
    def __init__(self, config, max_buckets, bucket_size, sinkhorn_iters=5, temperature=0.75, topk_for_softmax=-1, **kwargs):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.bucket_size = bucket_size
        self.max_buckets = max_buckets   # max num of buckets
        self.topk_for_softmax = topk_for_softmax
        [self.sortnet_keys, self.sortnet_vals] = [
            nn.Parameter(torch.randn(self.num_attention_heads, self.attention_head_size, max_buckets)) for _ in range(2)]
   
    def sort_net(self, x, mode=''):
        assert mode in ['keys', 'vals']
        b, h, buckets, d = x.shape
        assert buckets <= self.max_buckets
        W = self.sortnet_keys if mode == 'keys' else self.sortnet_vals
        W = W[..., :buckets]                       # [h, d, buckets]
        R = torch.einsum('bhid,hdf->bhif', x, W)   # [b, h, buckets, buckets]
        
        # add gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(R) + 1e-6) + 1e-6)
        R = (R + gumbel_noise) / self.temperature
        
        # sinkhorn normalization
        for _ in range(self.sinkhorn_iters):
            R = R - torch.logsumexp(R, dim=-1, keepdim=True)
            R = R - torch.logsumexp(R, dim=-2, keepdim=True)  
            # normalizing rows makes sense but why columns? what if a chunk needs to be copied to all chunks?
        return R.exp()
        
    def forward(self, q, k, v, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        q, k, v: [bsz*n_heads, seq_len, head_size]
        '''
        assert q.ndim == 3
        _, t, d = q.shape
        q, k, v = map(lambda x: x.reshape(-1, self.num_attention_heads, t, d), (q, k, v))
        
        b, h, t, d = q.shape
        bucket_size = self.bucket_size
        assert t % bucket_size == 0
        buckets = t // bucket_size
        assert buckets <= self.max_buckets

        # bucket query, key, values
        b_q, b_k, b_v = map(lambda x: bucket(x, buckets, -2), (q, k, v)) #[b, n_heads, buckets, bucket_sz, head_sz]
        
        # pool keys, vals
        pl_k, pl_v = b_k.sum(dim=-2), b_v.sum(dim=-2)        #[b, n_heads, buckets, head_sz]
        
        # compute reordering matrices for keys, vals
        R_k = self.sort_net(pl_k, mode='keys')               #[b, n_heads, buckets, buckets]
        R_v = self.sort_net(pl_v, mode='vals')
        
        # compute softly permuted buckets        
        b_k_r = torch.einsum('bhij,bhjsd->bhisd', R_k, b_k)  #[b, n_heads, buckets, bucket_sz, head_sz]
        b_v_r = torch.einsum('bhij,bhjsd->bhisd', R_v, b_v)

        # dot product wrt permuted and original buckets are added as in the paper
        # for improved version use sigma, 1-sigma (init : 1/2-1/2)
        b_k = b_k_r + b_k                                                  #[b, n_heads, buckets, bucket_sz, head_sz]
        dots = torch.einsum('bhuid,bhujd->bhuij', b_q, b_k) * (d ** -0.5)  #[b, n_heads, buckets, bucket_sz, bucket_sz]
        # attention
        dots = softmax_topk(dots, dim=-1, topk=self.topk_for_softmax)
        #dots = self.dropout(dots)
        
        out = torch.einsum('bhuij,bhujd->bhuid', dots, b_v)  #[b, n_heads, buckets, bucket_sz, head_sz]
        out = unbucket(out, -3)                              #[b, n_heads, seqlen, head_sz]
        assert out.shape == q.shape
        return out.reshape(-1, t, d)                         #[b*n_heads, seqlen, head_sz]


class PolynomialAttn(nn.Module):
    def __init__(self, type='polynomial', degree=2, topk_for_softmax=-1, **kwargs):
        super().__init__()
        self.type = type
        self.degree = degree
        self.topk_for_softmax = topk_for_softmax

    def forward(self, q, k, v, **kwargs):
        # Dot-products
        dots = torch.matmul(q, k.transpose(-1,-2))                            #(.., seqlen, seqlen)
        topk = self.topk_for_softmax
        def apply_kernel_normalize(x):
            x = x**self.degree
            return x / x.sum(dim=-1, keepdim=True).clamp_min_(1e-4)
        if topk <= 0:
            dots = apply_kernel_normalize(dots)
        else:
            tp_dots, tp_inds = dots.topk(topk, dim=-1)
            tp_dots = apply_kernel_normalize(tp_dots)
            dots.zero_().scatter_(-1, tp_inds, tp_dots)
        o = torch.matmul(dots, v)                                             #(.., seqlen, dim)
        assert o.shape == q.shape
        return o

    
class KernelAttn(nn.Module):
    """linear transformer"""
    def __init__(self, kernel='elu', **kwargs):
        super().__init__()
        assert kernel in ['elu']
        if kernel == 'elu':
            self.feature_map = lambda x: nn.ELU()(x) + 1
    
    @staticmethod
    def linear_attn(q, k, v):
        assert all([x.ndim == 3 for x in (q, k, v)])
        k_sum = k.sum(-2).unsqueeze(-1)               # (b, newdim, 1)
        kv = torch.bmm(k.transpose(-1, -2), v)        # (b, newdim, dim)
        deno = torch.bmm(q, k_sum).clamp_min_(1e-4)   # (b, seqlen, 1)
        o = torch.bmm(q, kv) / deno                   # (b, seqlen, dim)
        assert o.shape == v.shape                     # (b, seqlen, dim)
        return o
        
    def forward(self, q, k, v, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        qk, v: [bsz*n_heads, seq_len, head_size]
        '''
        batch_size, seqlen, dim = q.shape
        assert q.shape == k.shape == v.shape
        q, k = map(self.feature_map, (q, k))          # (b, seqlen, newdim)
        return self.linear_attn(q, k, v)              # (b, seqlen, dim)


class ORFAttn(nn.Module):
    """performer"""
    def __init__(self, out_dim, config, **kwargs):
        super().__init__()
        in_dim = int(config.hidden_size / config.num_attention_heads)
        assert (out_dim >= in_dim) and not (out_dim % in_dim)
        self.in_dim, self.out_dim = in_dim, out_dim # in_dim == head_size
        self.sigma = in_dim**0.25                   # of gaussian kernel
#         n_stacks = int(out_dim / in_dim)
        
#         random_weights_ = []
#         for _ in range(n_stacks):
#             W = np.random.randn(in_dim, in_dim)
#             S = np.diag(chi.rvs(df=in_dim, size=in_dim))
#             SQ, _ = qr_multiply(W, S)
#             random_weights_ += [torch.from_numpy(SQ) / self.sigma]
        
#         # shared for all heads
#         self.weight = nn.Parameter(torch.cat(random_weights_).float()) # this is ok as pre-trained init only considers nn.Linear
#         del random_weights_
        self.weight = nn.Parameter(torch.randn(self.out_dim,self.in_dim))
        torch.nn.init.orthogonal_(self.weight)
        self.offset = None#nn.Parameter(torch.rand(out_dim)*2*PI)
        
    def forward(self, q, k, v, **kwargs):
        '''Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        q, k, v: [bsz*n_heads, seq_len, head_size]
        '''
        assert q.ndim == 3 and self.in_dim == q.shape[-1]
        # <q,k>/c = <q/sqrt(c), k/sqrt(c)> 
        q, k = q / self.sigma, k / self.sigma
        # center keys for stability - doesn't alter attn scores
        k = k - k.mean(dim=-2, keepdim=True)  
        
        def ORF_map_sin_cos(x, is_query):
            # exp(<q,k>) = exp(|q|^2 / 2).exp(-|q-k|^2 / 2).exp(|k|^2 / 2)
            # cos(<w,q-k>) = cos(<w,q>-<w,k>) = cos(<w,q>)cos(<w,k>) + sin(<w,q>)sin(<w,k>)
            # w with N(0,1) entries : E[cos(<w,z>)] = exp(-|z|^2 / 2) (as <w,z> ~ N(0,|z|^2))
            x_proj = F.linear(x, self.weight, self.offset)
            x_phi = torch.cat((x_proj.cos(), x_proj.sin()), dim=-1) * self.out_dim**-0.5  # [b,seqlen,2*outdim]
            if is_query:
                # ignoring exp(|q|^2 / 2), gets cancelled in softmax normalization
                return x_phi
            x_norm_sqr = (x**2).sum(dim=-1, keepdim=True) / 2                             # [b, seq_len, 1]
            # following subtraction doesn't change attention scores
            renormalizer = x_norm_sqr.max(dim=-2, keepdim=True)[0]                        # [b, 1, 1]   
            x_norm_scale = (x_norm_sqr - renormalizer).exp()
            return x_norm_scale * x_phi
        
        def ORF_map_exp(x, is_query):
            # w with N(0,1) entries : E[exp(<w,z>)] = exp(|z|^2 / 2)  (as <w,z> ~ N(0,|z|^2))
            # => E[exp(<w,q>)*exp(<w,k>)] = E[exp(<w,q+k>)] = exp(|q+k|^2 / 2) = exp(<q,k> + |q|^2 / 2 + |k|^2 / 2)
            x_norm_sqr = (x**2).sum(dim=-1, keepdim=True) / 2                             # [b, seq_len, 1]
            x_proj = F.linear(x, self.weight, self.offset)                                # [b, seq_len,out_dim]
            normalizer = x_proj.max(dim=-1, keepdim=True)[0] if is_query else x_proj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            # following is equivalent to multiplying with a common scalar, gets cancelled in normalization
            x_phi = (x_proj - x_norm_sqr - normalizer).exp()                                           # [b,seqlen,outdim]
            return x_phi * self.out_dim**-0.5
        
        q, k = ORF_map_exp(q, True), ORF_map_exp(k, False)    # [b, seq_len, 2*out_dim]        
        return KernelAttn.linear_attn(q, k, v)                # (b, seqlen, dim)


class LinearProjectionAttn(nn.Module):
    """linformer"""
    def __init__(self, in_len, out_len, **kwargs):
        super().__init__()
        self.in_len, self.out_len = in_len, out_len
        self.weight = nn.Parameter(torch.Tensor(out_len, in_len))
        nn.init.orthogonal_(self.weight)  # this is ok as pre-trained init only looks at nn.Linear
        
    def forward(self, q, k, v, dropout_fn=None):
        '''linformer: Input is assumed to be already reshaped and permuted to have 
        each sample represent different heads of the original samples. 
        q, k, v: [bsz*n_heads, seq_len, head_size]'''
        assert self.in_len == k.shape[-2] == v.shape[-2]
        proj_fn = lambda x: F.linear(x.transpose(-1,-2), self.weight).transpose(-1,-2)
        k, v = map(proj_fn, (k, v))
        # Dot-product attention.
        dots = torch.einsum('bid,bjd->bij', q, k) * (k.shape[-1] ** -0.5)
        # Softmax.
        dots = nn.Softmax(dim=-1)(dots)   
        if dropout_fn is not None:
            dots = dropout_fn(dots)
        o = torch.einsum('bij,bjd->bid', dots, v)
        assert o.shape == q.shape
        return o         # (b, seqlen, dim)


class AttentionWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        assert not args, 'for safety only keyword args are accepted so specify the argument name'
        self.kwargs = kwargs  # dict
        
        self.attention, typ = None, kwargs['type']
        if typ in ['vanilla', 'chunked', 'chunked_neighbour', 'sliding_window']:
            self.attention = LocalAttn(**kwargs)
        elif typ == 'orf':
            self.attention = ORFAttn(**kwargs)
        elif typ == 'sinkhorn':
            self.attention = SinkhornAttn(**kwargs)
        elif typ == 'linear_projection':
            self.attention = LinearProjectionAttn(**kwargs)
        elif typ == 'lsh':
            self.attention = LSHAttn(**kwargs)
        elif typ == 'kernel':
            self.attention = KernelAttn(**kwargs)
        elif typ == 'polynomial':
            self.attention = PolynomialAttn(**kwargs)
        assert self.attention is not None, 'specify the type argument correctly'
    
    def forward(self, *args, **kwargs):
        assert not args, 'for safety only keyword args are accepted so specify the argument name'
        return self.attention(**kwargs)


































# class KMeansAttn(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.means = None
#         self.max_norm = 
               
#     def cluster_vectors(self, vecs, centroids, kmeans_steps=0):
#         # vecs will be keys : (newb, t, head_dim)
#         # centroids are used to cluster the keys : (newb, num_centroids, head_dim)
#         for _ in range(kmeans_steps):
#             # update the centroids using hard kmeans
#             scores = torch.einsum('btf,bcf->btc', vecs, centroids)                 # (b,seqlen,num_centroids)
#             pr_ctrd_given_vec = scores >= (scores.max(dim=-1, keepdim=True)[0].to(scores.dtype) - 1e-4)  # (b,seqlen,num_centroids)
#             p_ctrd = pr_ctrd_given_vec.sum(dim=-2).unsqueeze(-1)                   # (b,num_centroids,1)
#             centroids = torch.einsum('btc,btf->bcf', pr_ctrd_given_vec, vecs) / p_ctrd
#         scores = torch.einsum('btf,bcf->btc', vecs, centroids)                     # (b,seqlen,num_centroids)
#         buckets = torch.argmax(scores, dim=-1)                                     # (b, seqlen)  
#         return buckets, scores

#     def forward(self, qk, v, centroids, kmeans_steps=0, input_mask=None):
#         '''Input is assumed to be already reshaped and permuted to have 
#         each sample represent different heads of the original samples. 
#         qk, v: [bsz*n_heads, seq_len, head_size], centroids: [bsz*n_heads, num_centroids, head_size]
#         '''
#         batch_size, seqlen, dim = qk.shape
#         v = default(v, qk)
#         device = qk.device
        
#         n_bins = n_buckets = centroids.shape[-2]
        
#         with torch.no_grad():
#             buckets, scores = self.cluster_vectors(qk, centroids, kmeans_steps=kmeans_steps) # (b, seqlen), (b, seqlen, num_centroids)
#             assert buckets.shape == (batch_size, seqlen)
        
#             # Hash-based sort ("s" at the start of variable names means "sorted")
#             sbuckets, sticker = buckets.sort(dim=-1)      # ties are broken using pos
#             _, undo_sort = sticker.sort(dim=-1)
#             # an element of sticker denotes the index of buckets that was mapped here in sbuckets
            
#         buckets = buckets.detach()
#         sbuckets = sbuckets.detach()
#         sticker = sticker.detach()                       
#         undo_sort = undo_sort.detach()
        
#         st = sticker                                # (b, seqlen)
#         sqk = batched_index_select(qk, st)          # (b, seqlen, dim)
#         sv = batched_index_select(v, st)            # (b, seqlen, dim)
        
#         # Split off a "bin" axis so that attention only occurs within chunks.
# #         bq_t = bkv_t = torch.reshape(st, (batch_size, n_bins, -1))
#         # st, bkv_t entries specify the index from orig seq thats mapped here
#         bqk = torch.reshape(sqk, (batch_size, n_bins, -1, dim))
#         bv = torch.reshape(sv, (batch_size, n_bins, -1, dim))
# #         bq_buckets = bkv_buckets = torch.reshape(sbuckets, (batch_size, n_bins, -1))

#         bq = bk = bqk

#         # Dot-product attention.
#         dots = torch.einsum('bnid,bnjd->bnij', bq, bk) * (dim ** -0.5)
#         masked_value = max_neg_value(dots)
        
# #         # attend to the global mem
# #         dots_gmem = None
# #         if gm is not None:
# #             dots_gmem = torch.einsum('bnid,bjd->bnij', bq, gm) * (dim ** -0.5)  # (b, n_bins, bin_size, gmem_size)
            
#         # Input mask for padding in variable length sequences
#         if input_mask is not None:
#             assert False, 'mask not implemented'
# #             input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
# #             mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
# #             mkv = look_one_back(mq)
# #             mask = mq.unsqueeze(-1) * mkv.unsqueeze(-2)
# #             dots.masked_fill_(~mask, masked_value)
# #             del mask
            
#         # Mask out attention to other buckets. default: ignore
#         if not self._attend_across_buckets:
#             assert False, 'make sure buckets are hard'
#             bucket_mask = bq_buckets.unsqueeze(-1) != bkv_buckets.unsqueeze(-2)
#             dots.masked_fill_(bucket_mask, masked_value)
#             del bucket_mask
                
#         # Softmax.
#         dots = nn.Softmax(dim=-1)(dots)             #(b, n_bins, bin_size, bin_size+gmem_size)
            
#         bo = torch.einsum('bnij,bnjd->bnid', dots, bv)              #(b, n_bins, bin_size, dim)
# #         if dots_gmem is not None:
# #             bo = bo + torch.einsum('bnij,bjd->bnid', dots_gmem, gm)  #(b, n_bins, bin_size, dim)
#         so = torch.reshape(bo, (batch_size, -1, dim))   #(b, seqlen, dim)       
#         o = so.gather(1, undo_sort.unsqueeze(-1).expand_as(so))
#         # torch appropriately re-indexes the grads after sort(), gather()         
#         assert o.shape == v.shape == (batch_size, seqlen, dim)
#         return o #, buckets     # (b, seqlen, dim), (b, seqlen)
    

# class KMeansNystromAttn(nn.Module):
#     def __init__(self, num_heads, head_dim, num_clusters):
#         super().__init__()
#         self.num_heads, self.head_dim, self.num_clusters = num_heads, head_dim, num_clusters
#         self.means = nn.Parameter(torch.randn(num_heads, num_clusters, head_dim))
        
#     def nystrom_attn(q, k, v, landmarks, exp_kernel=False):
#         batch_size, seqlen, dim = *q.shape, pivots.shape[-2]
#         assert q.shape == k.shape == v.shape
        
# #         if exp_kernel:
# #             # exp(<q,k> / c) = exp(|q|**2 / 2c).exp(-|q-k|**2 / 2c).exp(|k|**2 / 2c)
# #             q_norms, k_norms = map(lambda x: x.norm(dim=-1, keepdim=True), (q, k))
# #             q, k = q / q_norms, k / k_norms
# #             q_scale, k_scale = q_norms**2
        
#         # subtract to stablize 
        
#         # K(landmarks, landmarks) inverse
        
#         # K(Q, landmarks)
        
#         # K(landmarks, K)
        
# #         if exp_kernel:
#             # nultiple by the exp sq norms
        
        
        
        
#     def soft_kmeans_iters(self, vecs, centroids, kmeans_iters=0):
#         # vecs will be either query or key : (newb, t, head_dim)
#         # centroids are used to cluster the vecs : (newb, num_centroids, head_dim)
#         for _ in range(kmeans_steps):
#             # update the centroids using soft kmeans
#             scores = torch.einsum('btf,bcf->btc', vecs, centroids) # (b,seqlen,num_centroids)
#             pr_ctrd_given_vec = nn.Softmax(dim=-1)(scores)
#             p_ctrd = pr_ctrd_given_vec.sum(dim=-2).unsqueeze(-1)    # (b,num_centroids,1)
#             centroids = torch.einsum('btc,btf->bcf', pr_ctrd_given_vec, vecs) / p_ctrd
        
#         scores = torch.einsum('btf,bcf->btc', vecs, centroids)                     # (b,seqlen,num_centroids)
#         buckets = torch.argmax(scores, dim=-1)                                     # (b, seqlen)
# #         pr_ctrd_given_vec = nn.Softmax(dim=-1)(scores)
# #         buckets = (pr_ctrd_given_vec*torch.arange(centroids.shape[-2], dtype=scores.dtype).view(1, 1, -1)).sum(-1) # (b,seqlen)  
#         return buckets, scores, centroids

#     def forward(self, q, k, v, kmeans_iters=2):
#         '''Applies a combination of routing + nystrom attention
#         Input is assumed to be already reshaped and permuted to have 
#         each sample represent different heads of the original samples. 
#         qk, v: [bsz*n_heads, seq_len, head_size], gm: [bsz*n_heads, gmem_size, head_size]
#         '''
#         batch_size, seqlen, dim = qk.shape
#         device = qk.device
        
#         n_bins = n_buckets = max(1, seqlen // centroids.shape[-2])
        
#         # buckets are soft scores, their compuation is part of computation graph
#         buckets, scores, centroids = self.cluster_vectors(qk, centroids, kmeans_steps=kmeans_steps) # (b, seqlen), (b, seqlen, num_centroids)
#         assert buckets.shape == (batch_size, seqlen)
        
#         '''DO NOT FORGET TO SUBTRACT THE MAX or work in log space'''
        
#         with torch.no_grad():
#             # Hash-based sort ("s" at the start of variable names means "sorted")
#             sbuckets, sticker = buckets.sort(dim=-1)      # ties are broken using pos
#             _, undo_sort = sticker.sort(dim=-1)
#             # an element of sticker denotes the index of buckets that was mapped here in sbuckets
#             # dont attend to sticker < seqlen
            
#         buckets = buckets.detach()
#         sbuckets = sbuckets.detach()
#         sticker = sticker.detach()
#         undo_sort = undo_sort.detach()

#         st = sticker                                    # (b, seqlen)
#         sqk = batched_index_select(qk, st)              # (b, seqlen, dim)
#         sv = batched_index_select(v, st)                # (b, seqlen, dim)
        
#         # Split off a "bin" axis so that attention only occurs within chunks.
#         bq_t = bkv_t = torch.reshape(st, (batch_size, n_bins, -1))
#         # st, bkv_t entries specify the index from orig seq thats mapped here
#         bqk = torch.reshape(sqk, (batch_size, n_bins, -1, dim))
#         bv = torch.reshape(sv, (batch_size, n_bins, -1, dim))
#         bq_buckets = bkv_buckets = torch.reshape(sbuckets, (batch_size, n_bins, -1))

#         bq = bk = bqk

#         # Dot-product attention.
#         dots = torch.einsum('bnid,bnjd->bnij', bq, bk) * (dim ** -0.5)
#         masked_value = max_neg_value(dots)
        
# #         # attend to the global mem
# #         dots_gmem = None
# #         if gm is not None:
# #             dots_gmem = torch.einsum('bnid,bjd->bnij', bq, gm) * (dim ** -0.5)  # (b, n_bins, bin_size, gmem_size)
            
#         # Input mask for padding in variable length sequences
#         if input_mask is not None:
#             assert False, 'mask not implemented'
# #             input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
# #             mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
# #             mkv = look_one_back(mq)
# #             mask = mq.unsqueeze(-1) * mkv.unsqueeze(-2)
# #             dots.masked_fill_(~mask, masked_value)
# #             del mask
            
#         # Mask out attention to other buckets. default: ignore
#         if not self._attend_across_buckets:
#             assert False, 'make sure buckets are hard'
#             bucket_mask = bq_buckets.unsqueeze(-1) != bkv_buckets.unsqueeze(-2)
#             dots.masked_fill_(bucket_mask, masked_value)
#             del bucket_mask
                
#         # Softmax.
# #         if dots_gmem is not None:
# #             dots = torch.cat((dots, dots_gmem), dim=-1)                
#         dots = nn.Softmax(dim=-1)(dots)             #(b, n_bins, bin_size, bin_size+gmem_size)
#         dots = self.dropout(dots)
# #         if dots_gmem is not None:
# #             dots, dots_gmem = dots.split([dots.size(-1)-gmem_size, gmem_size], dim=-1)
            
#         bo = torch.einsum('bnij,bnjd->bnid', dots, bv)              #(b, n_bins, bin_size, dim)
# #         if dots_gmem is not None:
# #             bo = bo + torch.einsum('bnij,bjd->bnid', dots_gmem, gm)  #(b, n_bins, bin_size, dim)
#         so = torch.reshape(bo, (batch_size, -1, dim))   #(b, seqlen, dim)       
#         o = so.gather(1, undo_sort.unsqueeze(-1).expand_as(so))
#         # torch appropriately re-indexes the grads after sort(), gather() 
        
#         assert o.shape == (batch_size, seqlen, dim)

#         assert o.shape == v.shape
#         return o #, buckets     # (b, seqlen, dim), (b, seqlen)
