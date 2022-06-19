""" Standalone version of Diagonal State Space (DSS) model. """


import logging
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only

from einops import rearrange, repeat
import opt_einsum as oe

einsum = contract = oe.contract
contract_expression = oe.contract_expression

from omegaconf import DictConfig


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
        
    return logger

log = get_logger(__name__)


""" simple nn.Module components """

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

        
def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias


def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


""" Optimizer utilities """

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
            

""" Misc functional utilities """

_c2r = torch.view_as_real
_r2c = torch.view_as_complex

def reciprocal(x, epsilon=1e-7, clamp=False):
    """ returns 1 / x, with bounded norm """
    x_conj = x.conj()
    norm_sq = (x*x_conj).real.clamp(epsilon) if clamp else (x*x_conj + epsilon)
    return x_conj / norm_sq


def multiply_polynomials(x, y, pad_to_pow2=True):
    """ 
    x : coefficients of polynomial x(z)  [..., a+1]
    y : coefficients of polynomial y(z)  [..., b+1]
    returns coeffs of product x(z).y(z)  [..., a+b+1]
    i.e. (x_0,...,x_a) * (y_0,...,y_b) --> (x_0*y_0, ..., x_0*y_r+...+x_r*y_0 , ..., x_a*y_b)
    """
    if all(t.is_floating_point() for t in (x, y)):
        fft, ifft = torch.fft.rfft, torch.fft.irfft
    else:
        fft, ifft = torch.fft.fft, torch.fft.ifft
    
    # pad the polynomials to degree deg(x)+deg(y) to avoid wrap-around
    n = (x.shape[-1] - 1) + (y.shape[-1] - 1) + 1
    if pad_to_pow2:
        n = 2**math.ceil(math.log2(n))                     # next pow of 2
    x_f = fft(x, n=n)
    y_f = fft(y, n=n)
    xy_f = x_f * y_f
    return ifft(xy_f, n=n)


""" HiPPO utilities """

def hippo_skew_evals(N):
    """ eigenvalues of (Hippo - Hippo.t()) / 2  (largest imag part first) """
    i = torch.arange(N, dtype=torch.float)
    x = 2*i + 1
    Hippo = (x.view(-1,1) * x.view(1,-1)).sqrt().tril(diagonal=-1)  # [N N]
    Skew = (Hippo - Hippo.t()) / 2                                  # [N N] 
    evals = torch.linalg.eigvals(Skew)                              # [N]
    # decreasing order of imag
    return evals[evals.imag.argsort(descending=True)]               # [N]


""" DSS """

class DSSKernel(OptimModule):
    """ DSS kernel based on structured softmax (arxiv.org/abs/2203.14343).  
        OptimModule is for conveniently setting learning rates for parameters.
    """
    def __init__(
        self,
        H,
        N=64,
        l_max=None,             # currently unused
        channels=1,
        dt_min=1e-3,
        dt_max=1e-1,
        trainable=None,         # Dictionary of options to train various DSS parameters
        lr=None,                # Hook to set LR of DSS parameters differently
        sep_dt_re_im=True,      # use separate deltas for real, imag parts of Lambda
        Lambda_init='hippo_skew_pos_imag',
        epsilon=1e-7,           # avoids division by 0
        version='softmax',      # DSS implementation to use
        max_real_Lambda=1e-4,   # max real part of Lambda if version == 'clip'
        **kwargs,               # sink
    ):
        super().__init__()
        assert version in ['softmax', 'exp', 'exp-re-im', 'exp-no-scale', 'clip', 'clip-no-scale']
        self.version = version
        
        self.N, self.H, self.channels, self.sep_dt_re_im = N, H, channels, sep_dt_re_im
        self.Lambda_init, self.epsilon, self.max_real_Lambda = Lambda_init, epsilon, max_real_Lambda
        
        # complex tensors are stored as real with an extra last dim of size 2 
        # to denote real, imag parts as ADAM moments are non-linear  
        log_dt, Lambda, W = self.init(N, H, channels, l_max, dt_min, dt_max, sep_dt_re_im, Lambda_init)  # [H], [N 2], [H N 2] 
        
        self.lr = DictConfig({"log_dt": 1e-3, "Lambda": 1e-3, "W": 1e-3})
        if lr is not None:
            self.lr.update(lr)
        
        self.trainable = DictConfig({"log_dt": True, "Lambda": True, "W": True})
        if trainable is not None:
            self.trainable.update(trainable)
        
        self.register("log_dt", log_dt, self.trainable.log_dt, self.lr.log_dt, wd=0.0)  # [H] or [H,2]
        
        if 'exp' in version:
            assert (Lambda[:,0] <= 0).all()
            self.register("Lambda_log_neg_re", (-Lambda[:,0]).log(), self.trainable.Lambda, self.lr.Lambda, wd=0.0)
            if 'im' in version:
                self.register("Lambda_log_im", Lambda[:,1].log(), self.trainable.Lambda, self.lr.Lambda, wd=0.0)
            else:
                self.register("Lambda_im", Lambda[:,1], self.trainable.Lambda, self.lr.Lambda, wd=0.0)
        else:
            self.register("Lambda", Lambda, self.trainable.Lambda, self.lr.Lambda, wd=0.0)  # [N,2] 
        
        self.register("W", W, self.trainable.W, self.lr.W, wd=0.0)      # [C H N]


    def init(self, N, H, channels, l_max, dt_min, dt_max, sep_dt_re_im, Lambda_init):
        if Lambda_init == 'hippo_skew_pos_imag':
            w = hippo_skew_evals(2*N)[:N] - .5                          # [N]
        elif Lambda_init == 'randn':
            w = torch.randn(N, dtype=torch.cfloat)                      # [N]
        else:
            raise NotImplementedError(f"Lambda init {Lambda_init} is not implemented")
        
        Lambda = _c2r(w.reshape(-1).to(torch.cfloat))                   # [N 2]
        
        # log delta
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))  # [H]
        if sep_dt_re_im:
            log_dt = log_dt.view(-1,1).tile(2)                          # [H,2]
        
        W = torch.randn(channels, H, N, 2)                              # [C H N 2]
        return log_dt, Lambda, W 
    
    
    def _Lambda(self):
        if 'exp' in self.version:
            if 'im' in self.version:
                return -self.Lambda_log_neg_re.exp() + 1j*self.Lambda_log_im.exp()        # [N]
            return -self.Lambda_log_neg_re.exp() + 1j*self.Lambda_im                      # [N]
        if 'clip' in self.version:
            return self.Lambda[:,0].clip(max=self.max_real_Lambda) + 1j*self.Lambda[:,1]  # [N]
        return _r2c(self.Lambda)
    
    
    def forward(self, L, state=None):
        '''TODO: 1. We're slower than S4 in some cases as in S4 effective N is N/2 due to conj symmetry. 
                    Explore how to do it here.
        '''
        assert state is None, 'currently we dont support state'
        assert L >= 1
        
        Lambda = self._Lambda()                                              # [N]
        W = _r2c(self.W)                                                     # [C H N]
        
        if self.log_dt.ndim <= 1:
            dt_Lambda = self.log_dt.exp().unsqueeze(-1) * Lambda             # [H N]
        else:
            # Lambda.real * dt0  +  1j * Lambda.imag * dt1
            dt_Lambda = _r2c(self.log_dt.exp().unsqueeze(1) 
                             * _c2r(Lambda).unsqueeze(0))                    # [H N]
        
        P = dt_Lambda.unsqueeze(-1) * torch.arange(L, device=W.device)       # [H N L]
        
        if self.version in ['softmax']:
            # fast softmax using structure of P
            Lambda_gt_0 = Lambda.real > 0                                    # [N]
            if Lambda_gt_0.any():
                with torch.no_grad():
                    P_max = dt_Lambda * (Lambda_gt_0 * (L-1))                # [H N]
                P = P - P_max.unsqueeze(-1)                                  # [H N L]
            S = P.exp()                                                      # [H N L]

            dt_Lambda_neg = dt_Lambda * (1 - 2*Lambda_gt_0)                  # [H N]
            # 1 / S.sum(-1) == num / den
            num = dt_Lambda_neg.exp() - 1                                    # [H N]
            den = (dt_Lambda_neg * L).exp() - 1                              # [H N]
            W = W * num * reciprocal(den * Lambda, self.epsilon)             # [C H N]
        
        else:
            S = P.exp()                                                      # [H N L]
            if 'no-scale' not in self.version:
                W = W * (dt_Lambda.exp() - 1) * reciprocal(Lambda, clamp=True)  # [C H N]
                
        return einsum('chn,hnl->chl', W, S).float(), state                   # [C H L]


class DSS(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,               # currently unused
            channels=1,            # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',     # activation in between SS and FF
            postact=None,          # activation after FF
            initializer=None,      # initializer on FF
            weight_norm=False,     # weight normalization on FF
            hyper_act=None,        # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,       # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            max_kernel_length=None,  # max len of SSM kernel to be used
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]
        
        Other options are all experimental and should not need to be configured
        
        TODO: 1. Currently during grad accum, kernel is computed for each sub-batch which is wasteful.
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")
        
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.max_kernel_length = max_kernel_length
        self.kernel = DSSKernel(self.h, self.n, l_max=l_max, channels=channels, **kernel_args)
        
        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        
        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )


    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)
        k, _ = self.kernel(L=Lk)  # (C H Lk) (B C H Lk)
        
        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, Lk)) + F.pad(k1.flip(-1), (Lk, 0))
               
        # y = multiply_polynomials(u.unsqueeze(1), k.unsqueeze(0))[..., :L]  # (B 1 H L), (C 1 H Lk) -> (B C H L)
        n = L + Lk
        k_f = torch.fft.rfft(k, n=n)  # (C H ~n/2)
        u_f = torch.fft.rfft(u, n=n)  # (B H ~n/2)
        y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=n)[..., :L] # (B C H L)
        
        
        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, None
    

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

#     def step(self, u, state):
#         """ Step one time step as a recurrent model. Intended to be used during validation.

#         u: (B H)
#         state: (B H N)
#         Returns: output (B H), state (B H N)
#         """
#         assert not self.training

#         y, next_state = self.kernel.step(u, state) # (B C H)
#         y = y + u.unsqueeze(-2) * self.D
#         y = rearrange(y, '... c h -> ... (c h)')
#         y = self.activation(y)
#         if self.transposed:
#             y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
#         else:
#             y = self.output_linear(y)
#         return y, next_state

#     def default_state(self, *batch_shape, device=None):
#         return self.kernel.default_state(*batch_shape)
# 
#     @property
#     def state_to_tensor(self):
#         return lambda state: rearrange('... h n -> ... (h n)', state)
