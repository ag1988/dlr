import os
import torch

import pykeops
from pykeops.torch import LazyTensor
    
for _ in range(2):
    try:
        pykeops.test_torch_bindings()
    except RuntimeError:
        pykeops.clean_pykeops()

print('conda env:', os.environ['CONDA_DEFAULT_ENV'], torch.cuda.device_count())  # conda env

def test(x, y):
    x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (1e6, 1, 3)
    y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, 2e6,3)
    
    D_ij = ((x_i - y_j)**2).sum(dim=2)
    K_ij = (- D_ij).exp()             
    a_i = K_ij.sum(dim=1) 
    print(a_i)


for i in range(torch.cuda.device_count()):
    # gpu types
    print((i, torch.cuda.get_device_properties(i)))
        
    # cuda matmul 
    a, b = (torch.randn(8, 3, device=f'cuda:{i}') for _ in range(2))
    test(a, b)

   
    