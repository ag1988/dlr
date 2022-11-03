import os
import torch

print(os.environ['CONDA_DEFAULT_ENV'])  # conda env

# gpu types
print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])

# cuda matmul 
a = torch.randn(2, 2, device='cuda')
print(a.mm(a))

print('testing pykeops')

import pykeops

for _ in range(2):
    try:
        pykeops.test_torch_bindings()
    except Error:
        pykeops.clean_pykeops()