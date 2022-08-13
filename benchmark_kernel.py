import torch
import torch.nn as nn
import torch.optim as optim
# import torch.backends.cudnn as cudnn

import numpy as np
import os, time
import argparse

from src.models.sequence.ss.standalone.s4 import HippoSSKernel
from src.models.sequence.ss.standalone.dss import DSSKernel
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['s4', 'dss-exp', 'dss-softmax'], type=str)
parser.add_argument('--H', default=256, type=int, help='Model dimension')
parser.add_argument('--L', default=128, type=int, help='length')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
print('==> Building model..')
if args.model == 's4':
    model = HippoSSKernel(args.H, N=64, L=args.L)
else:
    model = DSSKernel(args.H, N=64, version=args.model[4:])

model = model.to(device)
# if device == 'cuda':
    # model = torch.nn.DataParallel(model)
    # cudnn.benchmark = True

model.train()
t0 = time.time()
times = []
pbar = tqdm(range(10000))
for i in pbar:
    torch.cuda.synchronize()
    # following https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/2
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    outputs = model(args.L)
    if type(outputs) == tuple:
        outputs = outputs[0]
    loss = outputs.mean()
    loss.backward()

    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end)*1e-3)  # secs
    
    pbar.set_description(f'{args.model} H={args.H} L={args.L} loss: {loss}, ms {round(np.mean(times)*1e3,1)}, it/s {round(1/np.mean(times),1)}')
    
    if time.time() - t0 > 30:
        exit()
exit()

'''
CUDA_VISIBLE_DEVICES=7 python -m benchmark_kernel.py --model dss-exp --L 4096


for model in s4 #dss-exp dss-softmax
do
    for L in 12 14 16
    do
        L=$((2**L))
        CUDA_VISIBLE_DEVICES=7 python -m benchmark_kernel.py --model $model --L $L
    done
done

'''