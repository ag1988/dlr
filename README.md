## Diagonal Linear RNNs (DLR)

This repository is built on a fork of an older version of [S4 repo](https://github.com/HazyResearch/state-spaces) & contains the accompanying code for the paper:

> **Simplifying and Understanding State Space Models with Diagonal Linear RNNs**\
> Ankit Gupta, Harsh Mehta, Jonathan Berant\
> Paper: https://arxiv.org/abs/2212.00768

For info about the general structure of the repo please refer to the S4 repo. In the following we primarily describe how to reproduce the experiments in the paper.

## Table of Contents
- [Setup](#setup)
- [DLR Experiments](#dlr-experiments)

## Setup <a name="setup"></a>

### Requirements
This repo requires Python 3.8+ and [Pytorch 1.9+](https://pytorch.org/get-started/locally/).
After installing PyTorch, other packages can be installed via `pip install -r requirements.txt`.

We strongly recommend installing [pykeops](https://www.kernel-operations.io/keops/index.html) as some experiments are on very long inputs and we need this library for memory efficiency.

Results reported in the paper can vary with the version of the installed libraries, especially Pytorch 1.11+. In case you're unable to reproduce our results using the above instructions, please create a new environment `dlr` as follows and retry:
```bash
conda deactivate
conda env create -f dlr-conda-env.yml
source activate dlr
```

### Data

#### Datasets and Dataloaders
All logic for creating and loading datasets is in `src/dataloaders`.
The data loaders we consider core are located in `src/dataloaders/datasets.py`.


#### Data
The raw data should be organized as follows.
The data path can be configured by the environment variable `DATA_PATH`, or defaults to `./data` by default, where `.` is the top level directory of this repository (e.g. `dlr`).

#### Data Generation

Atomic tasks such as Shift, Reverse, etc automatically generate data in every batch (see `./src/dataloaders/sequence1d.py`) and you dont need to generate data for these.

`ListOpsSubTrees`: You can generate data as described [here](./src/dataloaders/prepare/listops/README.md).  

`PathfinderSegmentation`: You can generate data as described [here](./src/dataloaders/prepare/pathfinder/README.md).

After generating the data, it should be organized as follows:
```
DATA_PATH/
  pathfinder_segmentation/
    pathfinder128_segmentation/
    pathfinder256_segmentation/
    pathfinder512_segmentation/
  listops_subtrees/
```

## DLR Experiments <a name="dlr-experiments"></a>

This section describes how to use the DLR/DSS/Attention models & reproduce the experiments. The DLR model is defined in this standalone [file](./src/models/sequence/ss/standalone/dss.py).

You must explicitly provide the model flag (e.g. `model=dlr`) to each command as shown below.

```bash
# --- pathfindersegmentation 128 ---

# DLR
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null  experiment=dss-pathfinder-segmentation model.n_layers=5 model=dlr model.layer.version='' model.layer.dt_min=0.0001 model.layer.dt_max=0.1 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.layer.d_state=1024 optimizer.lr=0.0001 loader.batch_size=16 model.layer.max_kernel_length=8192

# DSS-EXP
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null  experiment=dss-pathfinder-segmentation model.n_layers=5 model=dss model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.dt_max=0.01 model.layer.d_state=1024 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001  loader.batch_size=16 model.layer.max_kernel_length=8192

# LocalAttention
CUDA_VISIBLE_DEVICES=0,1 python -m train wandb=null  experiment=dss-pathfinder-segmentation model.n_layers=5 model=dlr model.layer.kernel_type=attn optimizer.lr=0.001 model.layer.chunk_size=1024 loader.batch_size=8 trainer.gpus=2 trainer.find_unused_parameters=false 


# --- pathfindersegmentation 256 ---

# DLR - 3 x 3090's
CUDA_VISIBLE_DEVICES=0,1,2 python -m train wandb=null experiment=dss-pathfinder-segmentation-256 model.n_layers=6 model=dlr model.layer.version='' model.layer.dt_min=0.0001 model.layer.dt_max=0.1 model.layer.lr.Lambda=0.00005 model.layer.lr.W=0.00005 model.layer.d_state=1024 optimizer.lr=0.00005 loader.batch_size=6 model.layer.max_kernel_length=32768 trainer.gpus=3 trainer.find_unused_parameters=false trainer.save_val_outputs=false 

# DSS-EXP - 3 x 3090's
CUDA_VISIBLE_DEVICES=0,1,2 python -m train wandb=null experiment=dss-pathfinder-segmentation-256 model.n_layers=6 model=dss model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.dt_max=0.01 model.layer.d_state=1024 optimizer.lr=0.0005 model.layer.lr.Lambda=0.0005 model.layer.lr.W=0.0005 model.layer.lr.log_dt=0.0005  loader.batch_size=6 model.layer.max_kernel_length=32768 trainer.gpus=3 trainer.find_unused_parameters=false trainer.save_val_outputs=false 


# --- pathfindersegmentation 512 ---

# DLR - 7 x V100s
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m train  wandb=null experiment=dss-pathfinder-segmentation-512 model.n_layers=12 model=dlr model.layer.version='' model.layer.dt_min=0.0001 model.layer.dt_max=0.1 model.layer.lr.Lambda=0.00001 model.layer.lr.W=0.00001 model.layer.d_state=2048 optimizer.lr=0.00001 loader.batch_size=2 model.layer.max_kernel_length=32768 trainer.gpus=7 trainer.find_unused_parameters=false trainer.save_val_outputs=false model.d_model=64

# DSS-EXP - 7 x V100s
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -m train  wandb=null experiment=dss-pathfinder-segmentation-512 model.n_layers=12 model=dss model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.dt_max=0.01 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0005 model.layer.d_state=2048 optimizer.lr=0.0005 model.layer.lr.log_dt=0.0005 loader.batch_size=2 model.layer.max_kernel_length=32768 trainer.gpus=7 trainer.find_unused_parameters=false trainer.save_val_outputs=false model.d_model=64


# --- listopssubtrees ---

# DLR
CUDA_VISIBLE_DEVICES=0 python -m train  wandb=null experiment=dss-listops-subtrees model=dlr model.layer.version='' model.layer.dt_min=0.0001 model.layer.dt_max=0.1 model.layer.lr.Lambda=0.0008 model.layer.lr.W=0.0008  model.layer.d_state=1024 optimizer.lr=0.0008 loader.batch_size=32 dataset.l_min=7000  dataset.l_max=8192 trainer.save_val_outputs=false

# DSS-EXP
CUDA_VISIBLE_DEVICES=0 python -m train  wandb=null experiment=dss-listops-subtrees model=dss model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.dt_max=0.01 model.layer.d_state=1024 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 loader.batch_size=32 dataset.l_min=7000  dataset.l_max=8192 trainer.save_val_outputs=false

# LocalAttention
CUDA_VISIBLE_DEVICES=0 python -m train  wandb=null experiment=dss-listops-subtrees model=dlr model.layer.kernel_type=attn loader.batch_size=32 dataset.l_min=7000  dataset.l_max=8192 trainer.save_val_outputs=false optimizer.lr=0.001 model.layer.chunk_size=1024
```

Experiments with atomic tasks can be run as follows.

```bash
# in the following $TASK can be one of "shift" "cumsum" "cummax" "sort" "reverse" "masked_select_fixed" "masked_select" "solve_fixed" "solve" "context_shift"

# DLR single layer, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.n_layers=1

# Attention single layer, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.001 model.layer.attn_ff=0

# DSS-EXP single layer, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dss experiment=dss-sequence1d dataset.task=$TASK model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.d_state=4096  model.layer.dt_max=0.01 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 model.n_layers=1

# DLR 6 layers, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=16000 loader.batch_size=16 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.n_layers=6

# Attention 2 layers, input len 512
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=$TASK model.layer.kernel_type=attn dataset.L=512 dataset.samples_per_epoch=16000 loader.batch_size=64 optimizer.lr=0.001 model.n_layers=2 model.layer.attn_ff=0

# DLR 6 layers, input len 512
CUDA_VISIBLE_DEVICES=$i python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=$TASK model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=512 dataset.samples_per_epoch=16000 loader.batch_size=64 optimizer.lr=0.00005 model.layer.lr.Lambda=0.00005 model.layer.lr.W=0.00005 model.n_layers=6
```

```bash 
# --- MIPS ---

# DLR single layer, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=mips model=dlr model.layer.version=''  model.layer.dt_min=0.00001 model.layer.d_state=4096  model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.0001 model.layer.lr.Lambda=0.0001 model.layer.lr.W=0.0001 model.n_layers=1 loader.num_workers=0

# DSS-EXP single layer, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dss experiment=dss-sequence1d dataset.task=mips model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.d_state=4096  model.layer.dt_max=0.01 model.layer.kernel_to_real=real dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 model.n_layers=1 loader.num_workers=0 

# Attention single layer, input len 4096
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dlr experiment=dss-sequence1d dataset.task=mips model.layer.kernel_type=attn dataset.L=4096 dataset.samples_per_epoch=64000 loader.batch_size=4 optimizer.lr=0.0001 model.n_layers=1 loader.num_workers=0
```

Shift task with long inputs.
```bash
# in the following $L can be any one of 16384  65536  262144  1048576

# DLR single layer  (for DLR-prod use model.layer.kernel_to_real=prod)
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=dss-sequence1d dataset.task=shift model=dlr model.layer.version='' model.layer.d_state=4096 model.layer.dt_min=0.00001 model.layer.dt_max=0.00001 model.layer.kernel_to_real=real dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00001 model.layer.lr.Lambda=0.00001 model.layer.lr.W=0.00001 model.d_model=32

# DSS-EXP single layer
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=dss-sequence1d model=dss dataset.task=shift model.layer.Lambda_init='lin' model.layer.dt_min=0.0001 model.layer.d_state=4096  model.layer.dt_max=0.01 model.layer.kernel_to_real=real dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.001 model.layer.lr.Lambda=0.001 model.layer.lr.W=0.001 model.layer.lr.log_dt=0.001 model.n_layers=1 model.d_model=32

# SGConv single layer
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=dss-sequence1d dataset.task=shift model=sgconv model.layer.d_state=4096 model.layer.alpha_min=1 model.layer.alpha_max=1 model.layer.l_max=$L dataset.L=$L dataset.samples_per_epoch=8000 loader.batch_size=4 optimizer.lr=0.00001 model.d_model=32
```


#### Tuning
You can directly tinker with hyperparameters via flags. E.g. 
```bash
python -m train wandb=null model=dss experiment=s4-lra-cifar model.n_layers=6 model.layer.max_kernel_length=256 model.layer.Lambda_init=randn model.layer.d_state=32 model.layer.bidirectional=true model.layer.sep_dt_re_im=false
```

#### Resuming from a checkpoint:
In case your training is incomplete, you can resume from the last checkpoint as follows (note that wandb will pick up from where the last partial run left off and will not copy the previous logs):
```bash
python -m train wandb=null model=dss experiment=s4-lra-pathx model.layer.lr.log_dt=0.0001 model.layer.dt_min=0.0001 model.layer.dt_max=0.01 trainer.resume_from_checkpoint=/--Global--path/dss/outputs/--The--run--dir--/checkpoints/last.ckpt
```

#### Gradient Accumulation
If you're getting OOMs with large batches, you can use gradient accumulation as
```bash
python -m train wandb=null model=dss experiment=s4-lra-pathx loader.batch_size=8 trainer.accumulate_grad_batches=2
# total batch size = 8 x 2 = 16
```
Currently during grad accumulation, same kernel is computed for *every* sub-batch which is wasteful. Caching of kernels will be fixed in the future.


### WandB

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of `configs/config.yaml` (or pass it on the command line `python -m train .... wandb.project=s4`).

Set `wandb=null` to turn off WandB logging.



## Overall Repository Structure
```
configs/         config files for model, data pipeline, training loop, etc.
data/            default location of raw data
extensions/      CUDA extension for Cauchy kernel
src/             main source code for models, datasets, etc.
  callbacks/     training loop utilities (e.g. checkpointing)
  dataloaders/   data loading logic
  models/        model backbones
    baselines/   misc. baseline models
    functional/  mathematical utilities
    nn/          standalone modules and components
    hippo/       core HiPPO logic
    sequence/    sequence model backbones and layers including RNNs and S4/LSSL
  tasks/         encoder/decoder modules to interface between data and model backbone
  utils/
sashimi/         SaShiMi README and additional code (generation, metrics, MTurk)
train.py         training loop entrypoint
```


## Citation
If you find our code or data useful, please cite:
```
@article{gupta2022dlr,
  title={Simplifying and Understanding State Space Models with Diagonal Linear {RNN}s},
  author={Ankit Gupta and Harsh Mehta and Jonathan Berant},
  journal={ArXiv},
  volume = {abs/2212.00768},
  year={2022},
}
```
