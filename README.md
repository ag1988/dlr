## Diagonal State Spaces (DSS)

This repository is built on a fork of the official [S4 repo](https://github.com/HazyResearch/state-spaces) & contains the accompanying code for the paper:

> **Diagonal State Spaces are as Effective as Structured State Spaces**\
> Ankit Gupta\
> Paper: https://arxiv.org/pdf/2203.14343

## Table of Contents
- [Setup](#setup)
- [DSS Experiments](#dss-experiments)
- [Repo Structure](#overall-repository-structure)

## Setup

### Requirements
This repository requires Python 3.8+ and [Pytorch 1.9+](https://pytorch.org/get-started/locally/).
After installing PyTorch, other packages can be installed via `pip install -r requirements.txt`.

If you'll only be using DSS, installing `pykeops` & the Cauchy kernels from [S4 repo](https://github.com/HazyResearch/state-spaces) is optional. But we strongly recommend following all installation instructions on S4 repo & installing these as they're required for S4.

### Data

#### Datasets and Dataloaders
All logic for creating and loading datasets is in `src/dataloaders`.
This folder may include old and experimental datasets.
The datasets that we consider core are located in `src/dataloaders/datasets.py`.


#### Data
The raw data should be organized as follows.
The data path can be configured by the environment variable `DATA_PATH`, or defaults to `./data` by default, where `.` is the top level directory of this repository (e.g. `dss` or `state-spaces`).

Most of the dataloaders download their datasets automatically if not found.
External datasets include Long Range Arena (LRA), which can be downloaded from their [GitHub page](https://github.com/google-research/long-range-arena),
and the WikiText-103 language modeling dataset, which can be downloaded by the `getdata.sh` script from the [Transformer-XL codebase](https://github.com/kimiyoung/transformer-xl).

E.g. LRA can be downloaded/extracted as:
```bash
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar -xvf lra_release.gz
```

These external datasets should be organized as follows:
```
DATA_PATH/
  pathfinder/
    pathfinder32/
    pathfinder64/
    pathfinder128/
    pathfinder256/
  aan/
  listops/
  wt103/
```
Fine-grained control over the data directory is allowed, e.g. if the LRA ListOps files are located in `/home/lra/listops-1000/`, you can pass in `+dataset.data_dir=/home/lra/listops-1000` on the command line.


## DSS Experiments

This section describes how to use the latest DSS model & reproduce the experiments.
More detailed descriptions of the infrastructure are in later sections.

The `DSS` layer is provided in a self-contained file `src/models/sequence/ss/standalone/dss.py`. You must explicitly provide the flag `model=dss` to each command as shown below.

### Quick Testing

For quick testing, we frequently use synthetic datasets or the Permuted MNIST dataset.
This can be run with `CUDA_VISIBLE_DEVICES=0 python -m train wandb=null model=dss pipeline=mnist`, which should get to around 90% after 1 epoch which takes 1-3 minutes depending on GPU.


### Long Range Arena (LRA)

```bash
python -m train wandb=null model=dss experiment=s4-lra-listops
python -m train wandb=null model=dss experiment=s4-lra-imdb
python -m train wandb=null model=dss experiment=s4-lra-aan
python -m train wandb=null model=dss experiment=s4-lra-cifar trainer.max_epochs=200 train.seed=0
python -m train wandb=null model=dss experiment=s4-lra-pathfinder scheduler.patience=13
python -m train wandb=null model=dss experiment=s4-lra-pathx model.layer.dt_min=0.0001 model.layer.dt_max=0.01 model.layer.lr.log_dt=0.0001 loader.batch_size=16
```

### Speech Commands

The Speech Commands dataset modified as a [smaller](https://arxiv.org/abs/2005.08926) [10-way](https://arxiv.org/abs/2102.02611) classification task.

```bash
python -m train wandb=null model=dss experiment=s4-sc
```

#### DSS: approximate test accuracy (at best validation checkpoint) & training time on single A100:
|            | listops  | imdb |  aan  | lra-cifar | pathfinder | pathx |  sc  |
| ---        |    ---   |  --- |  ---  |   ---     |    ---     |  ---  | ---  |
| **acc**    | 58.2     | 76.3 |  87.8 | 85.7      | 84.6       | 85    | 97.7 |
| **time**   | 2h       |  20m |  9h   |  6h       |  9h        |  36h  | 19h  |

These metrics can vary depending on GPU. On Path-X, loss should start decreasing around global step 90k (10h).

#### Tuning
You can directly tinker with hyperparameters via flags. E.g. 
```bash
python -m train wandb=null model=dss experiment=s4-lra-cifar model.n_layers=6 model.layer.Lambda_init=randn model.layer.d_state=32 model.layer.bidirectional=true model.layer.sep_dt_re_im=false
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
Currently during grad accum, same kernel is computed for *every* sub-batch which is wasteful. Caching of kernels will be fixed in the future.


## Overall Repository Structure
Please refer to official [S4 repo](https://github.com/HazyResearch/state-spaces) for further details on running experiments and the repo strucure.

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
If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{gupta2022dss,
  title={Diagonal State Spaces are as Effective as Structured State Spaces},
  author={Gupta, Ankit},
  journal={arXiv preprint arXiv:2203.14343},
  year={2022}
}

@article{goel2022sashimi,
  title={It's Raw! Audio Generation with State-Space Models},
  author={Goel, Karan and Gu, Albert and Donahue, Chris and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2202.09729},
  year={2022}
}

@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R\'e, Christopher},
  booktitle={The International Conference on Learning Representations ({ICLR})},
  year={2022}
}

@article{gu2021combining,
  title={Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers},
  author={Gu, Albert and Johnson, Isys and Goel, Karan and Saab, Khaled and Dao, Tri and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in neural information processing systems},
  volume={34},
  year={2021}
}

@article{gu2020hippo,
  title={HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in neural information processing systems},
  volume={33},
  year={2020}
}
```
