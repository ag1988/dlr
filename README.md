# Code will be released sometime this week.

## Diagonal State Spaces (DSS)

This repository is built on a fork of the official [S4 repo](https://github.com/HazyResearch/state-spaces) & contains the accompanying code for the paper:

> **Diagonal State Spaces are as Effective as Structured State Spaces**\
> Ankit Gupta\
> Paper: https://arxiv.org/pdf/2203.14343

## Table of Contents
- [Repository Setup](#setup)
- DSS / S4
  - [Experiments](#s4-experiments)
  - [Training](#training)
  - [Models](#models)
- [SaShiMi](sashimi/README.md#sashimi)
- [Repository Structure](#overall-repository-structure)
- [Citation](#citation)

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

### Diagonal State Spaces (DSS)

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

#### Approximate test accuracy (at best validation checkpoint) & training time on A100:
| experiment | listops  | imdb |  aan | lra-cifar | pathfinder | pathx |  sc  |
| ---        |    ---   |  --- |  --- |   --- |    ---     |  ---   | ---  |
| acc        | 58.2     | 76.3 |  87.8| 85.7  | 84.6       | 85    | 97.7 |
| time       | 2h       |  20m |  <9h | <6h   |  9h        |  36h   | <19h |

These metrics can vary depending on GPU. On pathx, loss should start to decrease around global step 90k (10h).

#### Tuning
You can directly tinker with hyperparameters via flags. E.g. 
```bash
python -m train wandb=null model=dss experiment=s4-lra-cifar train.seed=42 scheduler.patience=15 trainer.max_epochs=250
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

---
---

# S4

### Cauchy Kernel

A core operation of S4 is the "Cauchy kernel" described in the [paper](https://arxiv.org/abs/2111.00396).
The implementation of this requires one of two methods:

#### Custom CUDA Kernel

This version is faster but requires manual compilation on each machine.
Run `python setup.py install` from the directory `extensions/cauchy/`.

#### Pykeops

This version is provided by the [pykeops library](https://www.kernel-operations.io/keops/index.html).
Installation usually works out of the box with `pip install pykeops cmake` which are provided in the requirements file.

Note that running in a Colab requires installing a different pip package; instructions can be found in the pykeops documentation.

## S4 Experiments

This section describes how to use the latest S4 model and reproduce experiments immediately.
More detailed descriptions of the infrastructure are in the subsequent sections.

### Structured State Space (S4)

The S4 module is found at
`src/models/sequence/ss/s4.py`.

For users who would like to import a single file that has the self-contained S4 layer,
a standalone version can be found at `src/models/sequence/ss/standalone/s4.py`.

### Testing

For testing, we frequently use synthetic datasets or the Permuted MNIST dataset.
This can be run with `python -m train wandb=null pipeline=mnist model=s4`, which should get to around 90% after 1 epoch which takes 1-3 minutes depending on GPU.

### Long Range Arena (LRA)

```
python -m train wandb=null experiment=s4-lra-listops
python -m train wandb=null experiment=s4-lra-imdb
python -m train wandb=null experiment=s4-lra-cifar
python -m train wandb=null experiment=s4-lra-aan
python -m train wandb=null experiment=s4-lra-pathfinder
python -m train wandb=null experiment=s4-lra-pathx
```

Note that these experiments may take different amounts of time to train. IMDB should take 1-2 hours, while Path-X will take several epochs to take off and take over a day to train to completion.

### CIFAR-10

```
python -m train wandb=null experiment=s4-cifar
```

The above command line reproduces our best sequential CIFAR model. Decreasing the model size should yield close results, e.g. decreasing the hidden dimension and number of layers with `model.d_model=512 model.n_layers=4`.

### Speech Commands

The Speech Commands dataset that our [baselines](https://arxiv.org/abs/2005.08926) [use](https://arxiv.org/abs/2102.02611) is a modified smaller 10-way classification task.

```
python -m train wandb=null experiment=s4-sc
```

To use the original version with the full 35 classes, pass in `dataset.all_classes=true`

### WikiText-103

```
python -m train wandb=null experiment=s4-wt103
```

The default settings require 8 GPUs with 32GB memory. Modifications can be made by decreasing batch size and accumulating gradients, e.g. `loader.batch_size=4 trainer.accumulate_grad_batches=2`


### Optimizer Hyperparameters

One notable difference in this codebase is that some S4 parameters use different optimizer hyperparameters. In particular, the SSM kernel is particularly sensitive to the A, B, and dt parameters, so the optimizer settings for these parameters are usually fixed to learning rate 0.001 and weight decay 0.

Our logic for setting these parameters can be found in the `OptimModule` class under `src/models/sequence/ss/kernel.py` and the corresponding optimizer hook in `SequenceLightningModule.configure_optimizers` under `train.py`.

## Training

The core training infrastructure of this repository is based on [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) with a configuration scheme based on [Hydra](https://hydra.cc/docs/intro/).
The structure of this integration largely follows the Lightning+Hydra integration template described in https://github.com/ashleve/lightning-hydra-template.

The main experiment entrypoint is `train.py` and configs are found in `configs/`.
In brief, the main config is found at `configs/config.yaml`, which is combined with other sets of configs that can be passed on the command line, to define an overall YAML config.
Most config groups define one single Python object (e.g. a PyTorch nn.Module).
The end-to-end training pipeline can broken down into the following rough groups, where group XX is found under `configs/XX/`:
```
model: the sequence-to-sequence model backbone (e.g. a src.models.sequence.SequenceModel)
dataset: the raw dataset (data/target pairs) (e.g. a pytorch Dataset)
loader: how the data is loaded (e.g. a pytorch DataLoader)
encoder: defines a Module that interfaces between data and model backbone
decoder: defines a Module that interfaces between model backbone and targets
task: specifies loss and metrics
```
Default combinations of dataset+loader+encoder+decoder+task are further consolidated into groups called `pipelines`.

A run can be performed by passing in a pipeline config, model config,
and any additional arguments modifying the default configurations.
A simple example experiment is
```
python -m train pipeline=mnist dataset.permute=True model=s4 model.n_layers=3 model.d_model=128 model.norm=batch model.prenorm=True wandb=null
```
This uses the permuted sequential MNIST task and uses an s4 model with a specified number of layers, backbone dimension, and normalization type.


### Hydra

It is recommended to read the Hydra documentation to fully understand the configuration framework. For help launching specific experiments, please file an Issue.

### Registries

This codebase uses a modification of the hydra `instantiate` utility that provides shorthand names of different classes, for convenience in configuration and logging.
The mapping from shorthand to full path can be found in `src/utils/registry.py`.

### WandB

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of `configs/config.yaml` (or pass it on the command line `python -m train .... wandb.project=s4`).

Set `wandb=null` to turn off WandB logging.


## Models

This repository provides a modular and flexible implementation of sequence models at large.

#### SequenceModule
SequenceModule `src/models/sequence/base.py` is the abstract interface that all sequence models adhere to.
In this codebase, sequence models are defined as a sequence-to-sequence map of shape `(batch size, sequence length, input dimension)` to `(batch size, sequence length, output dimension)`.

The SequenceModule comes with other methods such as `step` which is meant for autoregressive settings, and logic to carry optional hidden states (for stateful models such as RNNs or S4).

#### SequenceModel
SequenceModel `src/models/sequence/model.py` is the main backbone with configurable options for residual function, normalization placement and type, etc.
SequenceModel accepts a black box config for a layer. Compatible layers are SequenceModules (i.e. composable sequence transformations) found under `src/models/sequence/`.


### Baselines
Other sequence models are easily incorporated into this repository,
and several other baselines have been ported.

These include CNNs such as the [WaveGAN Discriminator](https://arxiv.org/abs/1802.04208) and [CKConv](https://arxiv.org/abs/2102.02611) and continuous-time/RNN models such as [UnICORNN](https://arxiv.org/abs/2102.02611) and [LipschitzRNN](https://arxiv.org/abs/2006.12070).

```
python -m train dataset=mnist model={ckconv,unicornn}
```



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
