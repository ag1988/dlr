""" Datasets for core experimental results """

from functools import partial
import os
import io
from pathlib import Path
import itertools
from collections import Counter
from tqdm.auto import tqdm 

import logging
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from PIL import Image  # Only used for Pathfinder
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import torchtext
from datasets import load_dataset, DatasetDict, Value

# from pytorch_lightning import LightningDataModule

from src.utils import permutations, is_list
import pickle

# Default data path is environment variable or hippo/data
if (default_data_path := os.getenv("DATA_PATH")) is None:
    default_data_path = Path(__file__).parent.parent.parent.absolute()
    default_data_path = default_data_path / "data"
else:
    default_data_path = Path(default_data_path).absolute()


# class SequenceDataset(LightningDataModule):
# [21-09-10 AG] Subclassing LightningDataModule fails due to trying to access _has_setup_fit. No idea why
class SequenceDataset:
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    # Since subclasses do not specify __init__ which is instead handled by this class
    # Subclasses can provide a list of default arguments which are automatically registered as attributes
    # TODO apparently there is a python 3.8 decorator that basically does this
    @property
    def init_defaults(self):
        return {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        # Add all arguments to self
        init_args = self.init_defaults
        init_args.update(
            dataset_cfg
        )  # TODO this overrides the default dict which is bad
        for k, v in init_args.items():
            setattr(self, k, v)

        self.init()  # Extra init stuff if desired # TODO get rid of this

        # train, val, test datasets must be set by class instantiation
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        
        self.shuffle_train = True

    def init(self):
        pass

    def setup(self):
        """This method should set self.dataset_train, self.dataset_val, and self.dataset_test"""
        raise NotImplementedError

    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(
                getattr(self, "seed", 42)
            ),  # PL is supposed to have a way to handle seeds properly, but doesn't seem to work for us
        )

    @staticmethod
    def collate_fn(batch, resolution=1):
        """batch: list of (x, y) pairs"""
        def _collate(batch, resolution=1):
            # From https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            elem = batch[0]
            if isinstance(elem, torch.Tensor):
                out = None
                if torch.utils.data.get_worker_info() is not None:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum(x.numel() for x in batch)
                    storage = elem.storage()._new_shared(numel)
                    out = elem.new(storage)
                x = torch.stack(batch, dim=0, out=out)
                if resolution is not None:
                    x = x[:, ::resolution] # assume length is first axis after batch
                return x
            else:
                return torch.tensor(batch)

        x, y = zip(*batch)
        # Drop every nth sample
        # x = torch.stack(x, dim=0)[:, ::resolution]
        # y = torch.LongTensor(y)
        # y = torch.tensor(y)
        # y = torch.stack(y, dim=0)
        x = _collate(x, resolution=resolution)
        y = _collate(y, resolution=None)
        return x, y

    def train_dataloader(self, train_resolution, eval_resolutions, **kwargs):
        if train_resolution is None:
            train_resolution = [1]
        if not is_list(train_resolution):
            train_resolution = [train_resolution]
        assert len(train_resolution) == 1, "Only one train resolution supported for now"
        
        if "shuffle" not in kwargs and self.shuffle_train is not None:
            kwargs["shuffle"] = self.shuffle_train
        return self._dataloader(
            self.dataset_train,
            resolutions=train_resolution,
            # shuffle=True,
            **kwargs,
        )[0]

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, train_resolution, eval_resolutions, **kwargs):
        if eval_resolutions is None:
            eval_resolutions = [1]
        if not is_list(eval_resolutions):
            eval_resolutions = [eval_resolutions]

        kwargs["shuffle"] = False if "shuffle" not in kwargs else kwargs["shuffle"]
        dataloaders = self._dataloader(
            dataset,
            resolutions=eval_resolutions,
            # shuffle=False, 
            **kwargs,
        )

        return (
            {
                str(res) if res > 1 else None: dl
                for res, dl in zip(eval_resolutions, dataloaders)
            }
            if dataloaders is not None
            else None
        )

    def _dataloader(self, dataset, resolutions, **loader_args):
        if dataset is None:
            return None

        return [
            torch.utils.data.DataLoader(
                dataset,
                collate_fn=partial(self.collate_fn, resolution=resolution)
                if self.collate_fn is not None
                else None,
                **loader_args,
            )
            for resolution in resolutions
        ]

    def __str__(self):
        return self._name_


class BIDMC(SequenceDataset):
    """BIDMC datasets for Respiratory Rate / Heart Rate / Oxygen Saturation regression"""

    _name_ = "bidmc"
    d_input = 2
    l_output = 0

    @property
    def d_output(self):
        return 2 if self.prediction else 1

    @property
    def init_defaults(self):
        return {
            "target": "RR",  # 'RR' | 'HR' | 'SpO2'
            "prediction": False,
            "reshuffle": True,
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        split = "reshuffle" if self.reshuffle else "original"
        # X: (dataset_size, length, d_input)
        # y: (dataset_size)
        X_train = np.load(self.data_dir / self.target / split / "trainx.npy")
        y_train = np.load(self.data_dir / self.target / split / "trainy.npy")
        X_val = np.load(self.data_dir / self.target / split / "validx.npy")
        y_val = np.load(self.data_dir / self.target / split / "validy.npy")
        X_test = np.load(self.data_dir / self.target / split / "testx.npy")
        y_test = np.load(self.data_dir / self.target / split / "testy.npy")

        if self.prediction:
            y_train = np.pad(X_train[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_val = np.pad(X_val[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
            y_test = np.pad(X_test[:, 1:, :], ((0, 0), (0, 1), (0, 0)))

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )

        self.dataset_val = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )

    def __str__(self):
        split = "reshuffle" if self.reshuffle else "original"
        return f"BIDMC{self.target}_{split}"

    
class MNIST(SequenceDataset):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    l_output = 0
    L = 784

    @property
    def init_defaults(self):
        return {
            "permute": True,
            "val_split": 0.1,
            "seed": 42,  # For train/val split
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / self._name_

        transform_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        ]  # (L, d_input)
        if self.permute:
            # below is another permutation that other works have used
            # permute = np.random.RandomState(92916)
            # permutation = torch.LongTensor(permute.permutation(784))
            permutation = permutations.bitreversal_permutation(self.L)
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: x[permutation])
            )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        transform = torchvision.transforms.Compose(transform_list)
        self.dataset_train = torchvision.datasets.MNIST(
            self.data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.MNIST(
            self.data_dir,
            train=False,
            transform=transform,
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CIFAR10(SequenceDataset):
    _name_ = "cifar"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "permute": None,
            "grayscale": False,
            "tokenize": False,  # if grayscale, tokenize into discrete byte inputs
            "augment": False,
            "cutout": False,
            "random_erasing": False,
            "val_split": 0.1,
            "seed": 42,  # For validation split
        }

    @property
    def d_input(self):
        if self.grayscale:
            if self.tokenize:
                return 256
            else:
                return 1
        else:
            assert not self.tokenize
            return 3

    def setup(self):
        if self.grayscale:
            preprocessors = [
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ]
            permutations = [
                torchvision.transforms.Lambda(
                    lambda x: x.view(1, 1024).t()
                )  # (L, d_input)
            ]

            if self.tokenize:
                preprocessors.append(
                    torchvision.transforms.Lambda(lambda x: (x * 255).long())
                )
                permutations.append(Rearrange("l 1 -> l"))
            else:
                preprocessors.append(
                    torchvision.transforms.Normalize(
                        mean=122.6 / 255.0, std=61.0 / 255.0
                    )
                )
        else:
            preprocessors = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                ),
            ]
            permutations = [
                torchvision.transforms.Lambda(
                    Rearrange("z h w -> (h w) z", z=3, h=32, w=32)
                )  # (L, d_input)
            ]

        # Permutations and reshaping
        if self.permute == "br":
            permutation = permutations.bitreversal_permutation(1024)
            print("bit reversal", permutation)
            permutations.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "snake":
            permutation = permutations.snake_permutation(32, 32)
            print("snake", permutation)
            permutations.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "hilbert":
            permutation = permutations.hilbert_permutation(32)
            print("hilbert", permutation)
            permutations.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "transpose":
            permutation = permutations.transpose_permutation(32, 32)
            transform = torchvision.transforms.Lambda(
                lambda x: torch.cat([x, x[permutation]], dim=-1)
            )
            permutations.append(transform)

        # Augmentation
        if self.augment:
            augmentations = [
                torchvision.transforms.RandomCrop(
                    32, padding=4, padding_mode="symmetric"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
            ]

        else:
            augmentations = []
        torchvision.transforms_train = (
            augmentations + preprocessors + permutations
        )
        torchvision.transforms_eval = preprocessors + permutations

        transform_train = torchvision.transforms.Compose(torchvision.transforms_train)
        transform_eval = torchvision.transforms.Compose(torchvision.transforms_eval)
        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}",
            train=True,
            download=True,
            transform=transform_train,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}", train=False, transform=transform_eval
        )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class CIFAR10Generation(SequenceDataset):
    """TODO there should be a way to combine this with main CIFAR class. the issue is making sure the torchvision.transforms are applied to output in the same way."""

    _name_ = "cifargen"

    @property
    def init_defaults(self):
        return {
            "transpose": False,
            "tokenize": True,
            "mixture": 0,
            "val_split": 0.02,
            "seed": 42,
        }

    @property
    def d_input(self):
        if not self.tokenize:
            return 1  # Returns None otherwise

    @property
    def d_output(self):
        return 256 if self.mixture == 0 else 3 * self.mixture

    @property
    def n_tokens(self):
        if self.tokenize:
            return 3 * 256 + 1

    @property
    def n_classes(self):  # TODO not used?
        return 10

    @property
    def permute(self):
        if self.transpose:  # R R ... G G ... B B ...
            return lambda x: rearrange(x, "... h w c -> ... (c h w) 1")
        else:  # R G B R G B ...
            return lambda x: rearrange(x, "... h w c -> ... (h w c) 1")

    @property
    def transforms0(self):
        """Transforms applied before permutation"""
        if self.tokenize:
            return torchvision.transforms.Lambda(
                lambda x: x + 1 + torch.arange(3) * 256
            )
        else:
            # return torchvision.transforms.Normalize(mean=127.5, std=127.5)
            return torchvision.transforms.Lambda(lambda x: (x.float() - 127.5) / 127.5)

    @property
    def transforms1(self):
        """Transforms applied after permutation"""
        if self.tokenize:
            return torchvision.transforms.Lambda(lambda x: x.squeeze(-1))
        else:
            return torchvision.transforms.Compose([])

    def setup(self):
        transforms = [
            torchvision.transforms.ToTensor(),  # (B, C, H, W)
            Rearrange("c h w -> h w c"),  # (B, H, W, C)
            torchvision.transforms.Lambda(
                lambda x: (x * 255).long()
            ),  # Convert back to ints
        ]
        transform = torchvision.transforms.Compose(transforms)

        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/cifar",
            train=True,
            download=True,
            transform=transform,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/cifar", train=False, transform=transform
        )
        self.split_train_val(self.val_split)

        def collate_batch(batch, resolution=1):
            """batch: list of (x, y) pairs"""
            inputs, labels = zip(*batch)
            x = torch.stack(inputs, dim=0)
            z = torch.LongTensor(labels)
            y = self.permute(x)
            x = self.transforms0(x)
            x = self.permute(x)
            x = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            x = self.transforms1(x)
            return x, y, z

        self.collate_fn = collate_batch

    def __str__(self):  # TODO not updated
        return f"{self._name_}"


class CIFAR10GenerationFactored(CIFAR10Generation):
    """Version of CIFAR-10 Density Estimation that keeps the sequence of length 1024 and factors the distribution over the 3 channels"""

    _name_ = "cifargenf"
    l_output = 1024  # Leaving this out or setting to None also works, to indicate that the entire length dimension is kept

    @property
    def init_defaults(self):
        return {
            "mixture": 0,
            "val_split": 0.02,
            "seed": 42,
        }

    @property
    def d_input(self):
        return 3

    @property
    def d_output(self):
        return 3 * 256 if self.mixture == 0 else 10 * self.mixture

    @property
    def permute(self):
        return lambda x: rearrange(x, "... h w c -> ... (h w) c")

    @property
    def transforms0(self):
        return torchvision.transforms.Lambda(lambda x: (x.float() - 127.5) / 127.5)
        # return torchvision.transforms.Normalize(mean=0.5, std=0.5)

    @property
    def transforms1(self):
        return torchvision.transforms.Compose([])


class Copying(SequenceDataset):
    _name_ = "copying"

    @property
    def init_defaults(self):
        return {
            "l_noise": 100,     # number of padding tokens
            "l_memorize": 10,   # number of tokens to memorize
            "n_tokens": 10,     # alphabet size
            "variable": False,  # Randomly distribute memorization tokens throughout sequence instead of frontloading them
            "n_samples": 50000,
            "val_split": 0.1,
        }

    @property
    def d_input(self):
        return self.n_tokens

    @property
    def d_output(self):
        return self.n_tokens

    @property
    def l_output(self):
        return self.l_memorize

    def setup(self):
        from .copying import copying_static_dataset

        self.dataset_train = copying_static_dataset(
            self.l_noise,
            self.l_memorize,
            self.n_tokens,
            self.variable,
            self.n_samples,
        )
        self.dataset_test = None
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{self._name_}{self.l_noise}{'v' if self.variable else ''}"

    
class Sequence1d(SequenceDataset):
    _name_ = "sequence1d"
        
    @property
    def init_defaults(self):
        return {
            "samples_per_epoch": 16000,
            "val_split": 0.1,
            "L": 1024,            
            "task": 'copy',
            "D": 4,                # mips: query/key/val dim
            "causal": True,        # mips: only prev keys matched
            "M": 32,               # masked_select: num tokens to copy
            "variable": True,      # masked_select: vary positions to copy across batches
            "consecutive": False,  # masked_select: use consecutive positions to copy
            "num_shifts": 8,       # shift task
        }

    def setup(self):
        from .sequence1d import Sequence1dTrainDataset, Sequence1dEvalDataset
        
        kwargs = {'L': self.L, 
                  'task': self.task, 
                  'D': self.D, 
                  'causal': self.causal, 
                  'M': self.M, 
                  'variable': self.variable, 
                  'consecutive': self.consecutive,
                  'num_shifts': self.num_shifts,
                 }
        self.dataset_train = Sequence1dTrainDataset(
            samples_per_epoch=int(self.samples_per_epoch*(1-self.val_split)), **kwargs
        )
        
        num_valid_samples = int(self.samples_per_epoch*self.val_split)# if int(self.val_split) < 1 else int(self.val_split)
        self.dataset_val = Sequence1dTrainDataset(
            samples_per_epoch=num_valid_samples, **kwargs
        )
        self.dataset_test = None
        self.shuffle_train = None  # IterableDataset doesn't accept "shuffle"
        
        if self.task == 'shift':
            self.d_input, self.d_output, self.l_output = 3, self.num_shifts, self.L
        elif 'masked_select' in self.task:
            self.d_input, self.d_output, self.l_output = 4, 1, self.M
        elif self.task == 'mips':
             self.d_input, self.d_output, self.l_output = 3*self.D+2, self.D, self.L
        else:
            self.d_input, self.d_output, self.l_output = 3, 1, self.L
        
    def __str__(self):
        return f"{self._name_}{self.L}{self.task}"

    
class Adding(SequenceDataset):
    _name_ = "adding"
    d_input = 2
    d_output = 1
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 1000,
            "n_samples": 50000,
            "val_split": 0.1,
        }

    def setup(self):
        from .adding import adding_static_dataset

        self.dataset_train = adding_static_dataset(self.l_max, self.n_samples)
        self.dataset_test = None
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{self._name_}{self.l_max}"


class SpeechCommands(SequenceDataset):
    _name_ = "sc"

    @property
    def init_defaults(self):
        return {
            "mfcc": False,
            "dropped_rate": 0.0,
            "length": 16000,
            "all_classes": False,
        }

    def init(self):
        if self.mfcc:
            self.d_input = 20
            self.L = 161
        else:
            self.d_input = 1
            self.L = self.length

        if self.dropped_rate > 0.0:
            self.d_input += 1

        self.d_output = 10 if not self.all_classes else 35
        self.l_output = 0

    def setup(self):
        from src.dataloaders.sc import _SpeechCommands

        # TODO refactor with data_dir argument
        self.dataset_train = _SpeechCommands(
            partition="train",
            length=16000,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
        )

        self.dataset_val = _SpeechCommands(
            partition="val",
            length=16000,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
        )

        self.dataset_test = _SpeechCommands(
            partition="test",
            length=16000,  # self.L,
            mfcc=self.mfcc,
            sr=1,
            dropped_rate=self.dropped_rate,
            path=default_data_path,
            all_classes=self.all_classes,
        )


""" LRA datasets """

class IMDB(SequenceDataset):
    _name_ = "imdb"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 4096,
            "level": "char",
            "min_freq": 15,
            "seed": 42,
            "val_split": 0.0,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 135,
            "n_workers": 4,  # Only used for tokenizing dataset before caching
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def init(self):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"
        assert self.level in [
            "word",
            "char",
        ], f"level {self.level} not supported"

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print(
            f"IMDB {self.level} level | min_freq {self.min_freq} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(
                test_size=self.val_split, seed=self.seed
            )
            self.dataset_train, self.dataset_val = (
                train_val["train"],
                train_val["test"],
            )

        def collate_batch(batch, resolution=1):
            xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, lengths

        self.collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(self._name_, cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.level == 'char'
            tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.min_freq,
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"


class TabularDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        format,
        col_idx=None,
        skip_header=False,
        csv_reader_params=None,
    ):
        """
        col_idx: the indices of the columns.
        """
        if csv_reader_params is None:
            csv_reader_params = {}
        format = format.lower()
        assert format in ["tsv", "csv"]
        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == "csv":
                reader = torchtext.utils.unicode_csv_reader(f, **csv_reader_params)
            elif format == "tsv":
                reader = torchtext.utils.unicode_csv_reader(
                    f, delimiter="\t", **csv_reader_params
                )
            else:
                reader = f
            if skip_header:
                next(reader)
            self._data = [
                line if col_idx is None else [line[c] for c in col_idx]
                for line in reader
            ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


# LRA tokenizer renames ']' to 'X' and delete parentheses as their tokenizer removes
# non-alphanumeric characters.
# https://github.com/google-research/long-range-arena/blob/264227cbf9591e39dd596d2dc935297a2070bdfe/lra_benchmarks/listops/input_pipeline.py#L46
def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()

class ListOps(SequenceDataset):
    _name_ = "listops"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 2048,
            "append_bos": False,
            "append_eos": True,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the listops-1000 directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch, resolution=1):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            return xs, ys, lengths

        self.collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t",
            keep_in_memory=True,
        )

        tokenizer = listops_tokenizer

        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                            str(Path(self.data_dir.stem) / image_path)
                            not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class PathFinder(SequenceDataset):
    _name_ = "pathfinder"
    d_output = 2
    l_output = 0
    
    @property
    def d_input(self):
        return 4 if self.all_corners else 1

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 32,
            "sequential": True,
            "tokenize": False,
            "pool": 1,
            "all_corners": False,     # stack 0,90,180,270 degree rotations
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split
        }

    def init(self):
        if self.data_dir is None:
            self.data_dir = (
                default_data_path / self._name_ / f"pathfinder{self.resolution}"
            )
    
    def rotations(self, x, dim=-3):
        assert x.shape[-2] == x.shape[-1], 'must be square'
        assert x.ndim >= 3
        rotations = [x] + [torchvision.transforms.functional.rotate(x, 90*i) for i in range(1,4)]
        return torch.cat(rotations, dim=dim)
    
    def default_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.all_corners:
            transform_list.append(self.rotations)    # (4 h w)
        if self.sequential:
            # If tokenize, it makes more sense to get rid of the channel dimension
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize and not self.all_corners
                else Rearrange("r h w -> (h w) r")
            )
        return torchvision.transforms.Compose(transform_list)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, download lra_release.gz from
            https://github.com/google-research/long-range-arena,
            then unzip it with tar -xvf lra_release.gz.
            Then point data_dir to the pathfinderX directory, where X is either 32, 64, 128, or 256.
            """
            )

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(self.data_dir, transform=self.default_transforms())
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )
        
        
class AAN(SequenceDataset):
    _name_ = "aan"
    d_output = 2  # Use accuracy instead of binary_accuracy
    l_output = 0

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # 'max_vocab': 100, # Full size 98
            "append_bos": False,
            "append_eos": True,
            "n_workers": 4,  # For tokenizing only
        }

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "eval", "test"]:
                split_path = self.data_dir / f"new_aan_pairs.{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    To get the dataset, download lra_release.gz from
                    https://github.com/google-research/long-range-arena,
                    then unzip it with tar -xvf lra_release.gz.
                    Then point data_dir to the tsv_data directory.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset, self.tokenizer, self.vocab = self.process_dataset()
        # self.vocab_size = len(self.vocab)
        print("AAN vocab size:", len(self.vocab))

        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch, resolution=1):
            xs1, xs2, ys = zip(
                *[
                    (data["input_ids1"], data["input_ids2"], data["label"])
                    for data in batch
                ]
            )
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(
                xs1, padding_value=self.vocab["<pad>"], batch_first=True
            )
            xs2 = nn.utils.rnn.pad_sequence(
                xs2, padding_value=self.vocab["<pad>"], batch_first=True
            )
            ys = torch.tensor(ys)
            # return xs1, xs2, ys, lengths1, lengths2

            # Concatenate two batches
            xs = torch.cat([xs1, xs2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, lengths

        self.collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )  # True)
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)

        tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(
            tokenize,
            remove_columns=["text1", "text2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        encode = lambda text: vocab(
            (["<bos>"] if self.append_bos else [])
            + text
            + (["<eos>"] if self.append_eos else [])
        )
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens1", "tokens2"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class PathFinderSegmentationDataset(torch.utils.data.Dataset):
    """Path Finder dataset with extra supervision."""

    def __init__(self, data_dir, input_transform, prepare_target):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            prepare_target: function for preparing target from path supervision file
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.input_transform = input_transform
        self.prepare_target = prepare_target
        samples = []
        path_list = sorted(
            list((self.data_dir / "metadata").glob("*.npy")),
            key=lambda path: int(path.stem),
        )
        assert path_list, "No metadata found"
        for metadata_file in path_list:
            for metadata in np.load(metadata_file).tolist():
                image_path = Path(metadata[0]) / metadata[1]   # 'imgs/0', 'sample_0.png'
                label = int(metadata[3])  
                segments_path = Path(metadata[0].replace('imgs/', 'paths/')) / metadata[1].replace('.png', '.pkl')
                # 'paths/0', 'sample_0.pkl'
                samples.append((image_path, label, segments_path))
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, segments_path = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / image_path, "rb") as f:
            orig_sample = Image.open(f).convert("L")  # Open in grayscale
        if self.input_transform is not None:
            sample = self.input_transform(orig_sample)
        
        segmentation = pickle.load(open(self.data_dir / segments_path,'rb'))
        target = self.prepare_target(segmentation, label, orig_sample)
        return sample, target


class PathFinderSegmentation(SequenceDataset):
    _name_ = "pathfinder_segmentation"
    d_output = 3
    l_output = None
    
    @property
    def d_input(self):
        return 4 if self.all_corners else 1

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 128,
            "sequential": True,
            "tokenize": False,
            "autoregressive": False,  # if sequential, pad by L 0's on right and take last L outputs
            "pool": 1,
            "all_corners": False,     # stack 0,90,180,270 degree rotations
            "val_split": 0.1,
            "test_split": 0.1,
            "seed": 42,  # Controls the train/val/test split
        }

    def init(self):
        if self.data_dir is None:
            self.data_dir = (
                default_data_path / self._name_ / f"pathfinder{self.resolution}_segmentation"
            )
        if self.autoregressive: 
            assert self.sequential
            self.l_output = self.resolution**2
            
    def rotations(self, x, dim=-3):
        assert x.shape[-2] == x.shape[-1], 'must be square'
        assert x.ndim >= 3
        rotations = [x] + [torchvision.transforms.functional.rotate(x, 90*i) for i in range(1,4)]
        return torch.cat(rotations, dim=dim)
    
    def zero_pad_right(self, x, dim=-2):
        assert dim < 0
        L = x.shape[dim]
        assert self.l_output == L
        return F.pad(x, (0,0)*abs(dim+1) + (0,L))
    
    def input_transform(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(
                Reduce(
                    "1 (h h2) (w w2) -> 1 h w",
                    "mean",
                    h2=self.pool,
                    w2=self.pool,
                )
            )
        if self.tokenize:
            transform_list.append(
                torchvision.transforms.Lambda(lambda x: (x * 255).long())
            )
        else:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.all_corners:
            transform_list.append(self.rotations)    # (4 h w)
        if self.sequential:
            # If tokenize, it makes more sense to get rid of the channel dimension
            transform_list.append(
                Rearrange("1 h w -> (h w)")
                if self.tokenize and not self.all_corners
                else Rearrange("r h w -> (h w) r")
            )
            if self.autoregressive:
                transform_list.append(
                    partial(self.zero_pad_right, dim=-1) 
                    if self.tokenize
                    else partial(self.zero_pad_right, dim=-2)
                )
        return torchvision.transforms.Compose(transform_list)
    
    def prepare_target(self, d, label=None, orig_sample=None):
        # d.keys(): 'segs', 'origin_tips', 'terminal_tips', 'marker_indices', 'image_size'
        [origin_mark_idx, terminal_mark_idx] = d['marker_indices']
        
        if label is not None:
            assert label == (origin_mark_idx == terminal_mark_idx)
        
        paths = []
        for i, path_inds in enumerate(d['segs']):
            path = np.zeros(d['image_size'], dtype=np.int16)
            path[path_inds] = 1 + (origin_mark_idx == terminal_mark_idx == i)
            paths.append(path)
        # 0: not on a long path, 1: on a long path but not any connecting path, 2: on a connecting path 
        label_mask = torch.LongTensor(np.maximum(*paths))        
        
        # sanity
        # label_mask = (torchvision.transforms.ToTensor()(orig_sample) > 0).long().squeeze(0)
        
        if self.sequential:
            label_mask = rearrange(label_mask, "h w -> (h w)")
        return label_mask
        
    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
            f"""
            Directory {str(self.data_dir)} not found.
            To get the dataset, generate pathfinder data using /src/dataloaders/prepare/pathfinder .
            Then point data_dir to the pathfinderX_segmentation directory, where X is either 128, or 256.
            """
            )

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        
        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderSegmentationDataset(self.data_dir, self.input_transform(), self.prepare_target)
        len_dataset = len(dataset)
        print(f'Total num of samples = {len_dataset}')
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        (
            self.dataset_train,
            self.dataset_val,
            self.dataset_test,
        ) = torch.utils.data.random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

    
class ListOpsSubTrees(SequenceDataset):
    _name_ = "listops_subtrees"
    l_output = 2048

    @property
    def init_defaults(self):
        return {
            "l_min": 1500,     # just a safety measure
            "l_max": 2048,
            # 'max_vocab': 20, # Actual size 18
            "n_workers": 4,  # Only used for tokenizing dataset
            "ignore_index": -100,
            "target": "SubTreeEvals",
        }

    @property
    def d_output(self):
        return len(self.vocab)
    
    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_min-{self.l_min}-l_max-{self.l_max}-target-{self.target}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name
        self.l_output = self.l_max
        
    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"subtrees_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "label_ids", "Id"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch, resolution=1):
            xs, ys, ids = zip(*[(data["input_ids"], data["label_ids"], data["Id"]) for data in batch])
            xs, ys, ids = map(torch.stack, (xs, ys, ids))
            assert xs.shape == ys.shape
            
            # replace self.vocab["<pad>"] by self.ignore_index to avoid computing loss at <pad>
            ys[ys == self.vocab["<pad>"]] = self.ignore_index
            return xs, ys, ids
        
        self.collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = self.cache_dir 
        if cache_dir is not None and cache_dir.is_dir():
            return self._load_from_cache(cache_dir)
        
        dataset = load_dataset(
            "csv",
            data_files={
                split: str(self.data_dir / f"subtrees_{split}.tsv") 
                for split in ['train', 'val', 'test']
            },
            delimiter="\t",
            keep_in_memory=True,
            download_mode="force_redownload",
        )
        
        from datasets import set_caching_enabled
        set_caching_enabled(False)
        
        columns_to_remove = set(dataset['train'].features.keys()).difference({'Source', self.target, 'SubTreeInds', 'Id'}) 
        dataset = dataset.remove_columns(columns_to_remove)
        
        tokenizer = listops_tokenizer
        
        def tokenize(example):
            in_tokens = tokenizer(example["Source"])
            out_tokens = tokenizer(example[self.target])
            inds = list(map(int, example["SubTreeInds"].split()))
            
            assert len(out_tokens) == len(inds), f"{len(out_tokens)}, {len(inds)}"
            assert self.l_min <= len(in_tokens) <= self.l_max, f'truncation not supported {self.l_min} <= {len(in_tokens)} <= {self.l_max}'
            
            _out_tokens = ["<pad>"]*len(in_tokens)
            # make predictions at "]"
            for ind, tok in zip(inds, out_tokens):
                _out_tokens[ind] = tok
            
            return {"in_tokens": in_tokens, "out_tokens": _out_tokens}
        
        logger = logging.getLogger(__name__)
        logger.info("Tokenizing...can take a long time and memory")
        
        dataset = dataset.map(
            tokenize,
            remove_columns=["Source", self.target, "SubTreeInds"],
            keep_in_memory=True,
            load_from_cache_file=False,
            # num_proc=max(self.n_workers, 1),
        )
        
        vocab = torchtext.vocab.build_vocab_from_iterator(
            itertools.chain(dataset["train"]["in_tokens"], dataset["train"]["out_tokens"]),
            specials=["<pad>", "<unk>"],
        )
        vocab.set_default_index(vocab["<unk>"])
        
        def numericalize(example):
            input_ids = vocab(
                example["in_tokens"] + ["<pad>"]*(self.l_max - len(example["in_tokens"]))
            )
            
            label_ids = vocab(
                example["out_tokens"] + ["<pad>"]*(self.l_max - len(example["in_tokens"]))
            )
            return {"input_ids": input_ids, "label_ids": label_ids}
        
        logger.info("Forming token ids...")
        dataset = dataset.map(
            numericalize,
            remove_columns=["in_tokens", "out_tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            # num_proc=max(self.n_workers, 1),
        )
        
        logger.info("caching...")
        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab
