"""
Dataloader for stock forecasting.
Modified from https://github.com/zhouhaoyi/Informer2020
"""

from typing import List
import os
import glob
import numpy as np
import pandas as pd
from pandas.tseries import offsets
import torch
from torch.utils import data
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from src.dataloaders.datasets import SequenceDataset, default_data_path


def time_features(dates, freq="t"):
    """
    > `time_features` takes in a `dates` dataframe with a 'date' column and extracts the date down to `freq` where freq can be any of the following:
    > * m - [year, month]
    > * w - [year, month]
    > * d - [year, month, day, weekday]
    > * b - [year, month, day, weekday]
    > * h - [year, month, day, weekday, hour]
    > * t - [year, month, day, weekday, hour, minute]    
    """
    dates['date'] = pd.to_datetime(dates['date'])
    dates["year"] = dates.date.apply(lambda row: row.year, 1)
    dates["month"] = dates.date.apply(lambda row: row.month, 1)
    dates["day"] = dates.date.apply(lambda row: row.day, 1)
    dates["weekday"] = dates.date.apply(lambda row: row.weekday(), 1)
    dates["hour"] = dates.date.apply(lambda row: row.hour, 1)
    dates["minute"] = dates.date.apply(lambda row: row.minute, 1)
    freq_map = {
        "y": ["year"],
        "m": ["year", "month"],
        "w": ["year", "month"],
        "d": ["year", "month", "day", "weekday"],
        "b": ["year", "month", "day", "weekday"],
        "h": ["year", "month", "day", "weekday", "hour"],
        "t": ["year", "month", "day", "weekday", "hour", "minute"],
    }
    return dates[freq_map[freq.lower()]]


class StandardScaler:
    def __init__(self, epsilon=1e-5):
        self.mean = 0.0
        self.std = 1.0
        self.epsilon = epsilon

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0).clip(min=self.epsilon)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean


class StockDataset(Dataset):
    def __init__(
        self,
        file_path,
        flag="train",                # data split
        size=None,                   # [context_len, pred_len]            
        target="close",
        scale=True,
        mode='diff',                 # instead of predicting target, forecast difference from most recent target
        freq="t",
        cols=['year', 'month', 'day', 'weekday', 'hour', 'minute', 
              'high', 'low', 'open', 'close', 'volume'], # numeric features used for prediction
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 128-7
            self.pred_len = 7
        else:
            self.seq_len = size[0]
            self.pred_len = size[-1]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        
        self.target = target
        self.scale = scale
        self.freq = freq
        self.cols = cols
        self.mode = mode
        self.file_path = file_path
        self.__read_data__()

    def _borders(self, df_raw):
        num_train = int(len(df_raw) * 0.9)
        num_test = int(len(df_raw) * 0.05)
        num_val = len(df_raw) - num_train - num_test
        # temporal split : [train, val, test]
        start_idx = [0,         num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        end_idx =   [num_train, num_train + num_val,      len(df_raw)]
        return start_idx, end_idx

    def _process_columns(self, df_raw):
        # select features
        cols = self.cols.copy()
        cols.remove(self.target)
        return df_raw[cols + [self.target]]
    
    def __read_data__(self):
        df_raw = pd.read_csv(self.file_path)
        assert not df_raw.isnull().any().any(), 'data should already be imputed'
        
        # some other targets that might generalize better
        close = df_raw['close']
        prev_close = close.shift()
        df_raw['return'] = (close - prev_close) / prev_close.clip(lower=1e-4)
        df_raw.dropna(inplace=True)
        
        df_time = time_features(df_raw[["date"]].copy(), freq=self.freq)
        df_raw = pd.concat([df_raw, df_time], axis=1)
                
        # select features
        df_data = self._process_columns(df_raw)
        
        border1s, border2s = self._borders(df_raw)
        border1 = border1s[self.set_type]    # split start 
        border2 = border2s[self.set_type]    # split end
        
        # each stock normalized independenly
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2]        # input
        self.data_y = data[border1:border2, -1:]   # target

    def __getitem__(self, index):
        # seq_x: sb -----sl----------- se 0 ---pl--- 0
        #                      seq_y:  se   ---pl--- re
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x = np.concatenate(
            [seq_x, np.zeros((self.pred_len, self.data_x.shape[-1]))], axis=0
        )
        
        if self.mode == 'diff':        
            assert self.target in self.cols
            # difference from last known target
            seq_y = self.data_y[s_end:r_end] - self.data_y[s_end-1]
        else:
            seq_y = self.data_y[s_end:r_end]
        
        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        
        return torch.tensor(seq_x), torch.tensor(seq_y)  # [sl+pl,features], [pl,1]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    @property
    def d_input(self):
        return self.data_x.shape[-1]

    @property
    def d_output(self):
        return self.data_y.shape[-1]


class StocksDataset(ConcatDataset):
    def __init__(self, root_path, max_num_stocks=None, **kwargs):
        split = kwargs['flag']
        datasets = []
        pbar = tqdm(sorted(glob.glob(f'{root_path}/*.csv')))
        
        for file_path in pbar:
            try:
                ds = StockDataset(file_path, **kwargs)
                if not len(ds): continue
            except ValueError:
                continue
            datasets.append(ds)
            pbar.set_description(f'{split} : {os.path.basename(file_path)}')
            
            if max_num_stocks > 0 and len(datasets) >= max_num_stocks:
                break
            # in rare cases its possible a stock can appear in train but not in test
        
        super().__init__(datasets)
        print(f'\n {split} set size : {len(self)} \n'.upper())
    
    @property
    def d_input(self):
        return self.datasets[0].d_input

    @property
    def d_output(self):
        return self.datasets[0].d_output
    
    @property
    def pred_len(self):
        return self.datasets[0].pred_len


class StocksSequenceDataset(SequenceDataset):

    @property
    def d_input(self):
        return self.dataset_train.d_input

    @property
    def d_output(self):
        return self.dataset_train.d_output

    @property
    def l_output(self):
        return self.dataset_train.pred_len

    def setup(self):
        
        self.dataset_train = self._dataset_cls(
            root_path=self.data_dir,
            max_num_stocks=self.max_num_stocks,
            flag="train",
            size=self.size,
            target=self.target,
            scale=self.scale,
            mode=self.mode,
            freq=self.freq,
            cols=self.cols,
        )

        self.dataset_val = self._dataset_cls(
            root_path=self.data_dir,
            max_num_stocks=self.max_num_stocks,
            flag="val",
            size=self.size,
            target=self.target,
            scale=self.scale,
            mode=self.mode,
            freq=self.freq,
            cols=self.cols,
        )

        self.dataset_test = self._dataset_cls(
            root_path=self.data_dir,
            max_num_stocks=self.max_num_stocks,
            flag="test",
            size=self.size,
            target=self.target,
            scale=self.scale,
            mode=self.mode,
            freq=self.freq,
            cols=self.cols,
        )


class StocksDayForecast(StocksSequenceDataset):
    _name_ = "stocks_1d"

    _dataset_cls = StocksDataset

    init_defaults = {
        "data_dir": default_data_path / 'stock' / 'day',
        "max_num_stocks": -1,
        "size": [512-7],   # next 7d forecast based on past 505d
        "target": "close", # 'return'
        "scale": True,
        "mode": 'diff',    # ''
        "freq": "t",
        "cols": ['year', 'month', 'day', 'weekday', 'high', 'low', 'open', 'close', 'volume', 'return'],
    }


# class StocksHourForecast(StocksSequenceDataset):
#     _name_ = "stocks_1h"

#     _dataset_cls = StocksDataset

#     init_defaults = {
#         "data_dir": default_data_path / 'stock' / 'hour',
#         "max_num_stocks": -1,
#         "size": [256-24, 24],   # next 24h forecast based on past 232h
#         "target": "close", 
#         "scale": True,
#         "mode": 'diff',
#         "freq": "t,
#         "cols": ['year', 'month', 'day', 'weekday', 'hour',
#                  'high', 'low', 'open', 'close', 'volume']
#     }


# class StocksMinuteForecast(StocksSequenceDataset):
#     _name_ = "stocks_2m"

#     _dataset_cls = StocksDataset

#     init_defaults = {
#         "data_dir": default_data_path / 'stock' / 'minute',
#         "max_num_stocks": -1,
#         "size": [256-15, 15],   # next 15x2m forecast
#         "target": "close", 
#         "scale": True,
#         "mode": 'diff',
#         "freq": "t",
#         "cols": ['year', 'month', 'day', 'weekday', 'hour', 'minute', 
#                  'high', 'low', 'open', 'close', 'volume']
#     }

