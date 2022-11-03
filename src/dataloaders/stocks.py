"""
Dataloader for stock forecasting.
Modified from https://github.com/zhouhaoyi/Informer2020
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

from src.dataloaders.datasets import SequenceDataset, default_data_path


def load_raw_data(root_path, max_num_stocks=-1):
    """ root_path: path to dir containing cleaned .csv's
        returns a list of df's
    """
    files = sorted(glob.glob(f'{root_path}/*.csv'))
    files = files[:max_num_stocks] if max_num_stocks > 0 else files
    dfs_raw = []
    
    pbar = tqdm(files)
    for file_path in pbar:
        df_raw = pd.read_csv(file_path)
        assert not df_raw.isnull().any().any(), 'data should already be imputed'
        dfs_raw.append(df_raw)
        pbar.set_description(f'{os.path.basename(file_path)}')
    
    assert len(dfs_raw)
    return dfs_raw


def cutoff_dates(all_dates, split_sizes=[0.9,.05,.05]):
    """ Determines dates for a temporal split according to split_sizes.
        all_dates: list of dataframes with a 'date' column.
        returns date ranges for train, val, test.
    """
    all_dates = pd.concat(all_dates)
    all_dates['date'] = pd.to_datetime(all_dates['date'])
    all_dates.sort_values(by=['date'], inplace=True, ascending=True)
    all_dates.reset_index(drop=True, inplace=True)
    n = len(all_dates)
    # train - val - test
    train_start = 0
    train_end = val_start = train_start + int(n * split_sizes[0])  # excl                           
    val_end = test_start = val_start + int(n * split_sizes[1])
    test_end = min(test_start + int(n * split_sizes[2]), n-1)

    date_start = [all_dates.date.iloc[idx] for idx in [train_start, val_start, test_start]]
    date_end = [all_dates.date.iloc[idx] for idx in [train_end, val_end, test_end]]

    return date_start, date_end


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
        cutoffs,                     # datetime ranges (temporal split)
        flag="train",                # data split
        size=[512-7,7],              # [context_len, pred_len]            
        target="return",
        freq="d",
        cols=['year', 'month', 'day', 'weekday', 'hour', 'minute', 
              'high', 'low', 'open', 'close', 'volume', 'return'], # numeric features used for prediction
    ):
        self.seq_len, self.pred_len = size        
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.cutoffs = cutoffs     
        self.target = target
        self.freq = freq
        self.cols = cols
    
    def process_columns(self, df_raw):
        df_raw = df_raw.copy()
        close = df_raw['close']
        prev_close = close.shift()
        df_raw['return'] = (close - prev_close) / prev_close.clip(lower=1e-5)
        df_raw = df_raw.dropna(inplace=True)
        
        # incude time info
        df_time = time_features(df_raw[["date"]].copy(), freq=self.freq)
        df_raw = pd.concat([df_raw, df_time], axis=1)
                
        # select features
        cols = self.cols.copy()
        cols.remove(self.target)
        return df_raw[cols + [self.target]]
    
    def borders(self, df):
        start_date, end_date = self.cutoffs
        num_train, num_val, num_test = [
            pd.to_datetime(df['date']).between(start_date[i], end_date[i], inclusive='left').sum() 
            for i in range(3)
        ]
        # temporal split : [train, val, test]
        start_idx = [0,         num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        end_idx =   [num_train, num_train + num_val,      len(df_raw)]
        return start_idx, end_idx
    
    def get_split_data(self, df, set_type=0):
        border1s, border2s = self.borders(df)
        border1 = border1s[set_type]    # split start 
        border2 = border2s[set_type]    # split end
        return df[border1 : border2]
        
    def prepare_data(self, df, scalar=None):
        df_split = self.get_split_data(df, self.set_type)
        
        if scaler is not None:
            data = scaler.transform(df_split.values)
        else:
            data = df_split.values
        
        self.data_x = data           # input
        self.data_y = data[:, -1:]   # target : rightmost col

    def __getitem__(self, index):
        # seq_x: sb -----sl----------- se 0 ...pl... 0
        #                      seq_y:  se   ---pl--- re
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        
        # seq_x = self.data_x[s_begin:r_end]
        # seq_x[s_end:,-1:] = 0  # mask out just the target from future 
        
        # TODO: we're masking out all features from future but should keep time features 
        seq_x = np.concatenate(
            [seq_x, np.zeros((self.pred_len, self.data_x.shape[-1]))], axis=0
        )
        
        seq_y = self.data_y[s_end:r_end]
        
        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        
        return torch.tensor(seq_x), torch.tensor(seq_y)  # [sl+pl,features], [pl,1]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, scalar):
        return scaler.inverse_transform(data)

    @property
    def d_input(self):
        return self.data_x.shape[-1]

    @property
    def d_output(self):
        return self.data_y.shape[-1]

    
class StocksDataset(ConcatDataset):
    def __init__(self, dfs_raw, **kwargs):
        split = kwargs['flag']
        
        # determine datetime ranges for a temporal data split
        cutoffs = cutoff_dates([dfs_raw[['date']] for df in dfs_raw])
        
        stock = StockDataset(cutoffs, **kwargs)
        
        # add additional features
        dfs_raw = list(map(stock.process_columns, dfs_raw))
        
        # train a scalar
        train_data = pd.concat([stock.get_split_data(df, 0) for df in dfs_raw])
        scaler = StandardScaler()
        scaler.fit(train_data.values)
        
        datasets = []
        for df in tqdm(dfs_raw, desc=f"{split}"):
            try:
                ds = StockDataset(cutoffs, **kwargs)
                ds.prepare_data(df, scalar)
                if not len(ds): continue
            except ValueError:
                continue
            datasets.append(ds)  
            # note: its possible a stock can appear in train but not in test
        
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
        
        # load cleaned data
        dfs_raw = load_raw_data(self.data_dir, self.max_num_stocks)
        
        self.dataset_train, self.dataset_val, self.dataset_test = (
            StocksDataset(
                dfs_raw,
                flag="train",
                size=self.size,
                target=self.target,
                scale=self.scale,
                freq=self.freq,
                cols=self.cols,
            ) 
            for split in ['train', 'val', 'test']
        )


class StocksDayForecast(StocksSequenceDataset):
    _name_ = "stocks_1d"

    init_defaults = {
        "data_dir": default_data_path / 'stock' / 'day',
        "max_num_stocks": -1,
        "size": [512-7],   # next 7d forecast based on past 505d
        "target": "return",
        "scale": True,
        "freq": "d",
        "cols": ['year', 'month', 'day', 'weekday', 'return'],
        # "cols": ['year', 'month', 'day', 'weekday', 'high', 'low', 'open', 'close', 'volume', 'return'],
    }


    
# CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=s4-stocks-day dataset.max_num_stocks=100 dataset.target='close'
    
    

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
#         "cols": ['high', 'close', 'volume', 'open', 'low', 'change']
#     }


# class StocksMinuteForecast(StocksSequenceDataset):
#     _name_ = "stocks_2m"

#     _dataset_cls = StocksDataset

#     init_defaults = {
#         "data_dir": default_data_path / 'stock' / 'minute',
#         "max_num_stocks": -1,
#         "size": [512-15, 15],   # next 15x2m forecast
#         "target": "close", 
#         "scale": True,
#         "mode": 'diff',
#         "freq": "t",
#         "cols": ['year', 'month', 'day', 'weekday', 'hour', 'minute', 
#                  'high', 'low', 'open', 'close', 'volume']
#     }
