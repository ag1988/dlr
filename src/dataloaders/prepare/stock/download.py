"""
Download historical stock market data.
source : https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data/notebook

python -m download.py --only_sp500 --period max  --interval 1d --end 2022-04-20 --output_dir stock/day
python -m download.py --only_sp500 --period 730d --interval 1h --end 2022-04-20 --output_dir stock/hour
python -m download.py --only_sp500 --period 60d  --interval 2m --end 2022-04-20 --output_dir stock/minute
"""

import os, contextlib, argparse, sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import yfinance as yf       # pip install yfinance

from pathlib import Path

# Default data path is environment variable or hippo/data
if (default_data_path := os.getenv("DATA_PATH")) is None:
    default_data_path = Path(__file__).parent.parent.parent.parent.parent.absolute()
    default_data_path = default_data_path / "data"
else:
    default_data_path = Path(default_data_path).absolute()


parser = argparse.ArgumentParser(description='Data download')
parser.add_argument('--only_sp500', action='store_true', help='Only download S and P 500 stocks.')
parser.add_argument('--period', default='max', type=str, choices=['max', '730d', '60d'], help='Period for which data downloaded') 
# 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
parser.add_argument('--interval', default='1d', type=str, choices=['1d', '1h', '2m'], help='interval for data queries') 
# 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
parser.add_argument('--end', default='2022-04-20', type=str, help='data after this not downloaded')        # for reproducibility
parser.add_argument('--output_dir', default='stock/day', type=str, help='relative to default_data_path')
parser.add_argument('--offset', default=0, type=int, help='only symbols[offset: offset+limit] downloaded')
parser.add_argument('--limit', default=-1, type=int, help='only symbols[offset: offset+limit] downloaded')
args = parser.parse_args()    

args.output_dir = os.path.join(default_data_path, args.output_dir)


# yfinance caps
if args.interval == '1h':
    assert args.period == '730d'    
if args.interval == '2m':
    assert args.period == '60d'


# symbols to download
if args.only_sp500:
    # list of companies with largest market caps
    SP500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    symbols = SP500['Symbol'].tolist()    
else:
    # download all NASDAQ traded symbols
    data = pd.read_csv("http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
    data_clean = data[data['Test Issue'] == 'N']
    symbols = data_clean['NASDAQ Symbol'].tolist()

print(f'total number of symbols to be downloaded = {len(symbols)}')


print(f'creating download dir {args.output_dir}')
os.makedirs(args.output_dir, exist_ok=True)

# download
offset = args.offset       
limit = args.limit if args.limit > 0 else len(symbols)
period = args.period
interval = args.interval

end = min(offset + limit, len(symbols))
is_valid = [False] * len(symbols)

# force silencing of verbose API
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        pbar = tqdm(range(offset, limit))
        for i in pbar:
            s = symbols[i]
            data = yf.download(s, period=period, interval=interval, end=args.end)
            orig_size = len(data)
            data.dropna(inplace=True)   # instead of imputing, we drop rows with NaNs
            if not len(data):
                continue
            is_valid[i] = True
            pbar.set_description(f'[keeping {s} {len(data)}/{orig_size}] [total valid symbols = {sum(is_valid)}]')
            
            # format data
            data.index.names = ['date']
            data.reset_index(inplace=True)
            data.columns = map(str.lower, data.columns)
            data.sort_values(by='date', inplace=True)
            
            # save data
            data.to_csv(f'{args.output_dir}/{s}.csv')

print("Done.\n")
