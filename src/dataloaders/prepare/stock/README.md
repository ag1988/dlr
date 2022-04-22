#### stock market data

Install `yfinance` via `pip install yfinance`.

Then run the following inside this directory to download+place the data for the relevant experiment. E.g. if you only want to work with the daily data you can ignore the rest.

```bash
# daily data for full available history
python -m download.py --only_sp500 --period max  --interval 1d --end 2022-04-20 --output_dir stock/day

# hourly data for last 730 days
python -m download.py --only_sp500 --period 730d --interval 1h --end 2022-04-20 --output_dir stock/hour

# every 2 minutes data for past 60 days
python -m download.py --only_sp500 --period 60d  --interval 2m --end 2022-04-20 --output_dir stock/minute
```

If you want to download more data, you can remove `--only_sp500` and use other flags described in `download.py`.

