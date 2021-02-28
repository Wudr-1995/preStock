# %%
import os
import datetime as dt
import time
from typing import Any, Dict, Optional, List
 
import requests
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import talib
import multiprocessing as mp
from requests.exceptions import ConnectionError, Timeout
 
# %%matplotlib inline
plt.style.use("fivethirtyeight")

## 撰写自定义函数，通过API获取数据
 
def fetch_trochil(url: str,
                  params: Dict[str, str],
                  attempt: int = 3,
                  timeout: int = 3) -> Dict[str, Any]:
    """装饰requests.get函数"""
    for i in range(attempt):
        try:
            resp = requests.get(url, params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()["data"]
            if not data:
                raise Exception("empty dataset")
            return data
        except (ConnectionError, Timeout) as e:
            print(e)
            i += 1
            time.sleep(i * 0.5)
 
 
def fetch_cnstocks(apikey: str) -> pd.DataFrame:
    """从蜂鸟数据获取A股产品列表"""
    url = "https://api.trochil.cn/v1/cnstock/markets"
    params = {"apikey": apikey}
 
    res = fetch_trochil(url, params)
 
    return pd.DataFrame.from_records(res)
 
 
def fetch_daily_ohlc(symbol: str,
                     date_from: dt.datetime,
                     date_to: dt.datetime,
                     apikey: str) -> pd.DataFrame:
    """从蜂鸟数据获取A股日图历史K线"""
    url = "https://api.trochil.cn/v1/cnstock/history"
    params = {
        "symbol": symbol,
        "start_date": date_from.strftime("%Y-%m-%d"),
        "end_date": date_to.strftime("%Y-%m-%d"),
        "freq": "daily",
        "apikey": apikey
    }
 
    res = fetch_trochil(url, params)
 
    return pd.DataFrame.from_records(res)
 
 
def fetch_index_ohlc(symbol: str,
                     date_from: dt.datetime,
                     date_to: dt.datetime,
                     apikey: str) -> pd.DataFrame:
    """获取股指的日图历史数据"""
    url = "https://api.trochil.cn/v1/index/daily"
    params = {
        "symbol": symbol,
        "start_date": date_from.strftime("%Y-%m-%d"),
        "end_date": date_to.strftime("%Y-%m-%d"),
        "apikey": apikey
    }
 
    res = fetch_trochil(url, params)
 
    return pd.DataFrame.from_records(res)

apikey = os.getenv("TROCHIL_API")  # use your apikey
cnstocks = fetch_cnstocks(apikey)
cnstocks

# 筛选前缀为'SH'的股票
cnstocks_shsz = cnstocks.query("symbol.str.startswith('SH')")
cnstocks_shsz

# %%time
 
# 下载2019年至今的历史数据
# 下载时剔除K线少于260个交易日的股票
date_from = dt.datetime(2019, 1, 1)
date_to = dt.datetime.today()
symbols = cnstocks_shsz.symbol.to_list()
min_klines = 260
 
# 逐个下载，蜂鸟数据的API没有分钟请求限制
# 先把数据存储在列表中，下载完成后再合并和清洗
ohlc_list = []
for symbol in symbols:
    try:
        ohlc = fetch_daily_ohlc(symbol, date_from, date_to, apikey)
        if ohlc is not None and len(ohlc) >= min_klines:
            ohlc.set_index("datetime", inplace=True)
            ohlc_list.append(ohlc)
    except Exception as e:
        pass
 
 
# CPU times: user 21.7 s, sys: 349 ms, total: 22 s
# Wall time: 49.3 s

