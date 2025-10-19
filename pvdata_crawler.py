
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:41:26 2025

@author: User
"""

import os
import datetime
import requests
import zipfile
from pathlib import Path
import pandas as pd



def download_and_extract_binance_data(destination,zip_url,zip_path,zip_filename):
    # 4. 下载ZIP文件
    print(f" ======= Downloading {zip_filename} =========")
    try:
        response = requests.get(zip_url, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {zip_filename}")
        
        # 5. 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print(f"Extracted {zip_filename}")
        
        # 6. 删除ZIP文件
        os.remove(zip_path)
        print(f"Deleted {zip_filename}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # 清理可能存在的部分下载文件
        if zip_path.exists():
            os.remove(zip_path)
            
def process_download(symbol,destination_path,name ='metrics',datlists = []):
    # 1. 计算前一天的日期
    prev_day = datetime.datetime.now() - datetime.timedelta(days=1)
    date_str = prev_day.strftime("%Y-%m-%d")
    
    # 2. 创建目标目录
    destination = Path(destination_path)
    destination.mkdir(parents=True, exist_ok=True)
    
    # 3. 构建下载URL和本地文件路径
    if name == 'metrics':
        base_url = "https://data.binance.vision/data/futures/um/daily/metrics/{}/".format(symbol)
    elif name == 'klines':
        base_url = "https://data.binance.vision/data/futures/um/daily/klines/{}/1m/".format(symbol)
    elif name == "spot_klines":
        base_url = "https://data.binance.vision/data/spot/daily/klines/{}/1m/".format(symbol)
        
    datlists.append(date_str)
    
    for i in datlists:
        if name == "metrics":
            zip_filename = "{}-metrics-{}.zip".format(symbol,i)
        else:
            zip_filename = "{}-1m-{}.zip".format(symbol,i)
        zip_url = base_url + zip_filename
        zip_path = destination / zip_filename
        download_and_extract_binance_data(destination,zip_url,zip_path,zip_filename)
   
    
def process_by_symbol(symbol):
    
    if symbol not in os.listdir(os.path.join(DATAPATH_CLOUD,"binance/historical_binance/metrics/")):
        os.mkdir(os.path.join(DATAPATH_CLOUD,"binance/historical_binance/metrics/{}".format(symbol)))
        
    metrics_path = os.path.join(DATAPATH_CLOUD , "binance/historical_binance/metrics/{}".format(symbol))
    filelists = [x.split("metrics-")[-1].split(".csv")[0] for x in os.listdir(metrics_path)]
    date_range = pd.date_range(start=datetime.date.today()-datetime.timedelta(30*4), end=datetime.date.today(), freq='D').strftime('%Y-%m-%d').tolist()
    toupdate = [x for x in date_range if x not in filelists]
    process_download(symbol,metrics_path,"metrics",toupdate)
    
    if symbol not in os.listdir(os.path.join(DATAPATH_CLOUD,"binance/historical_binance/pv/")):
        os.mkdir(os.path.join(DATAPATH_CLOUD,"binance/historical_binance/pv/{}".format(symbol)))
        
    pv_path = os.path.join(DATAPATH_CLOUD , "binance/historical_binance/pv/{}".format(symbol))
    filelists = [x.split("1m-")[-1].split(".csv")[0] for x in os.listdir(pv_path)]
    date_range = pd.date_range(start=datetime.date.today()-datetime.timedelta(30*4), end=datetime.date.today(), freq='D').strftime('%Y-%m-%d').tolist()
    toupdate = [x for x in date_range if x not in filelists]
    process_download(symbol,pv_path,"klines",toupdate)

def download_spot_by_symbol(symbol):
    if symbol not in os.listdir(os.path.join(DATAPATH_CLOUD,"binance/historical_binance_spot/")):
        os.mkdir(os.path.join(DATAPATH_CLOUD,"binance/historical_binance_spot/{}".format(symbol)))

    pv_path = os.path.join(DATAPATH_CLOUD, "binance/historical_binance_spot/{}".format(symbol))
    filelists = [x.split("1m-")[-1].split(".csv")[0] for x in os.listdir(pv_path)]
    date_range = pd.date_range(start=datetime.date(2017,8,17), end=datetime.date.today(),
                               freq='D').strftime('%Y-%m-%d').tolist()
    toupdate = [x for x in date_range if x not in filelists]
    process_download(symbol, pv_path, "spot_klines", toupdate)
    
if __name__ == "__main__":
    # 配置参数
    DATAPATH_CLOUD = "/Users/yingyunai/Desktop/crypto"
    print(DATAPATH_CLOUD)
    process_by_symbol('BTCUSDT')
  