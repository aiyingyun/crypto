import datetime
import os

import pandas as pd

DATA_PATH = "/Users/yingyunai/Desktop/crypto/binance/historical_binance/pv"


class CryptoDataLoader:
    def __init__(self, ):
        self.data_source = DATA_PATH

    def load_data(self, cryptocurrency_symbol, start_date=None, end_date=None, freq='1m'):
        date_range = pd.date_range(start=datetime.datetime.strptime(start_date, "%Y-%m-%d"),
                                   end=datetime.datetime.strptime(end_date, "%Y-%m-%d"),
                                   freq='D').strftime('%Y-%m-%d').tolist()
        df_all = []
        for date in date_range:
            f_name = f"{cryptocurrency_symbol}-{freq}-{date}.csv"
            df = pd.read_csv(os.path.join(f"{self.data_source}/{cryptocurrency_symbol}", f_name))
            df_all.append(df)
        data = pd.concat(df_all, ignore_index=True)
        return data


if __name__ == "__main__":
    loader = CryptoDataLoader()
    loader.load_data(start_date="2025-06-21", end_date="2025-07-01", cryptocurrency_symbol="BTCUSDT")
