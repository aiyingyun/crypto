import datetime
import os
import pandas as pd

DATA_PATH = "/Users/yingyunai/Desktop/crypto/binance/historical_binance/pv"


class DataLoader:
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
        data["open_dt"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
        data["close_dt"] = pd.to_datetime(data["close_time"], unit="ms", utc=True)

        data.sort_values("open_dt").reset_index(drop=True)
        front = ["open_dt", "close_dt"]
        rest = [c for c in data.columns if c not in front]
        data = data[front + rest]
        return data


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data(start_date="2025-06-21", end_date="2025-07-01", cryptocurrency_symbol="BTCUSDT")
