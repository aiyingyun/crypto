import numpy as np
import pandas as pd


class BaseStrategy:
    def __init__(self, price_col, allow_short):
        self.price_col = price_col
        self.allow_short = bool(allow_short)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Must return columns:
          - 'signal' : {-1,0,1}
          - 'signal_ready_time' : tz-aware timestamp Series
        """
        raise NotImplementedError

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def align_signal_ready_time(self, df: pd.DataFrame, raw_signals) -> pd.DataFrame:
        if not self.allow_short:
            raw_signals = np.where(raw_signals > 0, 1, 0)

        signal = pd.Series(raw_signals, index=df.index, dtype=float, name="signal")

        # --- timing: prefer shifted open_dt for close-based signals ---
        if "open_dt" in df.columns:
            ready = df["close_dt"] if self.price_col == "close" else df["open_dt"]
        else:
            ready = signal.index

        out = pd.DataFrame({
            "signal": signal,
            "signal_ready_time": pd.to_datetime(ready, utc=True)
        }).dropna(subset=["signal_ready_time"])

        return out

