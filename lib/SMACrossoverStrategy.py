import numpy as np
import pandas as pd

from lib.BaseStrategy import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast=10, slow=30, allow_short=False, price_col="close"):
        super().__init__(price_col=price_col, allow_short=allow_short)
        self.fast = int(fast)
        self.slow = int(slow)

    @staticmethod
    def name():
        return "SMACrossover"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        sma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = px.rolling(self.slow, min_periods=self.slow).mean()

        raw = np.where(sma_fast > sma_slow, -1,
                       np.where(sma_fast < sma_slow, 1, 0))
        out = self.align_signal_ready_time(df, raw)
        return out
