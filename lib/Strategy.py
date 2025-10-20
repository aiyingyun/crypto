import numpy as np
import pandas as pd


class BaseStrategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


# ========== Example Strategy: SMA Crossover ==========
class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast=10, slow=30, allow_short=False):
        self.fast = int(fast)
        self.slow = int(slow)
        self.allow_short = bool(allow_short)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = pd.to_numeric(df["close"], errors="coerce")
        sma_fast = close.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = close.rolling(self.slow, min_periods=self.slow).mean()

        raw = np.where(sma_fast > sma_slow, -1,
                       np.where(sma_fast < sma_slow, 1, 0))
        if not self.allow_short:
            raw = np.where(raw > 0, 1, 0)
        return pd.Series(raw, index=df.index, dtype=float)
