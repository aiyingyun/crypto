import numpy as np
import pandas as pd


class BaseStrategy:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Must return columns:
          - 'signal' : {-1,0,1}
          - 'signal_ready_time' : tz-aware timestamp Series
        """
        raise NotImplementedError


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast=10, slow=30, allow_short=False, price_col="close"):
        self.fast = int(fast)
        self.slow = int(slow)
        self.allow_short = bool(allow_short)
        self.price_col = price_col  # "close" (typical) or "open"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        px = pd.to_numeric(df[self.price_col], errors="coerce")
        sma_fast = px.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = px.rolling(self.slow, min_periods=self.slow).mean()

        raw = np.where(sma_fast > sma_slow, -1,
                       np.where(sma_fast < sma_slow, 1, 0))
        if not self.allow_short:
            raw = np.where(raw > 0, 1, 0)

        signal = pd.Series(raw, index=df.index, dtype=float, name="signal")

        # --- timing: prefer shifted open_dt for close-based signals ---
        if "open_dt" in df.columns:
            if self.price_col == "close":
                ready = df["close_dt"]
            else:
                ready = df["open_dt"]
        else:
            ready = signal.index

        out = pd.DataFrame({
            "signal": signal,
            "signal_ready_time": pd.to_datetime(ready, utc=True)
        }).dropna(subset=["signal_ready_time"])

        return out
