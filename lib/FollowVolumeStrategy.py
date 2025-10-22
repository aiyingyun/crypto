import numpy as np
import pandas as pd

from lib.BaseStrategy import BaseStrategy


class FollowVolumeStrategy(BaseStrategy):
    def __init__(self, window=40, allow_short=True, price_col="close", lower_threshold=0.5, higher_threshold=1.5):
        """
        window   : rolling lookback for the moving averages (default 10 bars)
        allow_short : if False, negative signals are mapped to 0 (flat)
        price_col   : 'close' (typical) or 'open'
        epsilon  : optional buffer; require ratio - MA > epsilon to trigger
        """
        super().__init__(price_col, allow_short)
        self.window = int(window)
        self.lower_threshold = float(lower_threshold)
        self.higher_threshold = float(higher_threshold)

    @staticmethod
    def name():
        return "FollowVolume"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- ratios (robust to zero volumes) ----
        volume = pd.to_numeric(df.get("volume"), errors="coerce").replace(0, np.nan)
        qvol = pd.to_numeric(df.get("quote_volume"), errors="coerce").replace(0, np.nan)
        tbv = pd.to_numeric(df.get("taker_buy_volume"), errors="coerce")
        tbqv = pd.to_numeric(df.get("taker_buy_quote_volume"), errors="coerce")

        buy_ratio = (tbv / volume).clip(0, 1)  # fraction of base volume that was taker-buy
        quote_buy_ratio = (tbqv / qvol).clip(0, 1)  # fraction of notional that was taker-buy

        # ---- rolling means ----
        ma_buy = buy_ratio.rolling(self.window, min_periods=self.window).mean()
        ma_quote = quote_buy_ratio.rolling(self.window, min_periods=self.window).mean()

        # ---- conditions ----
        bull = (buy_ratio >= self.lower_threshold * ma_buy) & (buy_ratio <= self.higher_threshold * ma_buy)
        bear = (buy_ratio <= self.lower_threshold * ma_buy) | (buy_ratio >= self.higher_threshold * ma_buy)

        bull &= (quote_buy_ratio >= self.lower_threshold * ma_quote) & (buy_ratio <= self.higher_threshold * ma_quote)
        bear &= (quote_buy_ratio <= self.lower_threshold * ma_quote) | (buy_ratio >= self.higher_threshold * ma_quote)

        raw = np.where(bull, 1, np.where(bear, -1, 0))
        out = self.align_signal_ready_time(df, raw)
        return out
