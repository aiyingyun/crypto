import numpy as np
import pandas as pd

from lib.BaseStrategy import BaseStrategy


class FollowVolumeStrategy(BaseStrategy):
    def __init__(self, window=10, allow_short=False, price_col="close", percentage_threshold=0.0):
        """
        window   : rolling lookback for the moving averages (default 10 bars)
        allow_short : if False, negative signals are mapped to 0 (flat)
        price_col   : 'close' (typical) or 'open'
        epsilon  : optional buffer; require ratio - MA > epsilon to trigger
        """
        super().__init__(price_col, allow_short)
        self.window = int(window)
        self.percentage_threshold = float(percentage_threshold)

    @staticmethod
    def name():
        return "FollowVolume"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- ratios (robust to zero volumes) ----
        vol = pd.to_numeric(df.get("volume"), errors="coerce").replace(0, np.nan)
        qvol = pd.to_numeric(df.get("quote_volume"), errors="coerce").replace(0, np.nan)
        tbv = pd.to_numeric(df.get("taker_buy_volume"), errors="coerce")
        tbqv = pd.to_numeric(df.get("taker_buy_quote_volume"), errors="coerce")

        buy_ratio = (tbv / vol).clip(0, 1)  # fraction of base volume that was taker-buy
        quote_buy_ratio = (tbqv / qvol).clip(0, 1)  # fraction of notional that was taker-buy

        # ---- rolling means ----
        ma_buy = buy_ratio.rolling(self.window, min_periods=self.window).mean()
        ma_quote = quote_buy_ratio.rolling(self.window, min_periods=self.window).mean()

        # ---- conditions ----
        bull = (buy_ratio - ma_buy) > self.percentage_threshold * ma_buy
        bear = (buy_ratio - ma_buy) < -self.percentage_threshold * ma_buy
        bull &= (quote_buy_ratio - ma_quote) > self.percentage_threshold * ma_quote
        bear &= (quote_buy_ratio - ma_quote) < -self.percentage_threshold * ma_quote

        raw = np.where(bull, 1, np.where(bear, -1, 0))
        out = self.align_signal_ready_time(df, raw)
        return out
