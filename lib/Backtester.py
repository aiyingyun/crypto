# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.Strategy import BaseStrategy


class Backtester:
    def __init__(self, df: pd.DataFrame, strategy: BaseStrategy,
                 notional=1.0, transaction_cost_bps=0.0, resample_freq=None):
        """
        df: native (minute) OHLCV with 'dt' or 'open_time' (ms)
        strategy: generates signals on the NATIVE df (granular)
        resample_freq: trading cadence (e.g., '30min','1H','1D').
                       If None -> trade on native cadence.
        """
        self.df = df
        self.strategy = strategy
        self.notional = float(notional)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.resample_freq = resample_freq

    # ---------------- public API ----------------

    def run(self):
        bar_df = self._build_trade_frame_from_native_signals(self.df)
        daily_df = self._daily_group(bar_df)
        if len(daily_df) > 1 and daily_df["daily_return"].std(ddof=1) > 0:
            sharpe = daily_df["daily_return"].mean() / daily_df["daily_return"].std(ddof=1) * np.sqrt(252.0)
        else:
            sharpe = float("nan")
        total_return = float((1.0 + daily_df["daily_return"]).prod() - 1.0)

        return {"bar_df": bar_df, "daily_df": daily_df, "sharpe": sharpe, "total_return": total_return}

    # ---------------- core helpers ----------------

    def _build_trade_frame_from_native_signals(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """
        1) Generate signals on native df
        2) Build trading bars
        3) Align 'last native signal up to trade bar END' -> then shift one bar to avoid look-ahead
        4) Compute returns/PnL/turnover on trading cadence
        """
        # --- 1) signals on native ---
        signals_native = self.strategy.generate_signals(df_native).astype(float)
        sig_df = df_native[["open_dt"]].copy()
        sig_df["signal_native"] = signals_native.values

        # --- 2) trading bars ---
        trade_bars = self._build_trade_bars(df_native)  # dt is bar END if resampled
        # trading returns (close-to-close)
        trade_close = pd.to_numeric(trade_bars["close"], errors="coerce")
        trade_ret = trade_close.pct_change().fillna(0.0)

        # --- 3) align native signals to trade bar end ---
        # take the last native signal up to each trade bar end
        # merge_asof: left=trade bars, right=native signals, direction='backward'
        aligned = trade_bars.merge(sig_df, on="open_dt", )
        # realized position = signal at previous TRADE bar close (one-bar delay)
        position = aligned["signal_native"].shift(1).fillna(0.0)

        # --- 4) turnover, costs, pnl ---
        turnover = (position - position.shift(1)).abs().fillna(position.abs())

        pnl_gross = self.notional * position * trade_ret
        cost_per_unit = (self.transaction_cost_bps / 1e4) * self.notional
        costs = cost_per_unit * turnover if self.transaction_cost_bps > 0 else 0.0
        pnl_net = pnl_gross - costs

        out = trade_bars.copy()
        out["ret"] = trade_ret.values
        out["signal_native_at_trade_close"] = aligned["signal_native"].values
        out["position"] = position.values
        out["turnover"] = turnover.values
        out["pnl_gross"] = pnl_gross.values
        out["costs"] = costs if np.isscalar(costs) else costs.values
        out["pnl"] = pnl_net.values
        return out

    def _daily_group(self, bar_df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(bar_df["open_dt"], utc=True)
        tmp = bar_df.copy()
        tmp["day"] = dt.dt.date

        daily = tmp.groupby("day", as_index=False).agg(
            daily_pnl=("pnl", "sum"),
            daily_turnover=("turnover", "sum"),
            bars=("ret", "count"),
        )
        daily["daily_return"] = daily["daily_pnl"] / self.notional
        daily["profit_over_turnover"] = 10000 * np.where(
            daily["daily_turnover"] > 0, daily["daily_pnl"] / daily["daily_turnover"], np.nan
        )
        daily["equity_curve"] = (1.0 + daily["daily_return"]).cumprod()
        return daily

    def _build_trade_bars(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """
        Build trading bars (OHLCV) at self.resample_freq.
        If resample_freq is None -> use native cadence as 'trade bars'.
        """
        if not self.resample_freq:
            cols = ["open_dt", "open", "high", "low", "close", "volume"]
            return df_native[cols].copy()

        x = df_native.set_index("open_dt")
        agg = {"open": "first", "high": "max",
               "low": "min", "close": "last",
               "close_dt": "last"}
        if "volume" in df_native.columns:
            agg["volume"] = "sum"

        trade = (
            x.resample(self.resample_freq, label="left", closed="left")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return trade
