# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lib.Strategy import BaseStrategy


class Backtester:
    def __init__(self, df: pd.DataFrame, strategy: BaseStrategy,
                 notional=1.0, transaction_cost_bps=0.0,
                 day_boundary_tz="UTC", resample_freq=None):
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
        self.day_boundary_tz = day_boundary_tz
        self.resample_freq = resample_freq

    # ---------------- core helpers ----------------
    @staticmethod
    def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "dt" in out.columns:
            out["dt"] = pd.to_datetime(out["dt"], utc=True)
        elif "open_time" in out.columns:
            out["dt"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
        else:
            raise ValueError("DataFrame must contain 'dt' or 'open_time' (milliseconds).")
        return out.sort_values("dt").reset_index(drop=True)

    def _build_trade_bars(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """
        Build trading bars (OHLCV) at self.resample_freq.
        If resample_freq is None -> use native cadence as 'trade bars'.
        """
        if not self.resample_freq:
            # trade on native cadence
            cols = ["dt", "open", "high", "low", "close", "volume"]
            missing = [c for c in ["open", "high", "low", "close"] if c not in df_native.columns]
            if missing:
                raise ValueError(f"Native trading requires OHLC columns: {missing}")
            return df_native[cols].copy()

        req = {"open", "high", "low", "close"}
        if not req.issubset(df_native.columns):
            raise ValueError(f"Resampling requires columns {req}.")

        x = df_native.set_index("dt")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df_native.columns:
            agg["volume"] = "sum"

        # label/closed='right' => bar timestamp = bar end
        trade = (
            x.resample(self.resample_freq, label="right", closed="right")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return trade

    def _build_trade_frame_from_native_signals(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """
        1) Generate signals on native df
        2) Build trading bars
        3) Align 'last native signal up to trade bar END' -> then shift one bar to avoid look-ahead
        4) Compute returns/PnL/turnover on trading cadence
        """
        # --- 1) signals on native ---
        signals_native = self.strategy.generate_signals(df_native).astype(float)
        sig_df = df_native[["dt"]].copy()
        sig_df["signal_native"] = signals_native.values

        # --- 2) trading bars ---
        trade_bars = self._build_trade_bars(df_native)  # dt is bar END if resampled
        # trading returns (close-to-close)
        trade_close = pd.to_numeric(trade_bars["close"], errors="coerce")
        trade_ret = trade_close.pct_change().fillna(0.0)

        # --- 3) align native signals to trade bar end ---
        # take the last native signal up to each trade bar end
        # merge_asof: left=trade bars, right=native signals, direction='backward'
        aligned = pd.merge_asof(
            trade_bars.sort_values("dt"),
            sig_df.sort_values("dt"),
            left_on="dt", right_on="dt",
            direction="backward"
        )
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
        dt = pd.to_datetime(bar_df["dt"], utc=True)
        if self.day_boundary_tz and self.day_boundary_tz.upper() != "UTC":
            dt = dt.dt.tz_convert(self.day_boundary_tz)
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

    # ---------------- public API ----------------
    def run(self):
        # Ensure dt and sort
        df_native = self._ensure_dt(self.df)

        # Build trading frame from native signals
        bar_df = self._build_trade_frame_from_native_signals(df_native)

        # Daily metrics
        daily_df = self._daily_group(bar_df)

        # Sharpe (annualized 252)
        if len(daily_df) > 1 and daily_df["daily_return"].std(ddof=1) > 0:
            sharpe = daily_df["daily_return"].mean() / daily_df["daily_return"].std(ddof=1) * np.sqrt(252.0)
        else:
            sharpe = float("nan")
        total_return = float((1.0 + daily_df["daily_return"]).prod() - 1.0)

        return {"bar_df": bar_df, "daily_df": daily_df, "sharpe": sharpe, "total_return": total_return}

    @staticmethod
    def save_daily_csv(daily_df: pd.DataFrame, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        daily_df.to_csv(out_path, index=False)
        return out_path

    @staticmethod
    def plot_equity(daily_df: pd.DataFrame, title="Equity Curve", save_path: str | None = None, show: bool = True):
        plt.figure(figsize=(10, 5))
        x = pd.to_datetime(daily_df["day"])
        y = daily_df["equity_curve"]
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Equity (Start = 1.0)")
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
