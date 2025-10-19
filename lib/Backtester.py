# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================= Strategy Interface =================
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

        raw = np.where(sma_fast > sma_slow, 1,
              np.where(sma_fast < sma_slow, -1, 0))
        if not self.allow_short:
            raw = np.where(raw > 0, 1, 0)
        return pd.Series(raw, index=df.index, dtype=float)


# =================== Backtest Engine ===================
class Backtester:
    def __init__(self, df: pd.DataFrame, strategy: BaseStrategy,
                 notional=1.0, transaction_cost_bps=0.0,
                 day_boundary_tz="UTC", resample_freq=None):
        self.df = df
        self.strategy = strategy
        self.notional = float(notional)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.day_boundary_tz = day_boundary_tz
        self.resample_freq = resample_freq

    def _ensure_dt(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "dt" in out.columns:
            out["dt"] = pd.to_datetime(out["dt"], utc=True)
        elif "open_time" in out.columns:
            out["dt"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
        else:
            raise ValueError("DataFrame must contain 'dt' or 'open_time' (milliseconds).")
        return out.sort_values("dt").reset_index(drop=True)

    def _maybe_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.resample_freq:
            return df
        req = {"open", "high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError(f"Resampling requires columns {req}.")
        x = df.set_index("dt")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in df.columns:
            agg["volume"] = "sum"
        y = x.resample(self.resample_freq).agg(agg).dropna(subset=["open", "high", "low", "close"])
        return y.reset_index()

    def _build_bar_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change().fillna(0.0)

        signals = self.strategy.generate_signals(df).astype(float)
        position = signals.shift(1).fillna(0.0)
        turnover = (position - position.shift(1)).abs().fillna(position.abs())

        pnl_gross = self.notional * position * ret
        cost_per_unit = (self.transaction_cost_bps / 1e4) * self.notional
        costs = cost_per_unit * turnover if self.transaction_cost_bps > 0 else 0.0
        pnl_net = pnl_gross - costs

        out = df.copy()
        out["ret"] = ret
        out["signal"] = signals
        out["position"] = position
        out["turnover"] = turnover
        out["pnl_gross"] = pnl_gross
        out["costs"] = costs
        out["pnl"] = pnl_net
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
        daily["profit_over_turnover"] = np.where(
            daily["daily_turnover"] > 0, daily["daily_pnl"] / daily["daily_turnover"], np.nan
        )
        daily["equity_curve"] = (1.0 + daily["daily_return"]).cumprod()
        return daily

    def run(self):
        df0 = self._ensure_dt(self.df)
        df = self._maybe_resample(df0)
        bar_df = self._build_bar_frame(df)
        daily_df = self._daily_group(bar_df)

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


