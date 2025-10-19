# =========================== PROGRAMMATIC API ===========================
import os

from lib.Backtester import Backtester, SMACrossoverStrategy
from lib.CryptoDataLoader import CryptoDataLoader


def main(
        symbol: str,
        start: str,
        end: str,
        *,
        freq: str = "1m",
        resample: str | None = None,  # e.g., "15min", "1H", "1D"
        tz: str = "UTC",  # daily boundary timezone
        fast: int = 10,
        slow: int = 30,
        allow_short: bool = False,
        notional: float = 1.0,
        tcbps: float = 0.0,  # transaction cost (bps) per unit turnover
        outdir: str = "./bt_out",
        show: bool = True,
) -> dict:
    """
    Programmatic entry point. Returns a dict with:
      - 'result' (bar_df, daily_df, sharpe, total_return)
      - 'csv_path'
      - 'png_path'
    """
    # 1) Load with your existing loader
    loader = CryptoDataLoader()
    df = loader.load_data(
        cryptocurrency_symbol=symbol,
        start_date=start,
        end_date=end,
        freq=freq,
    )

    # 2) Strategy
    strategy = SMACrossoverStrategy(fast=fast, slow=slow, allow_short=allow_short)

    # 3) Backtest
    bt = Backtester(
        df=df,
        strategy=strategy,
        notional=notional,
        transaction_cost_bps=tcbps,
        day_boundary_tz=tz,
        resample_freq=resample,
    )
    result = bt.run()

    # 4) Outputs
    os.makedirs(outdir, exist_ok=True)
    base = f"{symbol}_{start}_to_{end}_{resample or 'native'}"
    csv_path = os.path.join(outdir, base + "_daily_metrics.csv")
    png_path = os.path.join(outdir, base + "_equity.png")

    Backtester.save_daily_csv(result["daily_df"], csv_path)
    Backtester.plot_equity(result["daily_df"], title="Equity Curve", save_path=png_path, show=show)

    print("Sharpe (ann.):", result["sharpe"])
    print("Total Return :", result["total_return"])
    print("Saved daily metrics CSV:", csv_path)
    print("Saved equity curve PNG :", png_path)

    return {"result": result, "csv_path": csv_path, "png_path": png_path}


# ============================== CLI GLUE ===============================
if __name__ == "__main__":
    out = main(
        symbol="BTCUSDT",
        start="2025-06-21",
        end="2025-07-01",
        freq="1m",
        resample="15min",
        tz="America/New_York",
        fast=10,
        slow=30,
        allow_short=False,
        notional=1.0,
        tcbps=0.0,
        outdir="./bt_out",
        show=True,
    )
