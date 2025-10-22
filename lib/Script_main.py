# =========================== PROGRAMMATIC API ===========================
import os

from lib.Backtester import Backtester
from lib.DataLoader import DataLoader
from lib.FollowVolumeStrategy import FollowVolumeStrategy
from lib.PerformancePlotter import PerformancePlotter

# ============================== CLI GLUE ===============================
if __name__ == "__main__":
    symbol = "BTCUSDT"
    start = "2025-01-01"
    end = "2025-06-30"
    freq = "1m"
    resample = "30min"
    notional = 1.0
    tcbps = 0.0

    loader = DataLoader()
    df = loader.load_data(
        cryptocurrency_symbol=symbol,
        start_date=start,
        end_date=end,
        freq=freq,
    )

    # 2) Strategy
    strategy = FollowVolumeStrategy(allow_short=False)

    # 3) Backtest
    bt = Backtester(
        df=df,
        strategy=strategy,
        notional=notional,
        transaction_cost_bps=tcbps,
        resample_freq=resample,
    )
    result = bt.run()

    # 4) Outputs
    outdir = "./bt_out"
    os.makedirs(outdir, exist_ok=True)
    base = f"{symbol}_{start}_to_{end}_{resample or 'native'}_{strategy.name()}"
    csv_path = os.path.join(outdir, base + "_daily_metrics.csv")
    png_path = os.path.join(outdir, base + "_equity.png")

    PerformancePlotter().save_daily_csv(result["daily_df"], csv_path)
    PerformancePlotter().plot_equity(result["daily_df"], title="Equity Curve", save_path=png_path, show=True)

    print("Sharpe (ann.):", result["sharpe"])
    print("Total Return :", result["total_return"])
    print("Saved daily metrics CSV:", csv_path)
    print("Saved equity curve PNG :", png_path)

    results = {"result": result, "csv_path": csv_path, "png_path": png_path}
