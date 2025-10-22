# =========================== PROGRAMMATIC API ===========================
import os

from lib.Backtester import Backtester
from lib.DataLoader import DataLoader
from lib.PerformancePlotter import PerformancePlotter
from lib.Strategy import SMACrossoverStrategy

# ============================== CLI GLUE ===============================
if __name__ == "__main__":
    symbol = "BTCUSDT"
    start = "2025-06-21"
    end = "2025-10-01"
    freq = "1m"
    resample = "30min"
    fast = 10
    slow = 30
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
    strategy = SMACrossoverStrategy(fast=fast, slow=slow, allow_short=False)

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
    base = f"{symbol}_{start}_to_{end}_{resample or 'native'}"
    csv_path = os.path.join(outdir, base + "_daily_metrics.csv")
    png_path = os.path.join(outdir, base + "_equity.png")

    PerformancePlotter().save_daily_csv(result["daily_df"], csv_path)
    PerformancePlotter().plot_equity(result["daily_df"], title="Equity Curve", save_path=png_path, show=True)

    print("Sharpe (ann.):", result["sharpe"])
    print("Total Return :", result["total_return"])
    print("Saved daily metrics CSV:", csv_path)
    print("Saved equity curve PNG :", png_path)

    results = {"result": result, "csv_path": csv_path, "png_path": png_path}
