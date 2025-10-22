import os

import pandas as pd
from matplotlib import pyplot as plt


class PerformancePlotter:
    @staticmethod
    def save_daily_csv(daily_df: pd.DataFrame, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        daily_df.to_csv(out_path, index=False)
        return out_path

    @staticmethod
    def plot_equity(daily_df: pd.DataFrame, title="Equity Curve",
                    save_path: str | None = None, show: bool = True):
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
