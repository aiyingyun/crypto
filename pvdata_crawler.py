from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import pandas as pd


# ------------------------------ HTTP session with retries ------------------------------
def _make_session(timeout: int = 30) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.request = _with_timeout(sess.request, timeout)  # type: ignore
    return sess


def _with_timeout(fn, timeout):
    def wrapper(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return fn(method, url, **kwargs)

    return wrapper


SESSION = _make_session()


# ------------------------------ URL builders ------------------------------
def _base_url(symbol: str, kind: str) -> str:
    """
    kind: 'metrics' | 'futures_1m' | 'spot_1m'
    """
    if kind == "metrics":
        return f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/"
    if kind == "futures_1m":
        return f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/1m/"
    if kind == "spot_1m":
        return f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/"
    raise ValueError(f"Unknown kind {kind!r}")


def _zip_filename(symbol: str, kind: str, date_str: str) -> str:
    if kind == "metrics":
        return f"{symbol}-metrics-{date_str}.zip"
    else:
        return f"{symbol}-1m-{date_str}.zip"


# ------------------------------ Helpers ------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _existing_dates_from_csvs(folder: Path, kind: str) -> set:
    """
    Derive date strings already present from csv filenames.
    metrics: {symbol}-metrics-YYYY-MM-DD.csv
    klines:  {symbol}-1m-YYYY-MM-DD.csv
    """
    dates: set = set()
    suffix = "-metrics-" if kind == "metrics" else "-1m-"
    for f in folder.glob("*.csv"):
        name = f.name
        if suffix in name:
            try:
                date_part = name.split(suffix, 1)[1].split(".csv", 1)[0]
                # quick validate
                dt.date.fromisoformat(date_part)
                dates.add(date_part)
            except Exception:
                continue
    return dates


def _daterange(start_date: dt.date, end_date: dt.date) -> List[str]:
    # inclusive start, inclusive end
    return pd.date_range(start=start_date, end=end_date, freq="D").strftime("%Y-%m-%d").tolist()


def _head_exists(url: str) -> bool:
    r = SESSION.head(url)
    # Some endpoints may not support HEAD consistently; fall back to GET headers-only style
    if r.status_code == 405:  # Method Not Allowed
        r = SESSION.get(url, stream=True)
        try:
            r.close()
        except Exception:
            pass
    return 200 <= r.status_code < 300


# ------------------------------ Core I/O ------------------------------
def _download_zip(url: str, out_path: Path) -> None:
    with SESSION.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def _extract_zip(zip_path: Path, dest_folder: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_folder)
    except zipfile.BadZipFile as e:
        # Clean up corrupt file to allow re-download next run
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Corrupt ZIP: {zip_path}") from e


def download_and_extract_binance_data(
        destination: Path,
        zip_url: str,
        zip_path: Path,
        zip_filename: str,
) -> None:
    print(f" ======= Downloading {zip_filename} =========")
    try:
        _download_zip(zip_url, zip_path)
        print(f"Downloaded {zip_filename}")

        _extract_zip(zip_path, destination)
        print(f"Extracted {zip_filename}")
    finally:
        # Always try to delete zip; if missing, ignore
        zip_path.unlink(missing_ok=True)


# ------------------------------ Public API ------------------------------
def download_dates_for(
        symbol: str,
        destination_path: Path,
        kind: str,
        dates: Iterable[str],
) -> None:
    """
    kind: 'metrics' | 'futures_1m' | 'spot_1m'
    """
    _ensure_dir(destination_path)
    base = _base_url(symbol, kind)

    for d in dates:
        zip_name = _zip_filename(symbol, kind, d)
        url = base + zip_name
        # Skip early if file doesn't exist upstream
        if not _head_exists(url):
            print(f"Skip {zip_name} (404 or unavailable).")
            continue

        zip_path = destination_path / zip_name
        try:
            download_and_extract_binance_data(destination_path, url, zip_path, zip_name)
        except requests.HTTPError as e:
            print(f"HTTP error for {zip_name}: {e}")
        except Exception as e:
            print(f"Error for {zip_name}: {e}")


def process_futures_symbol(
        symbol: str,
        root: Path,
        start: dt.date = dt.date(2020, 1, 1),
        end: Optional[dt.date] = None,
) -> None:
    if end is None:
        end = dt.date.today()
    """
    Downloads:
      - futures metrics -> {root}/binance/historical_binance/metrics/{symbol}
      - futures 1m klines -> {root}/binance/historical_binance/pv/{symbol}
    """
    all_days = set(_daterange(start, end))

    # --- metrics ---
    metrics_dir = root / f"binance/historical_binance/metrics/{symbol}"
    existing_metrics = _existing_dates_from_csvs(metrics_dir, "metrics")
    to_get_metrics = sorted(all_days - existing_metrics)
    if to_get_metrics:
        print(f"[{symbol}] metrics missing: {len(to_get_metrics)} days")
        download_dates_for(symbol, metrics_dir, "metrics", to_get_metrics)
    else:
        print(f"[{symbol}] metrics up to date")

    # --- futures 1m klines (pv) ---
    pv_dir = root / f"binance/historical_binance/pv/{symbol}"
    existing_pv = _existing_dates_from_csvs(pv_dir, "futures_1m")
    to_get_pv = sorted(all_days - existing_pv)
    if to_get_pv:
        print(f"[{symbol}] futures 1m klines missing: {len(to_get_pv)} days")
        download_dates_for(symbol, pv_dir, "futures_1m", to_get_pv)
    else:
        print(f"[{symbol}] futures 1m klines up to date")


def process_spot_symbol(
        symbol: str,
        root: Path,
        start: dt.date = dt.date(2020, 1, 1),
        end: Optional[dt.date] = None,
) -> None:
    """
    Downloads spot 1m klines into:
      {root}/binance/historical_binance_spot/{symbol}
    """
    if end is None:
        end = dt.date.today()
    all_days = set(_daterange(start, end))

    spot_dir = root / f"binance/historical_binance_spot/{symbol}"
    existing = _existing_dates_from_csvs(spot_dir, "spot_1m")
    to_get = sorted(all_days - existing)

    if to_get:
        print(f"[{symbol}] spot 1m klines missing: {len(to_get)} days")
        download_dates_for(symbol, spot_dir, "spot_1m", to_get)
    else:
        print(f"[{symbol}] spot 1m klines up to date")


# ------------------------------ CLI entry ------------------------------
if __name__ == "__main__":
    DATAPATH_CLOUD = Path("/Users/yingyunai/Desktop/crypto")
    print(DATAPATH_CLOUD)

    start = dt.date(2024, 1, 1)
    end = dt.date(2025, 1, 1)
    process_futures_symbol(symbol="BTCUSDT", root=DATAPATH_CLOUD, start=start, end=end)
    process_spot_symbol(symbol="BTCUSDT", root=DATAPATH_CLOUD, start=start, end=end)
