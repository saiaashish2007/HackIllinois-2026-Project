from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Allow module or script execution
try:
    from .config import StrategyConfig
except ImportError:  # direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pipeline.config import StrategyConfig


def fetch_symbol_history_intraday(symbol: str, config: StrategyConfig) -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars from Yahoo using yfinance.

    Notes:
    - Yahoo imposes lookback limits for fine intervals like 1m/2m/5m.
      For 5m, period="60d" is a typical maximum.
    - Index is returned as timestamps (often timezone-aware). We normalize and
      optionally filter to regular trading hours.
    """
    # Notebook kernels may have an older StrategyConfig class loaded without these
    # attributes. Use getattr defaults so the intraday pipeline still runs.
    interval = getattr(config, "intraday_interval", "5m")
    period = getattr(config, "intraday_period", "60d")
    tz = getattr(config, "intraday_timezone", "America/New_York")
    rth_only = bool(getattr(config, "intraday_rth_only", True))

    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        actions=False,
        prepost=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance commonly returns MultiIndex columns like (Price, Ticker).
        # We want the Price level ("Open/High/Low/Close/Adj Close/Volume").
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df = df.loc[:, ~df.columns.duplicated()]

    df.index = pd.to_datetime(df.index)
    df.index.name = "DateTime"

    # Ensure timezone is set / converted to America/New_York for market-hours filtering.
    try:
        if df.index.tz is None:
            # Yahoo sometimes returns naive but already in exchange time; assume NY time.
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)
    except Exception:
        # If tz ops fail, keep as-is (still usable, just won't filter as precisely).
        pass

    if rth_only:
        # Regular trading hours: 09:30 - 16:00 ET
        try:
            df = df.between_time("09:30", "16:00")
        except Exception:
            pass

    # Basic column sanity
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def compute_indicators_intraday(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Compute indicators on intraday bars, producing the same column names used by
    the rest of the pipeline: Return, FastSMA, SlowSMA, Volatility, Valid.

    Volatility is annualized using config.intraday_bars_per_year.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    data["Return"] = data["Close"].pct_change()
    data["FastSMA"] = data["Close"].rolling(config.fast_window).mean()
    data["SlowSMA"] = data["Close"].rolling(config.slow_window).mean()
    bars_per_day = int(getattr(config, "intraday_bars_per_day", 78))
    bars_per_year = int(getattr(config, "intraday_bars_per_year", 252 * bars_per_day))
    data["Volatility"] = data["Return"].rolling(config.vol_window).std() * np.sqrt(float(bars_per_year))
    data["Valid"] = (~data[["FastSMA", "SlowSMA", "Volatility"]].isna()).all(axis=1)

    # Drop warmup period
    first_valid_idx = data.index[data["Valid"]].min()
    if pd.isna(first_valid_idx):
        return data.iloc[0:0]
    return data.loc[first_valid_idx:]


if __name__ == "__main__":
    # Quick sanity-check helper:
    # `python pipeline/intraday.py --symbol AAPL --period 5d`
    import argparse

    ap = argparse.ArgumentParser(description="Intraday Yahoo (yfinance) fetch + indicator sanity check.")
    ap.add_argument("--symbol", type=str, default="AAPL")
    ap.add_argument("--period", type=str, default=None, help="Override StrategyConfig.intraday_period (e.g. 5d, 30d, 60d)")
    args = ap.parse_args()

    cfg = StrategyConfig()
    if args.period:
        cfg.intraday_period = args.period

    df_raw = fetch_symbol_history_intraday(args.symbol, cfg)
    print("raw shape:", df_raw.shape)
    if not df_raw.empty:
        print("raw head:", df_raw.head(3))
        print("raw tail:", df_raw.tail(3))
        df_ind = compute_indicators_intraday(df_raw, cfg)
        print("indicators shape:", df_ind.shape)
        if not df_ind.empty:
            print(df_ind[["Close", "FastSMA", "SlowSMA", "Volatility"]].tail(3))

