"""
Regime detection: tag each trading day with a market regime label.

Regimes are derived from rolling return & volatility of a benchmark index.
The backtester's equity / trades are then split by regime so we can see
how the strategy performs in each environment.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REGIME_LABELS = ("bull", "bear", "high_vol", "crisis")


def classify_regimes(
    benchmark_prices: pd.Series,
    *,
    return_window: int = 60,
    vol_window: int = 20,
    vol_threshold_percentile: float = 80.0,
    crisis_drawdown_pct: float = -0.10,
) -> pd.DataFrame:
    """
    Classify each trading day into one of four regimes:

    - **bull**: rolling return > 0 AND volatility below threshold
    - **bear**: rolling return <= 0 AND volatility below threshold
    - **high_vol**: volatility above threshold (regardless of direction)
    - **crisis**: benchmark in a drawdown deeper than `crisis_drawdown_pct`

    Returns a DataFrame indexed by date with columns:
        regime, rolling_return, rolling_vol, drawdown
    """
    prices = benchmark_prices.dropna().sort_index().astype(float)
    if len(prices) < max(return_window, vol_window) + 1:
        raise ValueError("Not enough benchmark data for regime classification")

    daily_ret = prices.pct_change()

    rolling_ret = prices / prices.shift(return_window) - 1.0
    rolling_vol = daily_ret.rolling(vol_window).std() * np.sqrt(252)
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax

    vol_thresh = float(np.nanpercentile(rolling_vol.dropna(), vol_threshold_percentile))

    regime = pd.Series("bull", index=prices.index)
    regime[rolling_ret <= 0] = "bear"
    regime[rolling_vol >= vol_thresh] = "high_vol"
    regime[drawdown <= crisis_drawdown_pct] = "crisis"

    out = pd.DataFrame(
        {
            "regime": regime,
            "rolling_return": rolling_ret,
            "rolling_vol": rolling_vol,
            "drawdown": drawdown,
        },
        index=prices.index,
    )
    return out.dropna(subset=["rolling_return", "rolling_vol"])


def split_equity_by_regime(
    equity_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Split the backtest equity curve into per-regime slices.
    """
    eq = equity_df.copy()
    if "date" in eq.columns:
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")

    regime_df = regime_df.copy()
    regime_df.index = pd.to_datetime(regime_df.index)

    merged = eq.join(regime_df[["regime"]], how="inner")
    return {label: merged[merged["regime"] == label] for label in REGIME_LABELS}


def regime_metrics(
    equity_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-regime performance metrics (Sharpe, return, vol, max DD).
    """
    splits = split_equity_by_regime(equity_df, regime_df)
    results: Dict[str, Dict[str, Any]] = {}

    for label, chunk in splits.items():
        pv = chunk["portfolio_value"] if "portfolio_value" in chunk.columns else chunk.iloc[:, 0]
        pv = pv.dropna().astype(float)
        n = len(pv)
        if n < 2:
            results[label] = {"days": n, "sharpe": None, "total_return": None, "annual_vol": None, "max_drawdown": None}
            continue

        rets = pv.pct_change().dropna()
        vol = float(rets.std(ddof=0) * np.sqrt(252)) if rets.std(ddof=0) > 0 else 0.0
        sharpe = float(rets.mean() * np.sqrt(252) / rets.std(ddof=0)) if rets.std(ddof=0) > 0 else 0.0
        total_ret = float(pv.iloc[-1] / pv.iloc[0] - 1.0) if pv.iloc[0] > 0 else 0.0
        roll_max = pv.cummax()
        dd = (pv - roll_max) / roll_max
        max_dd = float(dd.min()) if len(dd) else 0.0

        results[label] = {
            "days": n,
            "sharpe": round(sharpe, 4),
            "total_return": round(total_ret, 4),
            "annual_vol": round(vol, 4),
            "max_drawdown": round(max_dd, 4),
        }

    return results
