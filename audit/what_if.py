"""
What-If scenario engine.

Takes a base equity curve and applies user-controlled macro shocks:
  - Volatility multiplier
  - Interest rate drag (opportunity cost)
  - Liquidity haircut (wider spreads → lower returns)
  - Correlation shock (correlated drawdowns amplified)

All transformations are applied to daily returns, then compounded
back into an equity curve — so the chart morphs live as sliders move.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_what_if(
    equity_df: pd.DataFrame,
    *,
    vol_multiplier: float = 1.0,
    interest_rate_annual: float = 0.0,
    liquidity_haircut_bps: float = 0.0,
    correlation_amplifier: float = 1.0,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Apply macro shocks to a base equity curve and return the stressed version.

    Parameters
    ----------
    equity_df : DataFrame with 'date' and 'portfolio_value' columns
    vol_multiplier : 1.0 = no change, 2.0 = double the volatility
    interest_rate_annual : annual risk-free rate drag (e.g. 0.05 = 5%)
    liquidity_haircut_bps : extra cost per day in bps (wider spreads)
    correlation_amplifier : 1.0 = no change, >1 = amplify moves that
        coincide with benchmark direction (simulates correlated selloffs)
    benchmark_returns : daily returns of a benchmark (needed for
        correlation amplification)
    """
    eq = equity_df.copy()
    if "date" in eq.columns:
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")

    pv = eq["portfolio_value"].astype(float)
    base_returns = pv.pct_change().fillna(0.0)
    n = len(base_returns)

    # 1) Volatility scaling: stretch returns around their mean
    mean_ret = base_returns.mean()
    stressed = mean_ret + (base_returns - mean_ret) * float(vol_multiplier)

    # 2) Interest rate drag: daily opportunity cost subtracted from returns
    daily_rate_drag = float(interest_rate_annual) / 252.0
    stressed = stressed - daily_rate_drag

    # 3) Liquidity haircut: extra daily cost
    daily_liq_cost = float(liquidity_haircut_bps) / 10_000.0
    stressed = stressed - daily_liq_cost

    # 4) Correlation amplification: on days the benchmark drops,
    #    amplify the portfolio's move in the same direction
    if correlation_amplifier != 1.0 and benchmark_returns is not None:
        bench = benchmark_returns.reindex(stressed.index).fillna(0.0)
        # Days where both portfolio and benchmark move in same direction
        same_dir = (stressed * bench) > 0
        # Amplify those co-moves
        amp_factor = np.where(same_dir, float(correlation_amplifier), 1.0)
        stressed = mean_ret + (stressed - mean_ret) * amp_factor

    # Rebuild equity curve from stressed returns
    stressed_pv = pv.iloc[0] * (1 + stressed).cumprod()
    stressed_pv.iloc[0] = pv.iloc[0]

    result = pd.DataFrame({
        "portfolio_value": stressed_pv,
        "base_value": pv,
    }, index=pv.index)
    result.index.name = "date"
    return result


def load_benchmark_returns(
    benchmark_symbol: str = "QQQ",
    project_root: Optional[str] = None,
) -> Optional[pd.Series]:
    """Load benchmark daily returns from processed data."""
    from pathlib import Path
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    bench_path = root / "data" / "processed" / f"{benchmark_symbol}.csv"
    if not bench_path.exists():
        return None
    try:
        df = pd.read_csv(bench_path, parse_dates=["Date"]).set_index("Date")
        if "Close" in df.columns:
            return df["Close"].pct_change().fillna(0.0)
    except Exception:
        pass
    return None
