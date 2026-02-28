"""
Parameter sensitivity analysis.

Grid-search around the base strategy parameters and re-run the backtester
for each combination. Produces a DataFrame of (param_combo → metrics) that
can be visualized as a heatmap.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import StrategyConfig
from pipeline.backtest import run_backtest


DEFAULT_GRID: Dict[str, List[Any]] = {
    "fast_window": [10, 15, 20, 30, 40],
    "slow_window": [60, 80, 100, 120, 150],
}


def _make_config(base: StrategyConfig, overrides: Dict[str, Any]) -> StrategyConfig:
    """Return a new StrategyConfig with overrides applied."""
    cfg = deepcopy(base)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def run_single_combo(
    base_config: StrategyConfig,
    overrides: Dict[str, Any],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one backtest with the given param overrides and return a flat dict
    of params + key metrics.
    """
    cfg = _make_config(base_config, overrides)
    try:
        _eq, _trades, report = run_backtest(cfg, start=start, end=end)
        metrics = report.get("metrics", {})
    except Exception as exc:
        metrics = {"error": str(exc)}

    return {**overrides, **metrics}


def run_param_sweep(
    base_config: StrategyConfig,
    *,
    grid: Optional[Dict[str, List[Any]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full grid search (sequential). Returns a DataFrame where each
    row is one param combination and the columns are the param values +
    backtest metrics.
    """
    grid = grid or DEFAULT_GRID
    param_names = sorted(grid.keys())
    combos = list(itertools.product(*(grid[p] for p in param_names)))

    rows: List[Dict[str, Any]] = []
    for combo in combos:
        overrides = dict(zip(param_names, combo))

        if overrides.get("fast_window", 0) >= overrides.get("slow_window", float("inf")):
            continue

        result = run_single_combo(base_config, overrides, start=start, end=end)
        rows.append(result)

    return pd.DataFrame(rows)


def param_stability_score(sweep_df: pd.DataFrame, metric: str = "sharpe") -> float:
    """
    Measure how stable a metric is across parameter combos.

    Returns a score in [0, 1]:
        1.0 = perfectly stable (identical across all combos)
        0.0 = extremely unstable (huge variance)

    Uses coefficient of variation: stability = 1 / (1 + CV)
    """
    vals = pd.to_numeric(sweep_df.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
    if len(vals) < 2 or vals.mean() == 0:
        return 0.0
    cv = float(abs(vals.std(ddof=1) / vals.mean()))
    return round(1.0 / (1.0 + cv), 4)
