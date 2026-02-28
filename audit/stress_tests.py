"""
Stress tests: measure strategy fragility under adverse execution conditions.

Each test creates a modified StrategyConfig and re-runs the backtester,
then compares the stressed metrics to the baseline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import StrategyConfig
from pipeline.backtest import run_backtest


def _run_scenario(
    config: StrategyConfig,
    label: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        _eq, _trades, report = run_backtest(config, start=start, end=end)
        metrics = report.get("metrics", {})
    except Exception as exc:
        metrics = {"error": str(exc)}
    return {"scenario": label, **metrics}


def run_stress_tests(
    base_config: StrategyConfig,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    slippage_multiples: List[float] = [2.0, 5.0, 10.0],
    commission_multiples: List[float] = [2.0, 5.0, 10.0],
    vol_target_shocks: List[float] = [0.15, 0.25, 0.35],
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Run the backtester under multiple stress scenarios in parallel.

    1. **Slippage stress**: multiply base slippage_bps by 2×, 5×, 10×
    2. **Commission stress**: multiply base commission_bps similarly
    3. **Volatility-target stress**: change vol_target_annual

    Returns a DataFrame where each row is one scenario.
    """
    scenarios: List[Tuple[str, StrategyConfig]] = [
        ("baseline", base_config),
    ]
    for mult in slippage_multiples:
        cfg = deepcopy(base_config)
        cfg.slippage_bps = float(base_config.slippage_bps * mult)
        scenarios.append((f"slippage_{mult}x", cfg))
    for mult in commission_multiples:
        cfg = deepcopy(base_config)
        cfg.commission_bps = float(base_config.commission_bps * mult)
        scenarios.append((f"commission_{mult}x", cfg))
    for vt in vol_target_shocks:
        cfg = deepcopy(base_config)
        cfg.vol_target_annual = float(vt)
        cfg.vol_target_enabled = True
        scenarios.append((f"vol_target_{vt}", cfg))
    cfg = deepcopy(base_config)
    cfg.slippage_bps = float(base_config.slippage_bps * 10.0)
    cfg.commission_bps = float(base_config.commission_bps * 10.0)
    scenarios.append(("worst_case_costs", cfg))
    cfg = deepcopy(base_config)
    cfg.ml_enabled = False
    scenarios.append(("no_ml_filter", cfg))
    cfg = deepcopy(base_config)
    cfg.position_fraction = 0.10
    scenarios.append(("small_positions_10pct", cfg))

    def _run(label: str, config: StrategyConfig) -> Dict[str, Any]:
        return _run_scenario(config, label, start=start, end=end)

    workers = min(max_workers, len(scenarios))
    rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_label = {
            ex.submit(_run, label, cfg): label for label, cfg in scenarios
        }
        for future in as_completed(future_to_label):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({"scenario": future_to_label[future], "error": str(exc)})

    # Restore order (baseline first, etc.)
    label_order = [s[0] for s in scenarios]
    by_label = {r["scenario"]: r for r in rows}
    rows = [by_label.get(lbl, {"scenario": lbl, "error": "Failed"}) for lbl in label_order]

    return pd.DataFrame(rows)


def cost_sensitivity_score(stress_df: pd.DataFrame, metric: str = "sharpe") -> float:
    """
    Measure how much the strategy degrades under cost stress.

    Returns a score in [0, 1]:
        1.0 = completely insensitive to costs
        0.0 = strategy dies under increased costs
    """
    baseline_row = stress_df[stress_df["scenario"] == "baseline"]
    cost_rows = stress_df[stress_df["scenario"].str.contains("slippage|commission|worst_case", regex=True)]

    if baseline_row.empty or cost_rows.empty:
        return 0.0

    base_val = pd.to_numeric(baseline_row[metric], errors="coerce").iloc[0]
    if pd.isna(base_val) or base_val == 0:
        return 0.0

    cost_vals = pd.to_numeric(cost_rows[metric], errors="coerce").dropna()
    if cost_vals.empty:
        return 0.0

    avg_degradation = abs(float((cost_vals.mean() - base_val) / abs(base_val)))
    return round(max(0.0, 1.0 - avg_degradation), 4)
