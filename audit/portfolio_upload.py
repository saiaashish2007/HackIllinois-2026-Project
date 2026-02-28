"""
Portfolio upload adapter.

Accepts an uploaded equity curve (CSV) and runs a *partial* audit:
  - Regime detection (requires benchmark data)
  - Base performance metrics
  - Survivability score (param_stability and cost_resilience are unavailable)
  - LLM report

This path skips param sweeps and stress tests because we can't re-run
an external strategy with different parameters.
"""

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import StrategyConfig
from audit.regime_detection import classify_regimes, regime_metrics
from audit.survivability_score import compute_survivability_score
from audit.llm_report import generate_llm_report


def _compute_equity_metrics(pv: pd.Series) -> Dict[str, Any]:
    pv = pv.dropna().astype(float)
    if len(pv) < 2:
        return {"observations": len(pv)}

    rets = pv.pct_change().dropna()
    n = len(rets)
    total_return = float(pv.iloc[-1] / pv.iloc[0] - 1.0) if pv.iloc[0] > 0 else float("nan")
    cagr = float((pv.iloc[-1] / pv.iloc[0]) ** (252.0 / n) - 1.0) if pv.iloc[0] > 0 and n > 0 else float("nan")
    vol = float(rets.std(ddof=0) * np.sqrt(252)) if rets.std(ddof=0) > 0 else float("nan")
    sharpe = float(rets.mean() * np.sqrt(252) / rets.std(ddof=0)) if rets.std(ddof=0) > 0 else float("nan")
    roll_max = pv.cummax()
    dd = (pv - roll_max) / roll_max
    max_dd = float(dd.min()) if len(dd) else float("nan")

    underwater = dd < 0
    time_underwater_pct = float(underwater.sum() / len(dd)) if len(dd) > 0 else float("nan")

    recovery_days = []
    in_dd = False
    dd_start = 0
    for i in range(len(dd)):
        if dd.iloc[i] < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            recovery_days.append(i - dd_start)

    avg_recovery = float(np.mean(recovery_days)) if recovery_days else float("nan")
    worst_recovery = float(np.max(recovery_days)) if recovery_days else float("nan")

    if in_dd:
        ongoing_days = len(dd) - dd_start
        worst_recovery = float(max(worst_recovery if recovery_days else 0, ongoing_days))
        if not recovery_days:
            avg_recovery = float(ongoing_days)

    return {
        "observations": n,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_recovery_days": avg_recovery,
        "worst_recovery_days": worst_recovery,
        "time_underwater_pct": time_underwater_pct,
    }


def parse_uploaded_equity(
    csv_content: str | bytes,
    *,
    date_col: str = "date",
    value_col: str = "portfolio_value",
) -> pd.DataFrame:
    """
    Parse an uploaded CSV into a standardized equity DataFrame.

    Accepts flexible column names — tries common variations.
    """
    if isinstance(csv_content, bytes):
        csv_content = csv_content.decode("utf-8")

    df = pd.read_csv(StringIO(csv_content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    date_candidates = [date_col, "date", "timestamp", "time", "dt"]
    value_candidates = [value_col, "portfolio_value", "equity", "value", "nav", "balance", "pv", "total"]

    date_found = None
    for c in date_candidates:
        if c in df.columns:
            date_found = c
            break

    value_found = None
    for c in value_candidates:
        if c in df.columns:
            value_found = c
            break

    if date_found is None:
        raise ValueError(
            f"Could not find a date column. Expected one of: {date_candidates}. "
            f"Found: {list(df.columns)}"
        )
    if value_found is None:
        raise ValueError(
            f"Could not find a value column. Expected one of: {value_candidates}. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_found]),
        "portfolio_value": pd.to_numeric(df[value_found], errors="coerce"),
    })
    out = out.dropna().sort_values("date").reset_index(drop=True)
    return out


def run_upload_audit(
    equity_df: pd.DataFrame,
    *,
    benchmark_symbol: str = "QQQ",
    strategy_name: str = "Uploaded Portfolio",
    openai_api_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a partial audit on an uploaded equity curve.
    """
    config = StrategyConfig()
    config.ensure_directories()
    out_dir = output_dir or config.logs_dir / "audit_upload"
    out_dir.mkdir(parents=True, exist_ok=True)

    pv = equity_df.set_index("date")["portfolio_value"] if "date" in equity_df.columns else equity_df.iloc[:, 0]
    base_metrics = _compute_equity_metrics(pv)

    audit: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "strategy_name": strategy_name,
        "mode": "uploaded_portfolio",
        "base_metrics": base_metrics,
        "base_equity": equity_df.to_dict(orient="records"),
    }

    # Regime detection using benchmark
    try:
        bench_path = config.data_processed_dir / f"{benchmark_symbol}.csv"
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, parse_dates=["Date"]).set_index("Date")
        else:
            from pipeline.core import fetch_symbol_history, compute_indicators
            bench_raw = fetch_symbol_history(benchmark_symbol, config)
            bench_df = compute_indicators(bench_raw, config)

        regime_df = classify_regimes(bench_df["Close"])
        reg_metrics = regime_metrics(equity_df, regime_df)
        audit["regime_metrics"] = reg_metrics
        audit["regime_counts"] = {
            label: int((regime_df["regime"] == label).sum())
            for label in ["bull", "bear", "high_vol", "crisis"]
        }
    except Exception as exc:
        audit["regime_metrics"] = {}
        audit["regime_counts"] = {}

    # Param sweep & stress tests are NOT available for uploaded portfolios
    audit["param_sweep_summary"] = {
        "stability_score": None,
        "num_combos": 0,
        "note": "Parameter sweep not available for uploaded portfolios",
    }
    audit["stress_summary"] = {
        "cost_sensitivity_score": None,
        "num_scenarios": 0,
        "note": "Stress tests not available for uploaded portfolios",
    }

    # Survivability score (partial — use neutral defaults for unavailable components)
    survivability = compute_survivability_score(
        base_metrics=base_metrics,
        regime_metrics=audit.get("regime_metrics", {}),
        param_stability=0.5,
        cost_resilience=0.5,
    )
    audit["survivability"] = survivability

    # LLM report
    narrative = generate_llm_report(audit, api_key=openai_api_key)
    audit["report"] = narrative

    # Persist
    summary = {k: v for k, v in audit.items() if k != "base_equity"}
    with (out_dir / "upload_audit_report.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    with (out_dir / "upload_audit_report.md").open("w") as f:
        f.write(narrative)
    equity_df.to_csv(out_dir / "upload_equity.csv", index=False)

    return audit
