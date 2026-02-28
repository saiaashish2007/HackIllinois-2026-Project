"""
Audit orchestrator: runs the full audit pipeline end-to-end.

    base backtest → regime detection → param sweep → stress tests
        → survivability score → LLM report → full audit object

This is the single function you call (or that Streamlit calls).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import StrategyConfig
from pipeline.backtest import run_backtest

from audit.regime_detection import classify_regimes, regime_metrics
from audit.param_sweep import run_param_sweep, param_stability_score, DEFAULT_GRID
from audit.stress_tests import run_stress_tests, cost_sensitivity_score
from audit.survivability_score import compute_survivability_score
from audit.llm_report import generate_llm_report


def run_full_audit(
    config: Optional[StrategyConfig] = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    openai_api_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Orchestrate the full strategy audit.

    Returns a dict with all results that Streamlit can consume.
    Also writes JSON + CSV artifacts to `output_dir` if provided.
    """
    config = config or StrategyConfig()
    config.ensure_directories()
    out_dir = output_dir or config.logs_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "strategy_name": "SMA Crossover (ML-filtered)",
        "initial_capital": float(initial_capital),
        "config_summary": {
            "symbols": list(config.symbols),
            "benchmark": config.benchmark_symbol,
            "fast_window": config.fast_window,
            "slow_window": config.slow_window,
            "position_fraction": config.position_fraction,
            "slippage_bps": config.slippage_bps,
            "commission_bps": config.commission_bps,
            "ml_enabled": config.ml_enabled,
        },
    }

    grid = param_grid or DEFAULT_GRID

    # ── LOCAL: sequential execution ─────────────────────────────
    # 1. Baseline backtest
    print("[audit] Running baseline backtest locally ...")
    equity_df, trades_df, report = run_backtest(
        config, start=start, end=end, initial_value=float(initial_capital),
    )
    base_metrics = report.get("metrics", {})
    audit["base_metrics"] = base_metrics
    audit["base_equity"] = equity_df.to_dict(orient="records")
    audit["base_trades_count"] = len(trades_df)

    # 2. Regime detection
    print("[audit] Classifying market regimes ...")
    try:
        bench_path = config.data_processed_dir / f"{config.benchmark_symbol}.csv"
        if bench_path.exists():
            bench_df = pd.read_csv(bench_path, parse_dates=["Date"]).set_index("Date")
        else:
            from pipeline.core import fetch_symbol_history, compute_indicators
            bench_raw = fetch_symbol_history(config.benchmark_symbol, config)
            bench_df = compute_indicators(bench_raw, config)

        regime_df = classify_regimes(bench_df["Close"])
        reg_met = regime_metrics(equity_df, regime_df)
        audit["regime_metrics"] = reg_met
        audit["regime_counts"] = {
            label: int((regime_df["regime"] == label).sum())
            for label in ["bull", "bear", "high_vol", "crisis"]
        }
    except Exception as exc:
        print(f"[audit] Regime detection failed: {exc}")
        audit["regime_metrics"] = {}
        audit["regime_counts"] = {}

    # 3. Parameter sweep
    print("[audit] Running parameter sweep locally ...")
    sweep_df = run_param_sweep(config, grid=grid, start=start, end=end)

    # 4. Stress tests
    print("[audit] Running stress tests locally ...")
    stress_df = run_stress_tests(config, start=start, end=end)

    audit["gpu_ml"] = {"gpu_used": False, "reason": "Local execution"}

    # ── Process sweep results ──────────────────────────────────────
    p_stability = param_stability_score(sweep_df)
    audit["param_sweep"] = sweep_df.to_dict(orient="records")
    audit["param_sweep_summary"] = {
        "stability_score": p_stability,
        "num_combos": len(sweep_df),
        "best_sharpe": float(pd.to_numeric(sweep_df.get("sharpe", pd.Series(dtype=float)), errors="coerce").max()),
        "worst_sharpe": float(pd.to_numeric(sweep_df.get("sharpe", pd.Series(dtype=float)), errors="coerce").min()),
    }

    # ── Process stress results ─────────────────────────────────────
    c_resilience = cost_sensitivity_score(stress_df)

    worst_case_row = stress_df[stress_df["scenario"] == "worst_case_costs"]
    worst_sharpe = None
    if not worst_case_row.empty:
        worst_sharpe = pd.to_numeric(worst_case_row.get("sharpe", pd.Series(dtype=float)), errors="coerce").iloc[0]

    audit["stress_tests"] = stress_df.to_dict(orient="records")
    audit["stress_summary"] = {
        "cost_sensitivity_score": c_resilience,
        "worst_case_sharpe": float(worst_sharpe) if worst_sharpe is not None else None,
        "num_scenarios": len(stress_df),
    }

    # ── 5. Survivability score ────────────────────────────────────
    print("[audit] Computing survivability score ...")
    survivability = compute_survivability_score(
        base_metrics=base_metrics,
        regime_metrics=audit.get("regime_metrics", {}),
        param_stability=p_stability,
        cost_resilience=c_resilience,
    )
    audit["survivability"] = survivability

    # ── 6. LLM report ────────────────────────────────────────────
    print("[audit] Generating report ...")
    narrative = generate_llm_report(audit, api_key=openai_api_key)
    audit["report"] = narrative

    # ── 7. Persist artifacts ──────────────────────────────────────
    _write_artifacts(audit, equity_df, sweep_df, stress_df, out_dir)

    print(f"[audit] Done. Score = {survivability['score']}/100 ({survivability['grade']})")
    print(f"[audit] Artifacts written to {out_dir}")

    return audit


def _write_artifacts(
    audit: Dict[str, Any],
    equity_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main audit JSON (everything except large DataFrames)
    summary = {k: v for k, v in audit.items() if k not in ("base_equity", "param_sweep", "stress_tests")}
    with (out_dir / "audit_report.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Narrative markdown
    with (out_dir / "audit_report.md").open("w") as f:
        f.write(audit.get("report", ""))

    # CSVs
    equity_df.to_csv(out_dir / "audit_equity.csv", index=False)
    sweep_df.to_csv(out_dir / "audit_param_sweep.csv", index=False)
    stress_df.to_csv(out_dir / "audit_stress_tests.csv", index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run full strategy audit")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--openai-key", type=str, default=None)
    args = ap.parse_args()

    result = run_full_audit(start=args.start, end=args.end, openai_api_key=args.openai_key)
    print(f"\nSurvivability: {result['survivability']['score']}/100 ({result['survivability']['grade']})")
