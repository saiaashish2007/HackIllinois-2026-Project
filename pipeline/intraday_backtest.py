import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running as module (preferred) or directly as a script
try:
    from .config import StrategyConfig
    from .core import (
        State,
        compute_relative_strength,
        current_exposure,
        execute_pending_orders,
        generate_signal,
        mark_to_market,
    )
    from .intraday import compute_indicators_intraday, fetch_symbol_history_intraday
except ImportError:  # direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pipeline.config import StrategyConfig
    from pipeline.core import (
        State,
        compute_relative_strength,
        current_exposure,
        execute_pending_orders,
        generate_signal,
        mark_to_market,
    )
    from pipeline.intraday import compute_indicators_intraday, fetch_symbol_history_intraday


def _align_cutoff(ts: Optional[str], tz) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    t = pd.to_datetime(ts)
    if tz is not None:
        if getattr(t, "tz", None) is None:
            return t.tz_localize(tz)
        return t.tz_convert(tz)
    # times are tz-naive
    if getattr(t, "tz", None) is not None:
        return t.tz_convert(None)
    return t


def _plan_intraday_order(
    signal: Dict[str, Any],
    state: State,
    latest_prices: Dict[str, float],
    config: StrategyConfig,
    *,
    decision_ts: pd.Timestamp,
    planned_fill_ts: pd.Timestamp,
) -> Optional[Dict[str, Any]]:
    """
    Intraday order planning:
    - decision at close(decision_ts)
    - planned fill at open(planned_fill_ts) (next bar)
    """
    symbol = signal["symbol"]
    action = signal["action"]
    price_now = latest_prices.get(symbol)
    pos = state["positions"].get(symbol, {"status": "flat"})
    exposure = current_exposure(state, latest_prices)

    if action == "HOLD":
        return None

    if action == "BUY":
        if pos.get("status") == "long":
            return None
        available_capacity = max(config.max_portfolio_exposure - exposure, 0.0)
        allocation = float(min(config.position_fraction, available_capacity))
        if allocation <= 0 or price_now is None:
            return None

        # Vol targeting (uses the latest computed annualized vol for intraday bars)
        if config.vol_target_enabled and signal.get("latest_row") is not None:
            vol = signal["latest_row"].get("Volatility")
            if vol is not None and not pd.isna(vol) and float(vol) > 0:
                vol_eff = max(float(config.vol_target_floor), float(vol))
                scale = float(config.vol_target_annual) / vol_eff
                if not config.allow_leverage:
                    scale = min(1.0, scale)
                allocation *= float(max(scale, 0.0))

        target_notional = float(state["portfolio_value"] * allocation)
        # Core caps at execution time too; we keep planning simple here.
        if target_notional <= 0:
            return None
        return {
            "symbol": symbol,
            "action": "BUY",
            "decision_date": decision_ts.isoformat(),
            "planned_fill_date": planned_fill_ts.isoformat(),
            "target_notional": target_notional,
            "confidence": signal.get("confidence", 0.0),
            "price_assumption": "next_open",
            "reason": signal.get("reason", "signal"),
        }

    if action == "SELL":
        if pos.get("status") != "long":
            return None
        units = float(pos.get("units", 0.0))
        if units <= 0 or price_now is None:
            return None
        return {
            "symbol": symbol,
            "action": "SELL",
            "decision_date": decision_ts.isoformat(),
            "planned_fill_date": planned_fill_ts.isoformat(),
            "units": units,
            "confidence": signal.get("confidence", 0.0),
            "price_assumption": "next_open",
            "reason": signal.get("reason", "signal"),
        }

    return None


def _equity_metrics_intraday(equity: pd.Series, bars_per_year: int) -> Dict[str, Any]:
    equity = equity.dropna().astype(float)
    if len(equity) < 2:
        return {"observations": int(len(equity))}

    rets = equity.pct_change().dropna()
    n_bars = int(len(rets))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] > 0 else float("nan")
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (float(bars_per_year) / n_bars) - 1.0) if (equity.iloc[0] > 0 and n_bars > 0) else float("nan")
    vol = float(rets.std(ddof=0) * np.sqrt(float(bars_per_year))) if rets.std(ddof=0) > 0 else float("nan")
    sharpe = float((rets.mean() * np.sqrt(float(bars_per_year))) / rets.std(ddof=0)) if rets.std(ddof=0) > 0 else float("nan")
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = float(dd.min()) if len(dd) else float("nan")
    return {
        "observations": n_bars,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "bars_per_year": int(bars_per_year),
    }


def run_intraday_backtest(
    config: StrategyConfig,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_value: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    5m intraday event-driven backtest:
    - decision at close(t)
    - fill at open(t+1)

    Uses Yahoo 5m bars (yfinance) with its lookback limitations.
    """
    config.ensure_directories()

    # Benchmark (for optional RS filter; uses close-to-close returns on intraday bars)
    bench_raw = fetch_symbol_history_intraday(config.benchmark_symbol, config)
    bench = compute_indicators_intraday(bench_raw, config)

    processed: Dict[str, pd.DataFrame] = {}
    for sym in config.symbols:
        raw = fetch_symbol_history_intraday(sym, config)
        processed[sym] = compute_indicators_intraday(raw, config)

    # Simulation timestamps = union across all symbols + benchmark
    all_times: List[pd.Timestamp] = []
    for df in processed.values():
        if df is not None and not df.empty:
            all_times.extend(list(df.index))
    if bench is not None and not bench.empty:
        all_times.extend(list(bench.index))

    times = pd.DatetimeIndex(sorted(set(pd.to_datetime(all_times))))
    tz = times.tz
    start_ts = _align_cutoff(start, tz)
    end_ts = _align_cutoff(end, tz)
    if start_ts is not None:
        times = times[times >= start_ts]
    if end_ts is not None:
        times = times[times <= end_ts]
    if len(times) == 0:
        raise RuntimeError("No intraday bars available for the requested window (Yahoo 5m is limited).")

    state: State = {
        "cash": float(initial_value),
        "portfolio_value": float(initial_value),
        "positions": {},
        "pending_orders": [],
        "last_valuation_date": None,
    }
    trade_log: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    for i, t in enumerate(times):
        # Need a "next bar" for fills. Last bar can't be traded (no next open).
        if i >= len(times) - 1:
            break
        t_next = times[i + 1]

        # Execute fills scheduled for now (no look-ahead)
        try:
            execute_pending_orders(state, processed, trade_log, config, as_of_dt=t)
        except TypeError:
            # Backward-compatible fallback if core is stale in a notebook kernel
            due: List[Dict[str, Any]] = []
            future: List[Dict[str, Any]] = []
            for order in state.get("pending_orders", []):
                planned = order.get("planned_fill_date")
                if not planned:
                    due.append(order)
                    continue
                planned_dt = pd.to_datetime(planned)
                if planned_dt <= t:
                    due.append(order)
                else:
                    future.append(order)
            state["pending_orders"] = due
            execute_pending_orders(state, processed, trade_log, config)
            state["pending_orders"] = list(state.get("pending_orders", [])) + future

        # Mark-to-market using close(t)
        latest_prices: Dict[str, float] = {}
        for sym, df in processed.items():
            if df is None or df.empty or t not in df.index:
                continue
            latest_prices[sym] = float(df.loc[t, "Close"])
        mark_to_market(state, latest_prices)

        # Generate signals at close(t), plan orders for next bar
        for sym, df in processed.items():
            if df is None or df.empty or t not in df.index:
                continue
            data_slice = df.loc[:t]
            bench_slice = bench.loc[:t] if bench is not None and not bench.empty else None
            signal = generate_signal(data_slice, sym, config)

            # Optional RS filter (if you keep rel_strength_min > 0)
            if signal["action"] == "BUY" and bench_slice is not None:
                rs_value = compute_relative_strength(data_slice, bench_slice, config.rel_strength_lookback)
                if rs_value is not None and rs_value < config.rel_strength_min:
                    signal["action"] = "HOLD"
                    signal["reason"] = "rs_filter"

            order = _plan_intraday_order(
                signal,
                state,
                latest_prices,
                config,
                decision_ts=t,
                planned_fill_ts=t_next,
            )
            if order:
                state["pending_orders"].append(order)

        equity_rows.append(
            {
                "timestamp": t.isoformat(),
                "portfolio_value": float(state["portfolio_value"]),
                "cash": float(state["cash"]),
                "open_positions": int(len(state["positions"])),
                "pending_orders": int(len(state["pending_orders"])),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_log)
    eq = pd.to_numeric(equity_df["portfolio_value"], errors="coerce")

    bars_per_day = int(getattr(config, "intraday_bars_per_day", 78))
    bars_per_year = int(getattr(config, "intraday_bars_per_year", 252 * bars_per_day))
    interval = getattr(config, "intraday_interval", "5m")
    period = getattr(config, "intraday_period", "60d")
    rth_only = bool(getattr(config, "intraday_rth_only", True))

    report: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "config": {
            "interval": interval,
            "period": period,
            "rth_only": rth_only,
            "bars_per_year": int(bars_per_year),
            "symbols": list(config.symbols),
            "benchmark_symbol": config.benchmark_symbol,
            "fast_window_bars": int(config.fast_window),
            "slow_window_bars": int(config.slow_window),
            "vol_window_bars": int(config.vol_window),
            "commission_bps": config.commission_bps,
            "slippage_bps": config.slippage_bps,
            "allow_leverage": bool(config.allow_leverage),
            "vol_target_enabled": bool(config.vol_target_enabled),
            "vol_target_annual": float(config.vol_target_annual),
            "start": start,
            "end": end,
            "initial_value": float(initial_value),
        },
        "metrics": _equity_metrics_intraday(eq, bars_per_year),
        "trades": {
            "count": int(len(trades_df)),
            "win_rate": float((trades_df.get("realized_pnl", pd.Series(dtype=float)) > 0).mean())
            if ("realized_pnl" in trades_df.columns and len(trades_df))
            else None,
        },
    }
    return equity_df, trades_df, report


def main() -> None:
    ap = argparse.ArgumentParser(description="5m intraday backtest using Yahoo (yfinance).")
    ap.add_argument("--start", type=str, default=None, help="Start timestamp/date (e.g. 2026-01-06 or 2026-01-06 10:00)")
    ap.add_argument("--end", type=str, default=None, help="End timestamp/date")
    ap.add_argument("--initial", type=float, default=1.0, help="Initial portfolio value (normalized units)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: logs/)")
    args = ap.parse_args()

    cfg = StrategyConfig()
    outdir = Path(args.outdir) if args.outdir else cfg.logs_dir
    outdir.mkdir(parents=True, exist_ok=True)

    equity_df, trades_df, report = run_intraday_backtest(
        cfg, start=args.start, end=args.end, initial_value=args.initial
    )

    equity_path = outdir / "intraday_backtest_equity.csv"
    trades_path = outdir / "intraday_backtest_trades.csv"
    report_path = outdir / "intraday_backtest_report.json"

    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("Wrote:", equity_path)
    print("Wrote:", trades_path)
    print("Wrote:", report_path)
    print("Metrics:", report.get("metrics"))


if __name__ == "__main__":
    main()

