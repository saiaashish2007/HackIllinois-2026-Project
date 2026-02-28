import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running as module (preferred: `python -m pipeline.backtest`) or directly as a script.
try:
    from .config import StrategyConfig
    from .core import (
        State,
        compute_indicators,
        compute_relative_strength,
        execute_pending_orders,
        fetch_fundamentals,
        fetch_symbol_history,
        generate_signal,
        mark_to_market,
        plan_order,
    )
    from .ml import build_latest_feature_row, load_model_and_meta
except ImportError:  # direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pipeline.config import StrategyConfig
    from pipeline.core import (
        State,
        compute_indicators,
        compute_relative_strength,
        execute_pending_orders,
        fetch_fundamentals,
        fetch_symbol_history,
        generate_signal,
        mark_to_market,
        plan_order,
    )
    from pipeline.ml import build_latest_feature_row, load_model_and_meta


def _load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _load_or_fetch(symbol: str, config: StrategyConfig) -> pd.DataFrame:
    raw_path = config.data_raw_dir / f"{symbol}.csv"
    if raw_path.exists():
        return _load_raw_csv(raw_path)
    return fetch_symbol_history(symbol, config)


def _equity_metrics(equity: pd.Series) -> Dict[str, Any]:
    equity = equity.dropna().astype(float)
    if len(equity) < 2:
        return {"observations": int(len(equity))}

    rets = equity.pct_change().dropna()
    n_days = int(len(rets))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] > 0 else float("nan")
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (252.0 / n_days) - 1.0) if (equity.iloc[0] > 0 and n_days > 0) else float("nan")
    vol = float(rets.std(ddof=0) * np.sqrt(252)) if rets.std(ddof=0) > 0 else float("nan")
    sharpe = float((rets.mean() * np.sqrt(252)) / rets.std(ddof=0)) if rets.std(ddof=0) > 0 else float("nan")
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = float(dd.min()) if len(dd) else float("nan")

    # Recovery analysis: find every drawdown episode and how long each took to recover
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
        "observations": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_recovery_days": avg_recovery,
        "worst_recovery_days": worst_recovery,
        "time_underwater_pct": time_underwater_pct,
    }


def run_backtest(
    config: StrategyConfig,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_value: float = 1.0,
    use_ml: bool = False,
    use_fundamentals: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Event-driven backtest that replays daily bars:
    - decision at close(t)
    - fill at open(t+1) (or next available bar if market is closed)

    IMPORTANT: we execute pending orders only when the simulated 'as_of' date
    reaches their planned fill date. This avoids the look-ahead bug that happens
    if you preload the full history and just check 'planned_date in df.index'.
    """
    config.ensure_directories()

    # Load benchmark + assets
    benchmark_raw = _load_or_fetch(config.benchmark_symbol, config)
    benchmark = compute_indicators(benchmark_raw, config)

    processed: Dict[str, pd.DataFrame] = {}
    for sym in config.symbols:
        raw = _load_or_fetch(sym, config)
        processed[sym] = compute_indicators(raw, config)

    # Build simulation calendar (union of all available dates)
    all_dates: List[pd.Timestamp] = []
    for df in processed.values():
        if df is not None and not df.empty:
            all_dates.extend(list(df.index))
    if benchmark is not None and not benchmark.empty:
        all_dates.extend(list(benchmark.index))

    dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(all_dates))))

    # Keep a warmup buffer before the user's start date so indicators
    # (moving averages etc.) are valid on day 1 of the simulation window.
    warmup_bars = max(config.slow_window, config.fast_window, config.vol_window, 20) + config.warmup_buffer
    if start:
        warmup_start = dates[dates >= pd.to_datetime(start)]
        if len(warmup_start) == 0 and len(dates) > 0:
            warmup_start = dates[-min(10, len(dates)):]
        earliest_needed = warmup_start[0] if len(warmup_start) > 0 else pd.to_datetime(start)
        # Walk back warmup_bars trading days from the user's start
        start_idx = dates.get_indexer([earliest_needed], method="ffill")[0]
        safe_idx = max(start_idx - warmup_bars, 0)
        dates = dates[safe_idx:]
    if end:
        dates = dates[dates <= pd.to_datetime(end)]
    if len(dates) < 10:
        raise RuntimeError(
            f"Only {len(dates)} trading days in the requested window (need at least 10). "
            "Try an earlier start date or a wider date range."
        )

    # Optional filters
    fundamentals: Dict[str, Dict[str, Any]] = {}
    if use_fundamentals:
        fundamentals = {sym: fetch_fundamentals(sym, config) for sym in config.symbols}

    model = None
    meta = None
    if use_ml and config.ml_enabled:
        model, meta = load_model_and_meta(config)

    state: State = {
        "cash": float(initial_value),
        "portfolio_value": float(initial_value),
        "positions": {},
        "pending_orders": [],
        "last_valuation_date": None,
    }
    trade_log: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    for dt in dates:
        # 1) Execute fills scheduled for today (as-of dt)
        #
        # We prefer the core helper's built-in as_of_dt gate, but in notebooks it's
        # easy to end up with an older `pipeline.core` module loaded in-memory that
        # doesn't yet accept `as_of_dt`. The fallback below preserves correctness
        # (no look-ahead) even in that situation.
        try:
            execute_pending_orders(state, processed, trade_log, config, as_of_dt=dt)
        except TypeError:
            # Fallback: only pass "due" orders to the executor, keep the rest.
            due: List[Dict[str, Any]] = []
            future: List[Dict[str, Any]] = []
            for order in state.get("pending_orders", []):
                planned = order.get("planned_fill_date")
                if not planned:
                    due.append(order)
                    continue
                planned_dt = pd.to_datetime(planned)
                if planned_dt <= dt:
                    due.append(order)
                else:
                    future.append(order)

            state["pending_orders"] = due
            execute_pending_orders(state, processed, trade_log, config)
            # Merge back future orders after executing due orders.
            state["pending_orders"] = list(state.get("pending_orders", [])) + future

        # 2) Mark-to-market using CLOSE(dt) when available
        latest_prices: Dict[str, float] = {}
        for sym, df in processed.items():
            if df is None or df.empty or dt not in df.index:
                continue
            latest_prices[sym] = float(df.loc[dt, "Close"])
        mark_to_market(state, latest_prices)

        # 3) Generate signals at close(dt) and queue orders for next bar
        for sym, df in processed.items():
            if df is None or df.empty or dt not in df.index:
                continue

            data_slice = df.loc[:dt]
            bench_slice = benchmark.loc[:dt] if benchmark is not None and not benchmark.empty else None

            signal = generate_signal(data_slice, sym, config)

            # Mirror the BUY-side filters from run_daily (optional)
            if signal["action"] == "BUY":
                rs_value = compute_relative_strength(data_slice, bench_slice, config.rel_strength_lookback) if bench_slice is not None else None
                vol_current = float(data_slice.iloc[-1]["Volatility"]) if not data_slice.empty else None
                fund = fundamentals.get(sym, {"passed": True})

                if rs_value is not None and rs_value < config.rel_strength_min:
                    signal["action"] = "HOLD"
                    signal["reason"] = "rs_filter"
                elif (
                    vol_current is not None
                    and config.max_volatility
                    and vol_current > config.max_volatility
                ):
                    signal["action"] = "HOLD"
                    signal["reason"] = "vol_filter"
                elif use_fundamentals and not fund.get("passed", True):
                    signal["action"] = "HOLD"
                    signal["reason"] = "fundamentals_filter"

                if (
                    use_ml
                    and model is not None
                    and meta is not None
                    and bench_slice is not None
                ):
                    feat_row = build_latest_feature_row(sym, data_slice, bench_slice, meta)
                    if feat_row is not None:
                        try:
                            ml_prob = float(model.predict_proba(feat_row)[0][1])
                            if ml_prob < config.ml_prob_min:
                                signal["action"] = "HOLD"
                                signal["reason"] = "ml_filter"
                        except Exception:
                            pass

            order = plan_order(signal, state, latest_prices, config)
            if order:
                state["pending_orders"].append(order)

        # 4) Record equity point
        equity_rows.append(
            {
                "date": dt.date().isoformat(),
                "portfolio_value": float(state["portfolio_value"]),
                "cash": float(state["cash"]),
                "open_positions": int(len(state["positions"])),
                "pending_orders": int(len(state["pending_orders"])),
                "exposure_close": (
                    float(
                        sum(
                            abs(float(pos.get("units", 0.0)) * float(latest_prices[sym]))
                            for sym, pos in state["positions"].items()
                            if sym in latest_prices and latest_prices[sym] is not None and not np.isnan(latest_prices[sym])
                        )
                        / float(state["portfolio_value"])
                    )
                    if float(state["portfolio_value"]) > 0
                    else 0.0
                ),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_log)

    # Trim equity to the user's requested window (drop warmup bars)
    if start and not equity_df.empty:
        equity_df["_dt"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df[equity_df["_dt"] >= pd.to_datetime(start)].drop(columns=["_dt"]).reset_index(drop=True)

    # Basic report
    eq = pd.to_numeric(equity_df["portfolio_value"], errors="coerce")
    report: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "config": {
            "symbols": list(config.symbols),
            "benchmark_symbol": config.benchmark_symbol,
            "fast_window": config.fast_window,
            "slow_window": config.slow_window,
            "vol_window": config.vol_window,
            "position_fraction": config.position_fraction,
            "max_portfolio_exposure": config.max_portfolio_exposure,
            "allow_leverage": config.allow_leverage,
            "cash_buffer": config.cash_buffer,
            "commission_bps": config.commission_bps,
            "min_commission": config.min_commission,
            "slippage_bps": config.slippage_bps,
            "vol_target_enabled": config.vol_target_enabled,
            "vol_target_annual": config.vol_target_annual,
            "vol_target_floor": config.vol_target_floor,
            "use_ml": bool(use_ml),
            "use_fundamentals": bool(use_fundamentals),
            "start": start,
            "end": end,
            "initial_value": float(initial_value),
        },
        "metrics": _equity_metrics(eq),
        "trades": {
            "count": int(len(trades_df)),
            "win_rate": float((trades_df.get("realized_pnl", pd.Series(dtype=float)) > 0).mean())
            if ("realized_pnl" in trades_df.columns and len(trades_df))
            else None,
        },
    }
    return equity_df, trades_df, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-driven backtest for the daily quant pipeline.")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    ap.add_argument("--initial", type=float, default=1.0, help="Initial portfolio value (normalized units)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: logs/)")
    ap.add_argument("--ml", action="store_true", help="Enable ML filter (uses saved model; not walk-forward retrained).")
    ap.add_argument("--fundamentals", action="store_true", help="Enable fundamentals filter (may require network).")
    args = ap.parse_args()

    cfg = StrategyConfig()
    outdir = Path(args.outdir) if args.outdir else cfg.logs_dir
    outdir.mkdir(parents=True, exist_ok=True)

    equity_df, trades_df, report = run_backtest(
        cfg,
        start=args.start,
        end=args.end,
        initial_value=args.initial,
        use_ml=bool(args.ml),
        use_fundamentals=bool(args.fundamentals),
    )

    equity_path = outdir / "backtest_equity.csv"
    trades_path = outdir / "backtest_trades.csv"
    report_path = outdir / "backtest_report.json"

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

