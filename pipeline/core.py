import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Allow running as module (preferred) or directly as a script
try:
    from .config import StrategyConfig
    from .ml import (
        build_latest_feature_row,
        load_model_and_meta,
    )
except ImportError:  # direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pipeline.config import StrategyConfig
    from pipeline.ml import (
        build_latest_feature_row,
        load_model_and_meta,
    )


State = Dict[str, Any]
Order = Dict[str, Any]

def _bps_to_frac(bps: float) -> float:
    return float(bps) / 10000.0


def _estimate_commission(notional: float, config: StrategyConfig) -> float:
    """
    Very simple per-side commission model in portfolio-value units.
    """
    comm = float(notional) * _bps_to_frac(config.commission_bps)
    return max(float(config.min_commission), comm)


def _apply_slippage(fill_price: float, side: str, config: StrategyConfig) -> float:
    """
    Apply symmetric slippage in bps:
    - BUY pays up (price increases)
    - SELL sells down (price decreases)
    """
    slip = _bps_to_frac(config.slippage_bps)
    if side.upper() == "BUY":
        return float(fill_price) * (1.0 + slip)
    if side.upper() == "SELL":
        return float(fill_price) * (1.0 - slip)
    return float(fill_price)


def _cap_notional_by_cash(
    desired_notional: float, state: State, config: StrategyConfig
) -> float:
    """
    If leverage is disallowed, cap order notional so that (notional + commission)
    fits inside available cash while respecting a cash buffer.
    """
    desired_notional = float(max(desired_notional, 0.0))
    if config.allow_leverage:
        return desired_notional

    spendable = float(state.get("cash", 0.0)) - float(state.get("portfolio_value", 0.0)) * float(config.cash_buffer)
    spendable = max(spendable, 0.0)
    if spendable <= 0:
        return 0.0

    comm_frac = _bps_to_frac(config.commission_bps)
    # Ensure we can at least pay min_commission if it's non-zero
    if spendable <= float(config.min_commission):
        return 0.0

    cap = (spendable - float(config.min_commission)) / (1.0 + comm_frac) if comm_frac > 0 else (spendable - float(config.min_commission))
    cap = max(cap, 0.0)
    return min(desired_notional, cap)



def _today_utc_date() -> datetime.date:
    return datetime.utcnow().date()


def load_state(path: Path) -> State:
    if not path.exists():
        return {
            "cash": 1.0,  # start with normalized portfolio value of 1.0
            "portfolio_value": 1.0,
            "positions": {},  # symbol -> {status, units, entry_price, entry_date}
            "pending_orders": [],
            "last_valuation_date": None,
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, df], ignore_index=True, sort=False)
        combined.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def fetch_symbol_history(symbol: str, config: StrategyConfig) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a single symbol and persist an append-only raw file.
    """
    end_date = _today_utc_date() + timedelta(days=1)  # include today's bar if available
    start_date = end_date - timedelta(days=config.lookback_days)

    def _download(group_by: str) -> pd.DataFrame:
        return yf.download(
            symbol,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=False,
            actions=False,
            interval="1d",
            group_by=group_by,
        )

    def _normalize(df_in: pd.DataFrame) -> pd.DataFrame:
        df_local = df_in.copy()
        if isinstance(df_local.columns, pd.MultiIndex):
            df_local.columns = df_local.columns.get_level_values(-1)
        df_local.index = pd.to_datetime(df_local.index)
        rename_map = {
            "Adj Close": "AdjClose",
            "Close": "Close",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Volume": "Volume",
        }
        df_local = df_local.rename(
            columns={k: v for k, v in rename_map.items() if k in df_local.columns}
        )
        df_local = df_local.loc[:, ~df_local.columns.duplicated()]

        # If Close is missing but symbol column exists (single column), treat it as Close
        if "Close" not in df_local.columns and df_local.shape[1] == 1:
            only_col = df_local.columns[0]
            df_local = df_local.rename(columns={only_col: "Close"})

        # If Close is missing but AdjClose exists, use AdjClose as Close
        if "Close" not in df_local.columns and "AdjClose" in df_local.columns:
            df_local["Close"] = df_local["AdjClose"]

        return df_local

    df = _normalize(_download("column"))

    required_cols = ["Close", "Open", "High", "Low", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # retry with ticker grouping (common yfinance quirk)
        df_retry = _normalize(_download("ticker"))
        missing_retry = [c for c in required_cols if c not in df_retry.columns]
        if not missing_retry:
            df = df_retry
        else:
            raise RuntimeError(
                f"Missing columns from fetched data for {symbol}: {missing_retry}. "
                f"Available columns: {list(df_retry.columns)}"
            )

    raw_path = config.data_raw_dir / f"{symbol}.csv"
    if raw_path.exists():
        existing = pd.read_csv(raw_path, parse_dates=["Date"])
        existing = existing.set_index("Date")
        existing.index = pd.to_datetime(existing.index)
        existing = existing.loc[:, ~existing.columns.duplicated()]
        union_cols = sorted(set(existing.columns) | set(df.columns))
        existing = existing.reindex(columns=union_cols)
        df = df.reindex(columns=union_cols)
        df = pd.concat([existing, df], axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Date"

    df.reset_index(names="Date").to_csv(raw_path, index=False)
    return df


def compute_relative_strength(
    asset_df: pd.DataFrame, benchmark_df: pd.DataFrame, lookback: int
) -> Optional[float]:
    if asset_df is None or benchmark_df is None:
        return None
    if len(asset_df) < lookback + 1 or len(benchmark_df) < lookback + 1:
        return None
    a = asset_df["Close"]
    b = benchmark_df["Close"]
    # align on dates
    common = a.index.intersection(b.index)
    if len(common) < lookback + 1:
        return None
    a = a.loc[common]
    b = b.loc[common]
    a_ret = float(a.iloc[-1] / a.iloc[-lookback] - 1.0)
    b_ret = float(b.iloc[-1] / b.iloc[-lookback] - 1.0)
    return a_ret - b_ret


def fetch_fundamentals(symbol: str, config: StrategyConfig) -> Dict[str, Any]:
    """
    Lightweight fundamentals screen using yfinance fast_info/info.
    """
    pe = None
    ps = None
    reasons: List[str] = []
    passed = True
    try:
        tkr = yf.Ticker(symbol)
        fi = getattr(tkr, "fast_info", {}) or {}
        pe = fi.get("trailing_pe")
        ps = fi.get("price_to_sales_trailing_12_months")

        if pe is None or ps is None:
            info = tkr.get_info() or {}
            pe = pe if pe is not None else info.get("trailingPE")
            ps = ps if ps is not None else info.get("priceToSalesTrailing12Months")
    except Exception as exc:  # network or schema issues
        reasons.append(f"fundamentals_fetch_error:{exc}")
        if not config.fundamentals_allow_missing:
            passed = False
        return {"pe": pe, "ps": ps, "passed": passed, "reasons": reasons}

    if pe is None and ps is None and not config.fundamentals_allow_missing:
        passed = False
        reasons.append("missing_pe_ps")
    if pe is not None and config.fundamentals_pe_max and pe > config.fundamentals_pe_max:
        passed = False
        reasons.append("pe_too_high")
    if ps is not None and config.fundamentals_ps_max and ps > config.fundamentals_ps_max:
        passed = False
        reasons.append("ps_too_high")

    return {"pe": pe, "ps": ps, "passed": passed, "reasons": reasons}


def compute_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = df.copy()
    data["Return"] = data["Close"].pct_change()
    data["FastSMA"] = data["Close"].rolling(config.fast_window).mean()
    data["SlowSMA"] = data["Close"].rolling(config.slow_window).mean()
    data["Volatility"] = data["Return"].rolling(config.vol_window).std() * np.sqrt(252)
    data["Valid"] = (~data[["FastSMA", "SlowSMA", "Volatility"]].isna()).all(axis=1)

    # Drop warmup period
    first_valid_idx = data.index[data["Valid"]].min()
    if pd.isna(first_valid_idx):
        return data.iloc[0:0]  # empty after warmup failure
    data = data.loc[first_valid_idx:]
    return data


def _debounced_condition(series: pd.Series, condition: str, days: int) -> bool:
    if len(series) < days:
        return False
    window = series.iloc[-days:]
    if condition == "gt":
        return bool((window > 0).all())
    if condition == "lt":
        return bool((window < 0).all())
    return False


def generate_signal(
    data: pd.DataFrame, symbol: str, config: StrategyConfig
) -> Dict[str, Any]:
    """
    Use latest fully formed bar to produce BUY / SELL / HOLD with confidence.
    """
    if data.empty or len(data) < config.warmup_period + 2:
        return {
            "symbol": symbol,
            "action": "HOLD",
            "reason": "insufficient_history",
            "confidence": 0.0,
            "decision_date": None,
            "latest_row": None,
        }

    last = data.iloc[-1]
    prev = data.iloc[-2]
    if not last["Valid"]:
        return {
            "symbol": symbol,
            "action": "HOLD",
            "reason": "indicators_invalid",
            "confidence": 0.0,
            "decision_date": last.name.date().isoformat(),
            "latest_row": last.to_dict(),
        }

    fast_vs_slow = data["FastSMA"] - data["SlowSMA"]
    buy_ready = prev["FastSMA"] <= prev["SlowSMA"] and last["FastSMA"] > last["SlowSMA"]
    sell_ready = prev["FastSMA"] >= prev["SlowSMA"] and last["FastSMA"] < last["SlowSMA"]

    # Debounce: require consecutive days on one side
    if config.debounce_days > 1 and len(fast_vs_slow) >= config.debounce_days:
        buy_ready = buy_ready and _debounced_condition(
            fast_vs_slow, "gt", config.debounce_days
        )
        sell_ready = sell_ready and _debounced_condition(
            -fast_vs_slow, "gt", config.debounce_days
        )

    spread = float(last["FastSMA"] - last["SlowSMA"])
    vol = float(last["Volatility"]) if not pd.isna(last["Volatility"]) else 0.0
    denom = max(abs(last["Close"]) * 0.01 + vol, 1e-6)
    confidence = min(1.0, max(0.0, abs(spread) / denom))

    action = "HOLD"
    reason = "no_crossover"
    if buy_ready:
        action = "BUY"
        reason = "fast_cross_above_slow"
    elif sell_ready:
        action = "SELL"
        reason = "fast_cross_below_slow"

    return {
        "symbol": symbol,
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "decision_date": last.name.date().isoformat(),
        "latest_row": last.to_dict(),
    }


def mark_to_market(state: State, latest_prices: Dict[str, float]) -> None:
    position_value = 0.0
    for sym, pos in state["positions"].items():
        price = latest_prices.get(sym)
        if price is None or np.isnan(price):
            continue
        position_value += pos["units"] * price
    state["portfolio_value"] = state["cash"] + position_value
    state["last_valuation_date"] = datetime.utcnow().date().isoformat()


def current_exposure(state: State, latest_prices: Dict[str, float]) -> float:
    if state["portfolio_value"] <= 0:
        return 0.0
    gross = 0.0
    for sym, pos in state["positions"].items():
        price = latest_prices.get(sym)
        if price is None or np.isnan(price):
            continue
        gross += abs(pos["units"] * price)
    return gross / state["portfolio_value"]


def plan_order(
    signal: Dict[str, Any],
    state: State,
    latest_prices: Dict[str, float],
    config: StrategyConfig,
) -> Optional[Order]:
    symbol = signal["symbol"]
    action = signal["action"]
    decision_date = signal["decision_date"]
    price_now = latest_prices.get(symbol)
    pos = state["positions"].get(symbol, {"status": "flat"})
    exposure = current_exposure(state, latest_prices)

    if action == "HOLD":
        return None

    if action == "BUY":
        if pos.get("status") == "long":
            return None  # already long
        available_capacity = max(config.max_portfolio_exposure - exposure, 0.0)
        allocation = float(min(config.position_fraction, available_capacity))
        if allocation <= 0 or price_now is None:
            return None

        # Risk-based sizing: volatility targeting (scales DOWN in high-vol names)
        if config.vol_target_enabled:
            vol = None
            if signal.get("latest_row") is not None:
                vol = signal["latest_row"].get("Volatility")
            if vol is not None and not pd.isna(vol) and float(vol) > 0:
                vol_eff = max(float(config.vol_target_floor), float(vol))
                scale = float(config.vol_target_annual) / vol_eff
                if not config.allow_leverage:
                    scale = min(1.0, scale)
                allocation = float(allocation) * float(max(scale, 0.0))

        target_notional = float(state["portfolio_value"] * allocation)
        target_notional = _cap_notional_by_cash(target_notional, state, config)
        if target_notional <= 0:
            return None
        return {
            "symbol": symbol,
            "action": "BUY",
            "decision_date": decision_date,
            "planned_fill_date": (datetime.fromisoformat(decision_date).date() + timedelta(days=1)).isoformat()
            if decision_date
            else None,
            "target_notional": target_notional,
            "confidence": signal.get("confidence", 0.0),
            "price_assumption": config.execution_price,
            "reason": signal.get("reason", "signal"),
        }

    if action == "SELL":
        if pos.get("status") != "long":
            return None  # nothing to exit
        units = pos.get("units", 0.0)
        if units <= 0 or price_now is None:
            return None
        return {
            "symbol": symbol,
            "action": "SELL",
            "decision_date": decision_date,
            "planned_fill_date": (datetime.fromisoformat(decision_date).date() + timedelta(days=1)).isoformat()
            if decision_date
            else None,
            "units": units,
            "confidence": signal.get("confidence", 0.0),
            "price_assumption": config.execution_price,
            "reason": signal.get("reason", "signal"),
        }
    return None


def execute_pending_orders(
    state: State,
    processed: Dict[str, pd.DataFrame],
    trade_log: List[Dict[str, Any]],
    config: StrategyConfig,
    as_of_dt: Optional[pd.Timestamp] = None,
) -> None:
    remaining: List[Order] = []
    for order in state["pending_orders"]:
        symbol = order["symbol"]
        planned_date = order.get("planned_fill_date")
        df = processed.get(symbol)
        if df is None or not planned_date:
            remaining.append(order)
            continue
        planned_dt = pd.to_datetime(planned_date)
        if as_of_dt is not None and planned_dt > as_of_dt:
            # Not yet time to execute this order in an event-driven simulation.
            remaining.append(order)
            continue
        if planned_dt not in df.index:
            # keep order until the planned date bar exists
            remaining.append(order)
            continue
        fill_row = df.loc[planned_dt]
        fill_price_raw = float(fill_row["Open"]) if config.execution_price == "next_open" else float(fill_row["Close"])

        if order["action"] == "BUY":
            target_notional = float(order["target_notional"])
            fill_price = _apply_slippage(fill_price_raw, "BUY", config)
            # Re-cap at execution time in case other fills changed cash
            target_notional = _cap_notional_by_cash(target_notional, state, config)
            if target_notional <= 0 or fill_price <= 0:
                continue
            commission = _estimate_commission(target_notional, config)
            total_cost = target_notional + commission
            if (not config.allow_leverage) and total_cost > float(state.get("cash", 0.0)) + 1e-12:
                # Should not happen due to cap, but keep as a guardrail
                continue
            units = target_notional / fill_price
            state["cash"] -= total_cost
            state["positions"][symbol] = {
                "status": "long",
                "units": units,
                "entry_price": fill_price,
                "entry_date": planned_date,
                "entry_notional": target_notional,
                "entry_commission": commission,
            }
            trade_log.append(
                {
                    "date": planned_date,
                    "symbol": symbol,
                    "action": "BUY",
                    "price": fill_price,
                    "price_raw": fill_price_raw,
                    "slippage_bps": config.slippage_bps,
                    "units": units,
                    "notional": target_notional,
                    "commission": commission,
                    "notional_plus_commission": total_cost,
                    "decision_date": order.get("decision_date"),
                    "reason": order.get("reason"),
                }
            )
        elif order["action"] == "SELL":
            pos = state["positions"].get(symbol)
            if pos is None or pos.get("units", 0) <= 0:
                continue
            units = pos["units"]
            fill_price = _apply_slippage(fill_price_raw, "SELL", config)
            notional = float(units) * float(fill_price)
            commission = _estimate_commission(notional, config)
            proceeds = notional - commission

            entry_notional = float(pos.get("entry_notional", float(units) * float(pos.get("entry_price", fill_price))))
            entry_commission = float(pos.get("entry_commission", 0.0))
            realized = proceeds - (entry_notional + entry_commission)

            state["cash"] += proceeds
            state["positions"].pop(symbol, None)
            trade_log.append(
                {
                    "date": planned_date,
                    "symbol": symbol,
                    "action": "SELL",
                    "price": fill_price,
                    "price_raw": fill_price_raw,
                    "slippage_bps": config.slippage_bps,
                    "units": units,
                    "notional": notional,
                    "commission": commission,
                    "proceeds": proceeds,
                    "realized_pnl": realized,
                    "decision_date": order.get("decision_date"),
                    "reason": order.get("reason"),
                }
            )
    state["pending_orders"] = remaining


def build_latest_price_map(processed: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sym, df in processed.items():
        if df.empty:
            continue
        prices[sym] = float(df.iloc[-1]["Close"])
    return prices


def run_daily(config: Optional[StrategyConfig] = None) -> None:
    """
    Main orchestration for the daily pipeline. Steps:
      1) Fetch data
      2) Compute indicators
      3) Execute any pending orders with available next-day data
      4) Generate new signals and queue orders for next open
      5) Log signals/trades/performance
      6) Persist state
    """
    config = config or StrategyConfig()
    config.ensure_directories()
    state = load_state(config.state_path)

    processed: Dict[str, pd.DataFrame] = {}
    signal_log_rows: List[Dict[str, Any]] = []
    trade_log_rows: List[Dict[str, Any]] = []

    # Phase 2 & 3: Data ingestion + indicators for benchmark
    benchmark_processed: Optional[pd.DataFrame] = None
    try:
        bench_raw = fetch_symbol_history(config.benchmark_symbol, config)
        benchmark_processed = compute_indicators(bench_raw, config)
        benchmark_processed.reset_index(names="Date").to_csv(
            config.data_processed_dir / f"{config.benchmark_symbol}.csv", index=False
        )
    except Exception as exc:
        benchmark_processed = None

    # Phase 2 & 3: Data ingestion + indicators for symbols
    for symbol in config.symbols:
        raw = fetch_symbol_history(symbol, config)
        enriched = compute_indicators(raw, config)
        processed[symbol] = enriched
        # persist enriched data for inspection
        enriched.reset_index(names="Date").to_csv(
            config.data_processed_dir / f"{symbol}.csv", index=False
        )

    # Current prices for MTM and risk
    latest_prices = build_latest_price_map(processed)

    # Fundamentals cache
    fundamentals: Dict[str, Dict[str, Any]] = {
        sym: fetch_fundamentals(sym, config) for sym in config.symbols
    }

    # Phase 6: execute any pending orders whose planned fill date is now available
    execute_pending_orders(state, processed, trade_log_rows, config)

    # Mark to market after possible fills
    mark_to_market(state, latest_prices)

    # Phase 4 & 5: generate fresh signals and queue orders
    model = None
    meta = None
    if config.ml_enabled:
        model, meta = load_model_and_meta(config)

    for symbol, df in processed.items():
        signal = generate_signal(df, symbol, config)
        rs_value = compute_relative_strength(
            df, benchmark_processed, config.rel_strength_lookback
        )
        vol_current = float(df.iloc[-1]["Volatility"]) if not df.empty else None
        fundamentals_info = fundamentals.get(symbol, {"passed": True, "reasons": []})

        ml_prob = None
        if (
            model is not None
            and meta is not None
            and benchmark_processed is not None
            and not df.empty
        ):
            feat_row = build_latest_feature_row(symbol, df, benchmark_processed, meta)
            if feat_row is not None:
                try:
                    ml_prob = float(model.predict_proba(feat_row)[0][1])
                except Exception:
                    ml_prob = None

        # Apply filters to BUYs
        if signal["action"] == "BUY":
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
            elif not fundamentals_info.get("passed", True):
                signal["action"] = "HOLD"
                signal["reason"] = "fundamentals_filter"
            elif (
                config.ml_enabled
                and model is not None
                and meta is not None
                and ml_prob is not None
                and ml_prob < config.ml_prob_min
            ):
                signal["action"] = "HOLD"
                signal["reason"] = "ml_filter"

        signal_log_rows.append(
            {
                "date": signal.get("decision_date"),
                "symbol": symbol,
                "action": signal.get("action"),
                "reason": signal.get("reason"),
                "confidence": signal.get("confidence"),
                "fast_sma": signal["latest_row"].get("FastSMA") if signal.get("latest_row") else None,
                "slow_sma": signal["latest_row"].get("SlowSMA") if signal.get("latest_row") else None,
                "close": signal["latest_row"].get("Close") if signal.get("latest_row") else None,
                "execution_price": config.execution_price,
                "rel_strength": rs_value,
                "volatility": vol_current,
                "fundamentals_passed": fundamentals_info.get("passed"),
                "fundamentals_pe": fundamentals_info.get("pe"),
                "fundamentals_ps": fundamentals_info.get("ps"),
                "ml_prob_up": ml_prob,
                "ml_model": meta.get("model_name") if meta else None,
            }
        )
        order = plan_order(signal, state, latest_prices, config)
        if order:
            state["pending_orders"].append(order)

    # Mark to market again after any new signals (positions unchanged but for completeness)
    mark_to_market(state, latest_prices)

    # Phase 7: Logging
    _append_rows(config.logs_dir / "daily_signals.csv", signal_log_rows)
    _append_rows(config.logs_dir / "trades.csv", trade_log_rows)
    _append_rows(
        config.logs_dir / "performance.csv",
        [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "date": datetime.utcnow().date().isoformat(),
                "portfolio_value": state["portfolio_value"],
                "cash": state["cash"],
                "open_positions": len(state["positions"]),
                "pending_orders": len(state["pending_orders"]),
            }
        ],
    )

    # Optional: News + impact (best-effort; never breaks the trading pipeline)
    if getattr(config, "news_enabled", False):
        try:
            from .news import run_news_step

            run_news_step(
                symbols=list(config.symbols),
                logs_dir=config.logs_dir,
                prices=processed,
                benchmark=benchmark_processed,
                max_items_per_symbol=int(getattr(config, "news_max_items_per_symbol", 8)),
                horizons=tuple(getattr(config, "news_horizons", [1, 3, 5])),
                timeout_s=int(getattr(config, "news_timeout_s", 10)),
                cache_ttl_minutes=int(getattr(config, "news_cache_ttl_minutes", 180)),
                sleep_s=float(getattr(config, "news_sleep_s", 0.25)),
            )
        except Exception:
            # Intentionally swallow errors: news is an auxiliary UX feature.
            pass

    # Phase 7: Persist state
    save_state(config.state_path, state)


if __name__ == "__main__":
    run_daily()
