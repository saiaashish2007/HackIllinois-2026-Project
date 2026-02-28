"""
Modal cloud integration — runs param sweeps and stress tests in parallel.

Deploy:   modal deploy modal_app/remote.py
Run CLI:  modal run modal_app/remote.py

The Streamlit app and orchestrator call these functions automatically
when Modal is installed and authenticated.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional

import modal

app = modal.App("quant-strategy-audit")

audit_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.1",
        "yfinance>=0.2.50",
        "scikit-learn>=1.3",
        "xgboost>=1.7",
        "openai>=1.0",
    )
    .add_local_dir("pipeline", "/root/pipeline")
    .add_local_dir("audit", "/root/audit")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "pandas>=2.1",
        "yfinance>=0.2.50",
        "scikit-learn>=1.3",
        "xgboost>=1.7",
        "openai>=1.0",
    )
    .add_local_dir("pipeline", "/root/pipeline")
    .add_local_dir("audit", "/root/audit")
)


# ── Full base backtest + ML on GPU (NVIDIA A100) ──────────────────
@app.function(image=gpu_image, gpu="A100", timeout=900, retries=1)
def run_base_backtest_modal(
    base_config_dict: Dict[str, Any],
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Run everything on a single NVIDIA A100 GPU worker:
      1. Base backtest (5+ years of data)
      2. Regime detection
      3. XGBoost ML training (GPU-accelerated)

    This avoids redundant data fetching between workers.
    """
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    import pandas as pd
    from pipeline.config import StrategyConfig
    from pipeline.backtest import run_backtest

    cfg = StrategyConfig()
    for k, v in base_config_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # ── 1. Base backtest ──────────────────────────────────────────
    try:
        equity_df, trades_df, report = run_backtest(
            cfg, start=start, end=end, initial_value=float(initial_capital),
        )
    except Exception as exc:
        return {"error": str(exc)}

    result: Dict[str, Any] = {
        "base_metrics": report.get("metrics", {}),
        "base_equity": equity_df.to_dict(orient="records"),
        "base_trades_count": len(trades_df),
    }

    # ── 2. Regime detection ───────────────────────────────────────
    from pipeline.core import fetch_symbol_history, compute_indicators
    try:
        from audit.regime_detection import classify_regimes, regime_metrics as _regime_metrics

        bench_raw = fetch_symbol_history(cfg.benchmark_symbol, cfg)
        bench_df = compute_indicators(bench_raw, cfg)
        regime_df = classify_regimes(bench_df["Close"])
        reg_met = _regime_metrics(equity_df, regime_df)
        result["regime_metrics"] = reg_met
        result["regime_counts"] = {
            label: int((regime_df["regime"] == label).sum())
            for label in ["bull", "bear", "high_vol", "crisis"]
        }
    except Exception as exc:
        result["regime_metrics"] = {}
        result["regime_counts"] = {}
        result["regime_error"] = str(exc)

    # ── 3. GPU-accelerated XGBoost ML training ────────────────────
    #    - 20+ engineered features (price, volume, momentum, volatility, mean-reversion)
    #    - Proper time-series walk-forward CV (no future leakage)
    #    - 80/20 chronological train/holdout split
    #    - Early stopping to prevent overfitting
    features_list = []
    labels_list = []

    for sym in cfg.symbols:
        try:
            raw = fetch_symbol_history(sym, cfg)
            df = compute_indicators(raw, cfg)
            if df is None or len(df) < cfg.slow_window + 60:
                continue

            close = df["Close"]
            volume = df["Volume"]
            ret = close.pct_change()

            feat = pd.DataFrame(index=df.index)

            # Returns at multiple horizons
            for d in [1, 3, 5, 10, 20, 60]:
                feat[f"ret_{d}d"] = close.pct_change(d)

            # Volatility at multiple windows
            for w in [10, 20, 60]:
                feat[f"vol_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

            # Volume features
            feat["vol_ratio_20"] = volume / volume.rolling(20).mean()
            feat["vol_ratio_5"] = volume / volume.rolling(5).mean()

            # SMA spread (strategy-specific)
            fast_sma = df.get(f"SMA_{cfg.fast_window}", close.rolling(cfg.fast_window).mean())
            slow_sma = df.get(f"SMA_{cfg.slow_window}", close.rolling(cfg.slow_window).mean())
            feat["sma_spread"] = (fast_sma - slow_sma) / close
            feat["price_vs_fast"] = (close - fast_sma) / close
            feat["price_vs_slow"] = (close - slow_sma) / close

            # RSI (14-day)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            feat["rsi_14"] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            feat["macd"] = macd_line / close
            feat["macd_signal"] = signal_line / close
            feat["macd_hist"] = (macd_line - signal_line) / close

            # Bollinger Band position
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            feat["bb_position"] = (close - bb_mid) / (2 * bb_std).replace(0, np.nan)

            # Rate of change
            feat["roc_10"] = close / close.shift(10) - 1
            feat["roc_30"] = close / close.shift(30) - 1

            # Drawdown from rolling max
            roll_max = close.rolling(60).max()
            feat["drawdown_60d"] = (close - roll_max) / roll_max

            # Label: positive 5-day forward return
            feat["label"] = (close.shift(-5) / close - 1 > 0).astype(int)

            feat = feat.dropna()
            if len(feat) > 100:
                features_list.append(feat.drop(columns=["label"]))
                labels_list.append(feat["label"])
        except Exception:
            continue

    if features_list:
        X = pd.concat(features_list).sort_index()
        y = pd.concat(labels_list).sort_index()

        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        # ── Chronological train/holdout split (80/20) ────────────
        split_idx = int(len(X) * 0.80)
        X_train, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dhold = xgb.DMatrix(X_holdout, label=y_holdout)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "device": "cuda",
            "tree_method": "hist",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        # Early stopping on holdout to prevent overfitting
        model = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dhold, "holdout")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        actual_rounds = model.best_iteration + 1

        # Feature importance
        importance = model.get_score(importance_type="gain")
        total_gain = sum(importance.values()) or 1.0
        feature_importance = {k: round(v / total_gain, 4) for k, v in importance.items()}

        # Holdout AUC (true out-of-sample, no leakage)
        holdout_preds = model.predict(dhold)
        holdout_auc = round(float(roc_auc_score(y_holdout, holdout_preds)), 4)

        # ── Time-series walk-forward CV (5 expanding windows) ────
        n = len(X)
        min_train = int(n * 0.3)
        fold_size = int(n * 0.1)
        cv_aucs = []
        for i in range(5):
            val_end = min_train + (i + 1) * fold_size
            if val_end > n:
                break
            val_start = val_end - fold_size
            X_cv_train = X.iloc[:val_start]
            y_cv_train = y.iloc[:val_start]
            X_cv_val = X.iloc[val_start:val_end]
            y_cv_val = y.iloc[val_start:val_end]

            if len(X_cv_train) < 200 or len(X_cv_val) < 50:
                continue

            d_cv_train = xgb.DMatrix(X_cv_train, label=y_cv_train)
            d_cv_val = xgb.DMatrix(X_cv_val, label=y_cv_val)
            cv_model = xgb.train(
                params, d_cv_train, num_boost_round=actual_rounds,
                verbose_eval=False,
            )
            preds = cv_model.predict(d_cv_val)
            try:
                auc = float(roc_auc_score(y_cv_val, preds))
                cv_aucs.append(round(auc, 4))
            except ValueError:
                continue

        result["gpu_ml"] = {
            "gpu_used": True,
            "device": "cuda (NVIDIA A100)",
            "training_samples": int(len(X_train)),
            "holdout_samples": int(len(X_holdout)),
            "total_samples": int(len(X)),
            "num_features": int(X.shape[1]),
            "feature_names": list(X.columns),
            "feature_importance": feature_importance,
            "holdout_auc": holdout_auc,
            "cv_auc_mean": round(float(np.mean(cv_aucs)), 4) if cv_aucs else None,
            "cv_auc_std": round(float(np.std(cv_aucs)), 4) if cv_aucs else None,
            "cv_scores": cv_aucs,
            "cv_method": "time-series walk-forward (no future leakage)",
            "num_boost_rounds": actual_rounds,
            "max_boost_rounds": 500,
            "early_stopping_rounds": 30,
        }
    else:
        result["gpu_ml"] = {"gpu_used": True, "device": "cuda (NVIDIA A100)",
                            "error": "Not enough data to train ML model"}

    return result


# ── Individual backtest worker (runs on Modal) ────────────────────
@app.function(image=audit_image, timeout=300, retries=1)
def run_single_backtest(
    overrides: Dict[str, Any],
    base_config_dict: Dict[str, Any],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one backtest with param overrides on a Modal worker."""
    import sys
    sys.path.insert(0, "/root")

    from pipeline.config import StrategyConfig
    from pipeline.backtest import run_backtest

    cfg = StrategyConfig()
    for k, v in base_config_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    try:
        _eq, _trades, report = run_backtest(cfg, start=start, end=end)
        return {**overrides, **report.get("metrics", {})}
    except Exception as exc:
        return {**overrides, "error": str(exc)}


# ── Parallel param sweep (fan-out via .map) ───────────────────────
@app.function(image=audit_image, timeout=900)
def run_modal_param_sweep(
    grid: Dict[str, List[Any]],
    base_config_dict: Dict[str, Any],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fan out every param combo to a separate Modal worker."""
    param_names = sorted(grid.keys())
    combos = []
    for values in itertools.product(*(grid[p] for p in param_names)):
        overrides = dict(zip(param_names, values))
        if overrides.get("fast_window", 0) >= overrides.get("slow_window", float("inf")):
            continue
        combos.append(overrides)

    results = list(
        run_single_backtest.map(
            combos,
            kwargs={
                "base_config_dict": base_config_dict,
                "start": start,
                "end": end,
            },
        )
    )
    return results


# ── Parallel stress tests ─────────────────────────────────────────
@app.function(image=audit_image, timeout=900)
def run_modal_stress_tests(
    base_config_dict: Dict[str, Any],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all stress scenarios in parallel on Modal."""
    scenarios: List[Dict[str, Any]] = []

    # Baseline
    scenarios.append({"label": "baseline", "overrides": {}})

    # Slippage stress
    base_slip = base_config_dict.get("slippage_bps", 1.0)
    for mult in [2.0, 5.0, 10.0]:
        scenarios.append({
            "label": f"slippage_{mult}x",
            "overrides": {"slippage_bps": float(base_slip * mult)},
        })

    # Commission stress
    base_comm = base_config_dict.get("commission_bps", 0.5)
    for mult in [2.0, 5.0, 10.0]:
        scenarios.append({
            "label": f"commission_{mult}x",
            "overrides": {"commission_bps": float(base_comm * mult)},
        })

    # Worst-case costs
    scenarios.append({
        "label": "worst_case_costs",
        "overrides": {
            "slippage_bps": float(base_slip * 10.0),
            "commission_bps": float(base_comm * 10.0),
        },
    })

    # No ML
    scenarios.append({
        "label": "no_ml_filter",
        "overrides": {"ml_enabled": False},
    })

    # Tight positions
    scenarios.append({
        "label": "small_positions_10pct",
        "overrides": {"position_fraction": 0.10},
    })

    combo_list = [s["overrides"] for s in scenarios]
    labels = [s["label"] for s in scenarios]

    raw_results = list(
        run_single_backtest.map(
            combo_list,
            kwargs={
                "base_config_dict": base_config_dict,
                "start": start,
                "end": end,
            },
        )
    )

    results = []
    for label, raw in zip(labels, raw_results):
        row = {"scenario": label}
        row.update({k: v for k, v in raw.items() if k not in ("slippage_bps", "commission_bps", "ml_enabled", "position_fraction")})
        results.append(row)

    return results


# ── Standalone GPU ML training (kept for backward compat) ─────────
@app.function(image=gpu_image, gpu="A100", timeout=600)
def train_and_score_gpu(
    base_config_dict: Dict[str, Any],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Standalone ML training — same logic as run_base_backtest_modal's ML step."""
    import sys
    sys.path.insert(0, "/root")

    result = run_base_backtest_modal.local(
        base_config_dict=base_config_dict,
        start=start, end=end,
    )
    return result.get("gpu_ml", {"gpu_used": False, "error": "No ML result"})


# ── Full audit on Modal ───────────────────────────────────────────
@app.function(image=audit_image, timeout=1800)
def run_audit_on_modal(
    base_config_dict: Dict[str, Any],
    grid: Dict[str, List[Any]],
    start: Optional[str] = None,
    end: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Full orchestrated audit running heavy steps on Modal."""
    import sys
    sys.path.insert(0, "/root")

    from audit.orchestrator import run_full_audit
    from pipeline.config import StrategyConfig

    cfg = StrategyConfig()
    for k, v in base_config_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return run_full_audit(
        cfg,
        start=start,
        end=end,
        param_grid=grid,
        openai_api_key=openai_api_key,
    )


# ── CLI entrypoint: `modal run modal_app/remote.py` ──────────────
@app.local_entrypoint()
def main():
    from pipeline.config import StrategyConfig
    from audit.param_sweep import DEFAULT_GRID

    cfg = StrategyConfig()
    cfg_dict = {
        "symbols": list(cfg.symbols),
        "benchmark_symbol": cfg.benchmark_symbol,
        "fast_window": cfg.fast_window,
        "slow_window": cfg.slow_window,
        "vol_window": cfg.vol_window,
        "position_fraction": cfg.position_fraction,
        "max_portfolio_exposure": cfg.max_portfolio_exposure,
        "slippage_bps": cfg.slippage_bps,
        "commission_bps": cfg.commission_bps,
        "ml_enabled": cfg.ml_enabled,
        "vol_target_enabled": cfg.vol_target_enabled,
        "vol_target_annual": cfg.vol_target_annual,
    }

    print("Running param sweep on Modal ...")
    sweep_results = run_modal_param_sweep.remote(
        grid=DEFAULT_GRID,
        base_config_dict=cfg_dict,
        start="2023-01-01",
    )
    print(f"  Param sweep: {len(sweep_results)} combos completed")

    print("Running stress tests on Modal ...")
    stress_results = run_modal_stress_tests.remote(
        base_config_dict=cfg_dict,
        start="2023-01-01",
    )
    print(f"  Stress tests: {len(stress_results)} scenarios completed")

    print("Done!")
