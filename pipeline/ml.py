import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# sklearn is required for XGBoost's sklearn API + metrics/splitting
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Allow module or script import
try:
    from .config import StrategyConfig
except ImportError:  # direct script execution fallback
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from pipeline.config import StrategyConfig


def _require_xgboost():
    """
    Import xgboost lazily so the rest of the project can run even if the local
    Python/numpy/xgboost binaries are incompatible.
    """
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for XGBoost's sklearn API (XGBClassifier).\n"
            "Install it via: `python -m pip install scikit-learn` (or reinstall `-r requirements.txt`).\n"
            f"Original import error: {exc}"
        ) from exc
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import xgboost. This is almost always an environment mismatch.\n"
            "Recommended fix:\n"
            "- Use Python 3.11 in a fresh venv\n"
            "- Install pinned deps: numpy==1.26.4, pandas==2.1.4, xgboost==1.7.6\n"
            "- On macOS also install OpenMP: `brew install libomp`\n"
            f"Original import error: {exc}"
        ) from exc
    return xgb


def _normalize_history(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    rename_map = {
        "Adj Close": "AdjClose",
        "Close": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    if "Close" not in df.columns and "AdjClose" in df.columns:
        df["Close"] = df["AdjClose"]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def _download_history(symbol: str, years: int) -> pd.DataFrame:
    end_date = datetime.utcnow().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=int(years * 365) + 10)

    def _dl(group_by: str) -> pd.DataFrame:
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

    df = _normalize_history(_dl("column"))
    required = ["Close", "Open", "High", "Low", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        df_retry = _normalize_history(_dl("ticker"))
        missing_retry = [c for c in required if c not in df_retry.columns]
        if not missing_retry:
            df = df_retry
        else:
            raise RuntimeError(
                f"Missing columns for {symbol}: {missing_retry}; got {list(df_retry.columns)}"
            )
    return df


def backfill_and_store(symbol: str, years: int, config: StrategyConfig) -> pd.DataFrame:
    df_new = _download_history(symbol, years)
    raw_path = config.data_raw_dir / f"{symbol}.csv"
    if raw_path.exists():
        existing = pd.read_csv(raw_path, parse_dates=["Date"]).set_index("Date")
        existing = existing.loc[:, ~existing.columns.duplicated()]
        union_cols = sorted(set(existing.columns) | set(df_new.columns))
        existing = existing.reindex(columns=union_cols)
        df_new = df_new.reindex(columns=union_cols)
        df_new = pd.concat([existing, df_new], axis=0)
    df_new = df_new[~df_new.index.duplicated(keep="last")].sort_index()
    df_new.reset_index(names="Date").to_csv(raw_path, index=False)
    return df_new


def build_features(
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    symbol: str,
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    df = asset_df.copy()
    bench = benchmark_df.copy()
    common = df.index.intersection(bench.index)
    df = df.loc[common]
    bench = bench.loc[common]

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = df["Close"].pct_change(1)
    feat["ret_5"] = df["Close"].pct_change(5)
    feat["ret_10"] = df["Close"].pct_change(10)
    feat["ret_20"] = df["Close"].pct_change(20)
    feat["ret_60"] = df["Close"].pct_change(60)

    feat["sma_fast"] = df["Close"].rolling(20).mean()
    feat["sma_slow"] = df["Close"].rolling(100).mean()
    feat["sma_ratio"] = feat["sma_fast"] / feat["sma_slow"]
    feat["vol_20"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    feat["rs_60"] = df["Close"].pct_change(60) - bench["Close"].pct_change(60)
    feat["gap"] = df["Open"] / df["Close"].shift(1) - 1.0
    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    feat["vol_z"] = (df["Volume"] - vol_mean) / vol_std

    if include_target:
        # IMPORTANT: This uses NEXT day's close and therefore MUST ONLY be used for training labels,
        # never for inference. (Inference must not touch future prices.)
        fwd_ret = df["Close"].shift(-1) / df["Close"] - 1.0
        feat["target"] = (fwd_ret > 0).astype(int)
    feat["symbol"] = symbol
    feat["date"] = feat.index

    feat = feat.dropna().reset_index(drop=True)
    return feat


def prepare_training_data(config: StrategyConfig) -> pd.DataFrame:
    benchmark_df = backfill_and_store(config.benchmark_symbol, config.ml_years, config)
    benchmark_df = benchmark_df.sort_index()
    feats: List[pd.DataFrame] = []
    for sym in config.symbols:
        asset_df = backfill_and_store(sym, config.ml_years, config)
        asset_df = asset_df.sort_index()
        feats.append(build_features(asset_df, benchmark_df, sym, include_target=True))
    return pd.concat(feats, axis=0, ignore_index=True)


def train_pooled_classifier(config: StrategyConfig) -> Dict[str, float]:
    df = prepare_training_data(config)
    if df.empty:
        raise RuntimeError("No data to train on")

    df = df.sort_values("date").reset_index(drop=True)
    df = pd.get_dummies(df, columns=["symbol"], drop_first=False)
    y = df["target"]
    X = df.drop(columns=["target", "date"])

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    xgb = _require_xgboost()
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, preds)) if len(y_test) else float("nan")
    auc = float(roc_auc_score(y_test, proba)) if len(y_test) else float("nan")
    ll = float(log_loss(y_test, proba)) if len(y_test) else float("nan")

    config.models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(config.ml_model_path)
    meta = {
        "feature_columns": list(X.columns),
        "symbols": sorted(set(df.filter(like="symbol_").columns)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": acc,
        "test_auc": auc,
        "test_logloss": ll,
        "dates": {
            "min": df["date"].min().isoformat(),
            "max": df["date"].max().isoformat(),
        },
        "model_name": config.ml_model_name,
        "created_at": datetime.utcnow().isoformat(),
    }
    with config.ml_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return {"test_accuracy": acc, "test_auc": auc, "test_logloss": ll, "rows": len(df)}


def tune_and_train_pooled_classifier(
    config: StrategyConfig,
    trials: int = 25,
    seed: int = 7,
) -> Dict[str, any]:
    """
    Simple time-aware hyperparameter tuning (random search).

    - We keep a strict chronological split: last 20% of dates is validation.
    - We optimize logloss (lower is better), and also report AUC/accuracy.
    """
    df = prepare_training_data(config)
    if df.empty:
        raise RuntimeError("No data to train on")

    df = df.sort_values("date").reset_index(drop=True)
    df = pd.get_dummies(df, columns=["symbol"], drop_first=False)

    y = df["target"].astype(int)
    dates = pd.to_datetime(df["date"], errors="coerce")
    X = df.drop(columns=["target", "date"])

    unique_dates = pd.Series(dates.dropna().unique()).sort_values()
    if len(unique_dates) < 50:
        raise RuntimeError("Not enough unique dates for time split / tuning.")

    cutoff = unique_dates.iloc[int(len(unique_dates) * 0.8)]
    train_mask = dates <= cutoff
    valid_mask = dates > cutoff

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]

    if len(X_valid) == 0 or len(X_train) == 0:
        raise RuntimeError("Time split produced empty train/valid sets.")

    xgb = _require_xgboost()
    rng = np.random.default_rng(seed)

    def sample_params() -> Dict[str, any]:
        return {
            "n_estimators": int(rng.integers(200, 900)),
            "learning_rate": float(rng.uniform(0.01, 0.15)),
            "max_depth": int(rng.integers(2, 7)),
            "min_child_weight": float(rng.uniform(1.0, 20.0)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda": float(rng.uniform(0.0, 20.0)),
            "reg_alpha": float(rng.uniform(0.0, 5.0)),
            "gamma": float(rng.uniform(0.0, 5.0)),
        }

    best = {
        "logloss": float("inf"),
        "auc": float("nan"),
        "accuracy": float("nan"),
        "params": None,
        "model": None,
    }

    for _ in range(trials):
        params = sample_params()
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            **params,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        ll = float(log_loss(y_valid, proba))
        auc = float(roc_auc_score(y_valid, proba)) if len(np.unique(y_valid)) > 1 else float("nan")
        preds = (proba >= 0.5).astype(int)
        acc = float(accuracy_score(y_valid, preds))

        if ll < best["logloss"]:
            best = {"logloss": ll, "auc": auc, "accuracy": acc, "params": params, "model": model}

    if best["model"] is None:
        raise RuntimeError("Tuning failed to produce a model.")

    # Refit on all data (train+valid) using best params
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        **best["params"],
    )
    final_model.fit(X, y)

    config.models_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(config.ml_model_path)

    meta = {
        "feature_columns": list(X.columns),
        "symbols": sorted(set(df.filter(like="symbol_").columns)),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "valid_accuracy": best["accuracy"],
        "valid_auc": best["auc"],
        "valid_logloss": best["logloss"],
        "best_params": best["params"],
        "dates": {
            "min": pd.to_datetime(df["date"]).min().isoformat(),
            "max": pd.to_datetime(df["date"]).max().isoformat(),
            "cutoff": pd.to_datetime(cutoff).isoformat(),
        },
        "model_name": config.ml_model_name,
        "created_at": datetime.utcnow().isoformat(),
        "tuning": {"trials": trials, "seed": seed},
    }
    with config.ml_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return meta


def load_model_and_meta(config: StrategyConfig):
    if not config.ml_model_path.exists() or not config.ml_meta_path.exists():
        return None, None
    xgb = _require_xgboost()
    model = xgb.XGBClassifier()
    model.load_model(config.ml_model_path)
    with config.ml_meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def build_latest_feature_row(
    symbol: str,
    asset_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    meta: Dict[str, any],
) -> Optional[pd.DataFrame]:
    if asset_df is None or asset_df.empty:
        return None
    if benchmark_df is None or benchmark_df.empty:
        return None
    common = asset_df.index.intersection(benchmark_df.index)
    if len(common) < 120:
        return None
    asset_df = asset_df.loc[common]
    benchmark_df = benchmark_df.loc[common]
    feat = build_features(asset_df, benchmark_df, symbol, include_target=False)
    if feat.empty:
        return None
    latest = feat.iloc[[-1]].copy()
    latest = pd.get_dummies(latest, columns=["symbol"], drop_first=False)
    # align columns to training
    for col in meta.get("feature_columns", []):
        if col not in latest.columns:
            latest[col] = 0.0
    latest = latest[meta["feature_columns"]]
    return latest
