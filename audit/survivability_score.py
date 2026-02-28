"""
Survivability score: a single composite number [0–100] that captures
how likely a strategy is to survive real-world deployment.

Higher = more robust.  The score penalizes:
- Poor performance in bear / crisis regimes
- High sensitivity to parameter choices
- Large degradation under cost stress
- Deep drawdowns
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "regime_consistency": 0.25,
    "param_stability": 0.25,
    "cost_resilience": 0.20,
    "drawdown_penalty": 0.15,
    "base_performance": 0.15,
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _regime_consistency_score(regime_metrics: Dict[str, Dict[str, Any]]) -> float:
    """
    How consistent is the strategy across regimes?
    Penalizes negative Sharpe in any regime heavily.
    """
    sharpes = []
    for label, m in regime_metrics.items():
        s = _safe_float(m.get("sharpe"))
        sharpes.append(s)

    if not sharpes:
        return 0.0

    avg = np.mean(sharpes)
    negative_count = sum(1 for s in sharpes if s < 0)
    negative_penalty = negative_count / len(sharpes)

    raw = max(0.0, min(1.0, (avg + 1.0) / 3.0))
    return float(round(raw * (1.0 - 0.5 * negative_penalty), 4))


def _drawdown_score(base_metrics: Dict[str, Any]) -> float:
    """
    Convert max drawdown into a 0-1 score.
    0% DD → 1.0,  ≥50% DD → 0.0
    """
    dd = abs(_safe_float(base_metrics.get("max_drawdown")))
    return float(round(max(0.0, 1.0 - dd / 0.50), 4))


def _base_performance_score(base_metrics: Dict[str, Any]) -> float:
    """
    Normalize Sharpe into [0, 1]:  Sharpe ≥ 2 → 1.0,  Sharpe ≤ -1 → 0.0
    """
    sharpe = _safe_float(base_metrics.get("sharpe"))
    return float(round(max(0.0, min(1.0, (sharpe + 1.0) / 3.0)), 4))


def compute_survivability_score(
    *,
    base_metrics: Dict[str, Any],
    regime_metrics: Dict[str, Dict[str, Any]],
    param_stability: float,
    cost_resilience: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute the final survivability score [0–100] and return a breakdown.
    """
    w = weights or DEFAULT_WEIGHTS

    components = {
        "regime_consistency": _regime_consistency_score(regime_metrics),
        "param_stability": float(max(0.0, min(1.0, param_stability))),
        "cost_resilience": float(max(0.0, min(1.0, cost_resilience))),
        "drawdown_penalty": _drawdown_score(base_metrics),
        "base_performance": _base_performance_score(base_metrics),
    }

    weighted_sum = sum(components[k] * w.get(k, 0.0) for k in components)
    total_weight = sum(w.get(k, 0.0) for k in components)
    score = (weighted_sum / total_weight * 100.0) if total_weight > 0 else 0.0

    grade = "F"
    if score >= 80:
        grade = "A"
    elif score >= 65:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 35:
        grade = "D"

    return {
        "score": round(score, 1),
        "grade": grade,
        "components": {k: round(v, 4) for k, v in components.items()},
        "weights": dict(w),
    }
