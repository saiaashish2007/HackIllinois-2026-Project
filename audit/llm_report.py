"""
LLM-powered audit report generation.

Takes structured audit results and produces a human-readable narrative
explaining strategy fragility, risk flags, and survivability.

Supports:
- OpenAI API (if OPENAI_API_KEY is set)
- Fallback template-based report (no API key needed)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _build_structured_summary(audit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the audit object into a clean JSON summary suitable for
    inclusion in an LLM prompt.
    """
    base = audit.get("base_metrics", {})
    regime = audit.get("regime_metrics", {})
    survivability = audit.get("survivability", {})
    stress = audit.get("stress_summary", {})
    param = audit.get("param_sweep_summary", {})

    return {
        "strategy_name": audit.get("strategy_name", "SMA Crossover"),
        "overall_sharpe": base.get("sharpe"),
        "overall_cagr": base.get("cagr"),
        "overall_max_drawdown": base.get("max_drawdown"),
        "overall_annual_vol": base.get("annual_vol"),
        "avg_recovery_days": base.get("avg_recovery_days"),
        "worst_recovery_days": base.get("worst_recovery_days"),
        "time_underwater_pct": base.get("time_underwater_pct"),
        "regime_sharpes": {k: v.get("sharpe") for k, v in regime.items()},
        "regime_returns": {k: v.get("total_return") for k, v in regime.items()},
        "survivability_score": survivability.get("score"),
        "survivability_grade": survivability.get("grade"),
        "survivability_components": survivability.get("components", {}),
        "param_stability": param.get("stability_score"),
        "cost_sensitivity": stress.get("cost_sensitivity_score"),
        "worst_case_sharpe": stress.get("worst_case_sharpe"),
        "num_param_combos_tested": param.get("num_combos"),
    }


_SYSTEM_PROMPT = """You are a senior quantitative analyst reviewing a trading strategy audit.
Given the structured audit data, produce a concise report with these sections:

1. **Executive Summary** (2-3 sentences)
2. **Performance Overview** (key metrics)
3. **Regime Analysis** (how the strategy behaves in bull/bear/high_vol/crisis)
4. **Risk Flags** (bullet list of concerns)
5. **Parameter Sensitivity** (is the strategy robust to parameter changes?)
6. **Cost Sensitivity** (how much do transaction costs hurt?)
7. **Survivability Assessment** (final score interpretation)
8. **Recommendations** (2-3 actionable items)

Be specific. Use numbers from the data. Be honest about weaknesses."""


def generate_llm_report(
    audit: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a narrative audit report using an LLM.
    Falls back to a template if no API key is available.
    """
    summary = _build_structured_summary(audit)
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass

    if key:
        try:
            return _call_openai(summary, key, model)
        except Exception as exc:
            return _template_report(summary, fallback_reason=str(exc))

    return _template_report(summary)


def _call_openai(summary: Dict[str, Any], api_key: str, model: str) -> str:
    """Call OpenAI chat completions API."""
    try:
        from openai import OpenAI
    except ImportError:
        return _template_report(summary, fallback_reason="openai package not installed")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Here is the structured audit data:\n\n```json\n{json.dumps(summary, indent=2, default=str)}\n```\n\nGenerate the audit report.",
            },
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content or _template_report(summary)


_REGIME_FRIENDLY = {
    "bull": ("Rising Market", "📈"),
    "bear": ("Falling Market", "📉"),
    "high_vol": ("High Volatility", "🌊"),
    "crisis": ("Market Crash", "💥"),
}

_COMPONENT_FRIENDLY = {
    "regime_consistency": "Works in All Markets",
    "param_stability": "Not Overfit",
    "cost_resilience": "Survives Real Costs",
    "drawdown_penalty": "Manageable Losses",
    "base_performance": "Raw Performance",
}


def _score_bar(value: float) -> str:
    filled = int(min(max(value, 0), 1) * 10)
    return "🟩" * filled + "⬜" * (10 - filled)


def _pct_str(val) -> str:
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _grade_verdict(grade: str) -> str:
    return {
        "A": "This strategy is **well-built and resilient**. It holds up across different markets, parameter tweaks, and cost increases.",
        "B": "This strategy is **solid with minor concerns**. It performs well but has a few areas worth monitoring.",
        "C": "This strategy is **fair but has notable weaknesses**. It may struggle in certain market conditions or break under parameter changes.",
        "D": "This strategy has **significant vulnerabilities**. It is sensitive to market conditions or parameter choices and may not hold up in live trading.",
        "F": "This strategy is **fragile and high-risk**. It is unlikely to survive real-world trading conditions without major changes.",
    }.get(grade, "Unable to determine a verdict.")


def _template_report(
    summary: Dict[str, Any],
    fallback_reason: Optional[str] = None,
) -> str:
    s = summary
    score = s.get("survivability_score", "N/A")
    grade = s.get("survivability_grade", "?")
    sharpe = s.get("overall_sharpe")
    cagr = s.get("overall_cagr")
    max_dd = s.get("overall_max_drawdown")
    vol = s.get("overall_annual_vol")

    regime_sharpes = s.get("regime_sharpes", {})
    regime_returns = s.get("regime_returns", {})
    components = s.get("survivability_components", {})

    # -- Executive verdict --
    verdict = _grade_verdict(grade)

    # -- Performance table --
    perf_rows = []
    perf_rows.append(f"| **Risk-Adjusted Return (Sharpe)** | **{_fmt(sharpe)}** | {'Good' if _to_float(sharpe) >= 1.0 else 'Below average' if _to_float(sharpe) >= 0.5 else 'Poor'} — above 1.0 is considered good |")
    perf_rows.append(f"| **Annual Growth Rate** | **{_pct_str(cagr)}** | How fast your money compounds each year |")
    perf_rows.append(f"| **Worst Drop (Max Drawdown)** | **{_pct_str(max_dd)}** | {'Manageable' if _to_float(max_dd) > -0.20 else 'Concerning' if _to_float(max_dd) > -0.35 else 'Severe'} — the biggest peak-to-trough loss |")
    perf_rows.append(f"| **Yearly Volatility** | **{_pct_str(vol)}** | How much the portfolio swings up and down |")

    avg_rec = s.get("avg_recovery_days")
    worst_rec = s.get("worst_recovery_days")
    uw_pct = s.get("time_underwater_pct")
    if avg_rec is not None:
        try:
            ar = float(avg_rec)
            ar_interp = "Fast" if ar < 20 else "Moderate" if ar < 60 else "Slow"
            perf_rows.append(f"| **Avg Recovery Time** | **{ar:.0f} days** | {ar_interp} — how quickly the strategy bounces back on average |")
        except (TypeError, ValueError):
            pass
    if worst_rec is not None:
        try:
            wr = float(worst_rec)
            wr_interp = "Acceptable" if wr < 60 else "Extended" if wr < 150 else "Very long"
            perf_rows.append(f"| **Worst Recovery Time** | **{wr:.0f} days** | {wr_interp} — the longest time to climb back to a previous high |")
        except (TypeError, ValueError):
            pass
    if uw_pct is not None:
        try:
            uw = float(uw_pct) * 100
            uw_interp = "Healthy" if uw < 30 else "Fair" if uw < 55 else "Concerning"
            perf_rows.append(f"| **Time Underwater** | **{uw:.1f}%** | {uw_interp} — percentage of days spent below the previous peak |")
        except (TypeError, ValueError):
            pass

    # -- Regime breakdown --
    regime_rows = []
    for regime_key in ["bull", "bear", "high_vol", "crisis"]:
        name, icon = _REGIME_FRIENDLY.get(regime_key, (regime_key, ""))
        rs = regime_sharpes.get(regime_key)
        rr = regime_returns.get(regime_key)
        if rs is None and rr is None:
            continue
        rs_val = _to_float(rs)
        status = "Profitable" if rs_val > 0 else "Losing money" if rs_val < 0 else "Break-even"
        regime_rows.append(f"| {icon} **{name}** | {_fmt(rs)} | {_pct_str(rr)} | {status} |")

    # -- Risk flags --
    flags = []
    if _to_float(sharpe) < 0.5:
        flags.append("**Low risk-adjusted return** — the strategy doesn't generate enough return for the risk taken")
    if _to_float(max_dd) < -0.30:
        flags.append(f"**Deep drawdown ({_pct_str(max_dd)})** — you could lose a significant chunk of your investment at some point")
    for regime_key, rs in regime_sharpes.items():
        if _to_float(rs) < 0:
            name, _ = _REGIME_FRIENDLY.get(regime_key, (regime_key, ""))
            flags.append(f"**Loses money in {name.lower()}s** — the strategy struggles when the market {name.lower().replace(' market', 's')}")
    if _to_float(s.get("param_stability")) < 0.5:
        flags.append("**Possibly overfit** — small changes to strategy settings cause big performance swings")
    if _to_float(s.get("cost_sensitivity")) < 0.5:
        flags.append("**Cost-sensitive** — trading fees and slippage eat into profits significantly")

    if not flags:
        flags_section = "> No major risk flags detected."
    else:
        flags_section = "\n".join(f"- {f}" for f in flags)

    # -- Survivability components --
    comp_rows = []
    for key in ["regime_consistency", "param_stability", "cost_resilience", "drawdown_penalty", "base_performance"]:
        val = _to_float(components.get(key))
        name = _COMPONENT_FRIENDLY.get(key, key)
        bar = _score_bar(val)
        comp_rows.append(f"| **{name}** | {bar} | **{val*100:.0f}%** |")

    # -- Recommendations --
    recs = []
    if _to_float(max_dd) < -0.20:
        recs.append("Consider **reducing position sizes** or adding stop-loss rules to limit drawdowns")
    else:
        recs.append("Drawdown levels are acceptable — no immediate action needed on risk limits")

    if _to_float(s.get("param_stability")) < 0.5:
        recs.append("**Investigate overfitting** — try testing with out-of-sample data or simplifying the strategy")
    else:
        recs.append("Parameter stability looks good — the strategy isn't overly dependent on exact settings")

    if _to_float(s.get("cost_sensitivity")) < 0.5:
        recs.append("**Account for higher trading costs** — use limit orders and trade less frequently to reduce impact")
    else:
        recs.append("Transaction costs don't significantly impact this strategy — it should hold up in live markets")

    if _to_float(sharpe) < 0.3:
        recs.append("**Reconsider the strategy entirely** — the risk-adjusted return is too low to justify the risk")

    recs_section = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs))

    param_stab = _to_float(s.get("param_stability"))
    param_verdict = (
        "very stable — robust to setting changes" if param_stab >= 0.7
        else "moderately stable — some sensitivity to settings" if param_stab >= 0.4
        else "unstable — highly sensitive to parameter choices"
    )

    cost_res = _to_float(s.get("cost_sensitivity"))
    cost_verdict = (
        "very resilient — costs barely affect performance" if cost_res >= 0.8
        else "moderately resilient — costs reduce profits noticeably" if cost_res >= 0.5
        else "fragile — profits largely disappear under realistic costs"
    )

    header = ""
    if fallback_reason:
        header = f"> *Template report — for a richer AI-written analysis, add an OpenAI API key.*\n\n"

    regime_table = ""
    if regime_rows:
        regime_table = (
            "| Market Phase | Risk-Adj Return | Total Return | Verdict |\n"
            "|:---|:---:|:---:|:---|\n"
            + "\n".join(regime_rows)
        )
    else:
        regime_table = "> No regime data available."

    return f"""{header}---

## Executive Summary

**Score: {score}/100 (Grade {grade})**

{verdict}

---

## Performance Overview

| Metric | Value | Interpretation |
|:---|:---:|:---|
{chr(10).join(perf_rows)}

---

## How It Performs Across Market Conditions

{regime_table}

---

## Risk Flags

{flags_section}

---

## Is the Strategy Overfit?

**Stability: {param_stab*100:.0f}%** — {param_verdict}

We tested **{s.get('num_param_combos_tested', 'N/A')} variations** of the strategy settings. {
    'Performance stayed consistent, suggesting the strategy captures a real pattern.'
    if param_stab >= 0.7
    else 'Performance varied somewhat — the strategy works but is sensitive to exact settings.'
    if param_stab >= 0.4
    else 'Performance changed dramatically — this strategy may be curve-fit to historical data.'
}

---

## Can It Handle Real-World Costs?

**Cost Resilience: {cost_res*100:.0f}%** — {cost_verdict}

Worst-case scenario (10x slippage + 10x commissions): Sharpe ratio drops to **{_fmt(s.get('worst_case_sharpe'))}**. {
    'Even under extreme cost assumptions, the strategy remains profitable.'
    if _to_float(s.get('worst_case_sharpe')) > 0
    else 'Under extreme costs, the strategy turns unprofitable — but normal costs are fine.'
    if cost_res >= 0.5
    else 'The strategy is highly cost-sensitive and may not survive real trading friction.'
}

---

## Score Breakdown

| Component | | Score |
|:---|:---|:---:|
{chr(10).join(comp_rows)}

---

## Recommendations

{recs_section}
"""


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if v == v else default  # NaN check
    except (TypeError, ValueError):
        return default


def _fmt(val: Any) -> str:
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)
