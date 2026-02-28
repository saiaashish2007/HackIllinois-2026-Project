"""
Natural-language strategy parser.

Takes a plain-English strategy description and extracts a StrategyConfig.

Two modes:
  1. LLM-powered (if OpenAI key is available) — most flexible
  2. Rule-based fallback — keyword extraction, works offline
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import StrategyConfig


# Well-known ticker aliases
TICKER_ALIASES = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL",
    "alphabet": "GOOGL", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "broadcom": "AVGO", "amd": "AMD", "salesforce": "CRM", "oracle": "ORCL",
    "adobe": "ADBE", "intel": "INTC", "tsmc": "TSM", "qualcomm": "QCOM",
    "ibm": "IBM", "sap": "SAP", "servicenow": "NOW", "palo alto": "PANW",
    "netflix": "NFLX", "tesla": "TSLA", "spy": "SPY", "qqq": "QQQ",
    "gold": "GLD", "bonds": "TLT", "treasury": "TLT",
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman": "GS", "goldman sachs": "GS",
    "morgan stanley": "MS", "bank of america": "BAC", "visa": "V", "mastercard": "MA",
    "walmart": "WMT", "costco": "COST", "home depot": "HD", "mcdonalds": "MCD",
    "starbucks": "SBUX", "nike": "NKE", "disney": "DIS",
    "johnson & johnson": "JNJ", "j&j": "JNJ", "unitedhealth": "UNH",
    "pfizer": "PFE", "abbvie": "ABBV", "eli lilly": "LLY", "lilly": "LLY", "merck": "MRK",
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "micron": "MU", "snowflake": "SNOW", "datadog": "DDOG", "marvell": "MRVL",
    "snapchat": "SNAP", "pinterest": "PINS",
    "real estate": "VNQ", "reits": "VNQ", "commodities": "DBC", "utilities": "XLU",
}

SECTOR_TICKERS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
    "semiconductor": ["NVDA", "AMD", "AVGO", "INTC", "TSM", "QCOM", "MU", "MRVL"],
    "chip": ["NVDA", "AMD", "AVGO", "INTC", "TSM", "QCOM", "MU", "MRVL"],
    "software": ["CRM", "ORCL", "ADBE", "SAP", "NOW", "PANW", "SNOW", "DDOG"],
    "cloud": ["CRM", "ORCL", "SNOW", "DDOG", "NOW", "AMZN", "MSFT", "GOOGL"],
    "faang": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
    "mag7": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"],
    "magnificent seven": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"],
    "index": ["SPY", "VOO", "QQQ", "VTI", "IWM", "DIA"],
    "etf": ["SPY", "VOO", "QQQ", "VTI", "IWM", "DIA"],
    "equity etf": ["SPY", "VOO", "QQQ", "VTI", "IWM", "DIA"],
    "international": ["VXUS", "EFA", "EEM"],
    "emerging": ["EEM"],
    "developed": ["EFA", "VXUS"],
    "bond": ["TLT", "SHY", "LQD", "HYG", "BND", "AGG"],
    "fixed income": ["TLT", "SHY", "LQD", "HYG", "BND", "AGG"],
    "safe": ["SPY", "VOO", "TLT", "GLD", "BND", "AGG"],
    "defensive": ["TLT", "GLD", "XLU", "XLP", "XLV", "VNQ", "SHY"],
    "diversif": ["GLD", "DBC", "VNQ", "TLT", "VXUS", "EEM"],
    "finance": ["JPM", "BAC", "GS", "MS", "V", "MA"],
    "bank": ["JPM", "BAC", "GS", "MS"],
    "financial": ["JPM", "BAC", "GS", "MS", "V", "MA", "XLF"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK", "XLV"],
    "pharma": ["PFE", "ABBV", "LLY", "MRK", "JNJ"],
    "energy": ["XOM", "CVX", "COP", "SLB", "OXY", "XLE"],
    "oil": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "consumer": ["WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS"],
    "retail": ["WMT", "COST", "HD", "NKE"],
    "sector etf": ["XLF", "XLE", "XLK", "XLI", "XLC", "XLU", "XLP", "XLV", "XLRE"],
    "all weather": ["SPY", "TLT", "GLD", "DBC", "VTI"],
    "balanced portfolio": ["VTI", "VXUS", "BND", "GLD", "VNQ"],
}

RISK_KEYWORDS = {
    "conservative": {"position_pct": 15, "max_exposure_pct": 60, "vol_target_pct": 12},
    "safe": {"position_pct": 15, "max_exposure_pct": 60, "vol_target_pct": 12},
    "low risk": {"position_pct": 15, "max_exposure_pct": 60, "vol_target_pct": 12},
    "cautious": {"position_pct": 15, "max_exposure_pct": 60, "vol_target_pct": 12},
    "moderate": {"position_pct": 30, "max_exposure_pct": 100, "vol_target_pct": 20},
    "balanced": {"position_pct": 30, "max_exposure_pct": 100, "vol_target_pct": 20},
    "aggressive": {"position_pct": 45, "max_exposure_pct": 150, "vol_target_pct": 35},
    "high risk": {"position_pct": 45, "max_exposure_pct": 150, "vol_target_pct": 35},
    "yolo": {"position_pct": 50, "max_exposure_pct": 200, "vol_target_pct": 50},
}


def _extract_tickers_from_text(text: str) -> List[str]:
    """Pull explicit tickers ($AAPL, AAPL) and name/sector aliases."""
    tickers = set()
    upper = text.upper()

    # Explicit tickers: $AAPL or standalone AAPL (2-5 uppercase letters)
    for match in re.findall(r'\$?([A-Z]{2,5})\b', upper):
        tickers.add(match)

    lower = text.lower()

    # Company name aliases
    for alias, ticker in TICKER_ALIASES.items():
        if alias in lower:
            tickers.add(ticker)

    # Sector groups
    for keyword, group in SECTOR_TICKERS.items():
        if keyword in lower:
            tickers.update(group)

    # Filter to only known tickers
    known = set(TICKER_ALIASES.values()) | set(
        t for group in SECTOR_TICKERS.values() for t in group
    ) | {
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "AMD",
        "CRM", "ORCL", "ADBE", "INTC", "TSM", "QCOM", "IBM", "SAP",
        "NOW", "PANW", "NFLX", "TSLA", "MU", "MRVL", "SNOW", "DDOG",
        "SPY", "VOO", "QQQ", "VTI", "IWM", "DIA",
        "VXUS", "EFA", "EEM",
        "TLT", "SHY", "LQD", "HYG", "BND", "AGG",
        "GLD", "DBC", "VNQ", "XLU", "XLP", "XLV",
        "XLF", "XLE", "XLK", "XLI", "XLC", "XLRE",
        "JPM", "BAC", "GS", "MS", "V", "MA",
        "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK",
        "XOM", "CVX", "COP", "SLB", "OXY",
        "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS",
        "SNAP", "PINS",
    }
    return sorted(tickers & known)


def _extract_risk_level(text: str) -> Dict[str, int]:
    lower = text.lower()
    for keyword, params in RISK_KEYWORDS.items():
        if keyword in lower:
            return dict(params)
    return {"position_pct": 30, "max_exposure_pct": 100, "vol_target_pct": 20}


def _extract_numbers(text: str) -> Dict[str, Any]:
    """Try to pull explicit numeric params from text."""
    result: Dict[str, Any] = {}

    # Fast/slow window: "20 day fast" or "fast 20" or "20/100 crossover"
    crossover = re.search(r'(\d+)\s*/\s*(\d+)\s*(crossover|sma|moving average)', text.lower())
    if crossover:
        result["fast_window"] = int(crossover.group(1))
        result["slow_window"] = int(crossover.group(2))

    fast_match = re.search(r'fast\s*(?:sma|ma|window|period)?\s*(?:of\s*)?(\d+)', text.lower())
    if fast_match:
        result["fast_window"] = int(fast_match.group(1))

    slow_match = re.search(r'slow\s*(?:sma|ma|window|period)?\s*(?:of\s*)?(\d+)', text.lower())
    if slow_match:
        result["slow_window"] = int(slow_match.group(1))

    # Position size: "30% position" or "position size 30"
    pos_match = re.search(r'(\d+)\s*%?\s*position', text.lower())
    if pos_match:
        result["position_pct"] = int(pos_match.group(1))

    return result


def parse_strategy_text(
    text: str,
    *,
    openai_api_key: Optional[str] = None,
) -> Tuple[StrategyConfig, str]:
    """
    Parse natural language into a StrategyConfig.

    Returns (config, explanation) where explanation describes
    what was understood.
    """
    key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass

    if key:
        try:
            return _parse_with_llm(text, key)
        except Exception:
            pass

    return _parse_rule_based(text)


def _parse_rule_based(text: str) -> Tuple[StrategyConfig, str]:
    tickers = _extract_tickers_from_text(text)
    risk = _extract_risk_level(text)
    numbers = _extract_numbers(text)

    cfg = StrategyConfig()

    if tickers:
        cfg.symbols = tickers
    # else keep defaults

    cfg.position_fraction = float(numbers.get("position_pct", risk["position_pct"])) / 100.0
    cfg.max_portfolio_exposure = float(numbers.get("max_exposure_pct", risk["max_exposure_pct"])) / 100.0
    cfg.vol_target_annual = float(risk["vol_target_pct"]) / 100.0
    cfg.vol_target_enabled = True

    if "fast_window" in numbers:
        cfg.fast_window = numbers["fast_window"]
    if "slow_window" in numbers:
        cfg.slow_window = numbers["slow_window"]

    # ML filter
    if "no ml" in text.lower() or "without ai" in text.lower() or "no ai" in text.lower():
        cfg.ml_enabled = False

    explanation_parts = []
    explanation_parts.append(f"**Stocks**: {', '.join(cfg.symbols)}")
    explanation_parts.append(f"**Risk level**: position size {cfg.position_fraction*100:.0f}%, "
                             f"max exposure {cfg.max_portfolio_exposure*100:.0f}%")
    explanation_parts.append(f"**Strategy**: {cfg.fast_window}/{cfg.slow_window} day moving average crossover")
    explanation_parts.append(f"**AI filter**: {'enabled' if cfg.ml_enabled else 'disabled'}")

    explanation = "Here's what I understood from your description:\n\n" + "\n\n".join(explanation_parts)

    return cfg, explanation


def _parse_with_llm(text: str, api_key: str) -> Tuple[StrategyConfig, str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    system = """You are a trading strategy interpreter. Given a natural language description,
extract these parameters as JSON:
{
  "symbols": ["AAPL", ...],
  "fast_window": 20,
  "slow_window": 100,
  "position_fraction": 0.30,
  "max_portfolio_exposure": 1.0,
  "vol_target_annual": 0.20,
  "ml_enabled": true,
  "explanation": "Plain English summary of what you understood"
}

Rules:
- symbols: extract stock tickers or infer from sector mentions
- If user says "conservative" → small positions (0.15), low exposure (0.60)
- If user says "aggressive" → large positions (0.45), high exposure (1.50)
- Default to balanced if unclear
- Always include an explanation field
- Return ONLY valid JSON, no markdown"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    raw = response.choices[0].message.content or "{}"
    # Strip markdown fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw.strip())

    parsed = json.loads(raw)
    cfg = StrategyConfig()

    if parsed.get("symbols"):
        cfg.symbols = list(parsed["symbols"])
    if parsed.get("fast_window"):
        cfg.fast_window = int(parsed["fast_window"])
    if parsed.get("slow_window"):
        cfg.slow_window = int(parsed["slow_window"])
    if parsed.get("position_fraction") is not None:
        cfg.position_fraction = float(parsed["position_fraction"])
    if parsed.get("max_portfolio_exposure") is not None:
        cfg.max_portfolio_exposure = float(parsed["max_portfolio_exposure"])
    if parsed.get("vol_target_annual") is not None:
        cfg.vol_target_annual = float(parsed["vol_target_annual"])
        cfg.vol_target_enabled = True
    if parsed.get("ml_enabled") is not None:
        cfg.ml_enabled = bool(parsed["ml_enabled"])

    explanation = parsed.get("explanation", "Strategy parsed successfully.")
    return cfg, explanation
