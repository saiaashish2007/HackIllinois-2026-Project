"""
Streamlit dashboard — beginner-friendly strategy audit.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "logs" / "audit"
UPLOAD_AUDIT_DIR = PROJECT_ROOT / "logs" / "audit_upload"

POPULAR_STOCKS = {
    "Big Tech": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"],
    "Magnificent 7": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "INTC", "TSM", "QCOM", "MU", "MRVL"],
    "Software & Cloud": ["CRM", "ORCL", "ADBE", "SAP", "NOW", "PANW", "SNOW", "DDOG"],
    "Core Equity ETFs": ["SPY", "VOO", "QQQ", "VTI", "IWM", "DIA"],
    "International": ["VXUS", "EFA", "EEM"],
    "Bonds": ["TLT", "SHY", "LQD", "HYG", "BND", "AGG"],
    "Defensives & Diversifiers": ["GLD", "DBC", "VNQ", "XLU", "XLP", "XLV"],
    "Sector ETFs": ["XLF", "XLE", "XLK", "XLI", "XLC", "XLRE"],
    "Finance & Banking": ["JPM", "BAC", "GS", "MS", "V", "MA"],
    "Healthcare & Pharma": ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "Consumer": ["WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS"],
    "Streaming & Social": ["NFLX", "META", "GOOGL", "SNAP", "PINS"],
    "Custom": [],
}

ALL_SYMBOLS = sorted(set(
    s for group in POPULAR_STOCKS.values() for s in group
) | {
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "AMD",
    "CRM", "ORCL", "ADBE", "INTC", "TSM", "QCOM", "IBM", "SAP",
    "NOW", "PANW", "NFLX", "TSLA", "MU", "MRVL", "SNOW", "DDOG",
    # Core equity ETFs
    "SPY", "VOO", "QQQ", "VTI", "IWM", "DIA",
    # International
    "VXUS", "EFA", "EEM",
    # Bonds
    "TLT", "SHY", "LQD", "HYG", "BND", "AGG",
    # Defensives & diversifiers
    "GLD", "DBC", "VNQ", "XLU", "XLP", "XLV",
    # Sectors
    "XLF", "XLE", "XLK", "XLI", "XLC", "XLRE",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY",
    # Consumer
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "DIS",
    # Social / streaming
    "SNAP", "PINS",
})

PRESETS = {
    "Conservative": {
        "desc": "Lower risk, smaller positions, tighter controls. Good for beginners.",
        "fast_window": 20, "slow_window": 100, "vol_window": 20,
        "position_pct": 15, "max_exposure_pct": 60, "vol_target_pct": 12,
        "slippage": 1.0, "commission": 0.5, "ml": True,
    },
    "Balanced": {
        "desc": "Moderate risk and position sizing. A solid middle ground.",
        "fast_window": 20, "slow_window": 100, "vol_window": 20,
        "position_pct": 30, "max_exposure_pct": 100, "vol_target_pct": 20,
        "slippage": 1.0, "commission": 0.5, "ml": True,
    },
    "Aggressive": {
        "desc": "Higher risk, larger positions, wider exposure. For experienced traders.",
        "fast_window": 15, "slow_window": 80, "vol_window": 20,
        "position_pct": 45, "max_exposure_pct": 150, "vol_target_pct": 35,
        "slippage": 1.0, "commission": 0.5, "ml": True,
    },
}

GRADE_COLORS = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
GRADE_EMOJI = {"A": "Excellent", "B": "Good", "C": "Fair", "D": "Weak", "F": "Poor"}

FRIENDLY_METRIC_NAMES = {
    "Sharpe": ("Risk-Adjusted Return", "How much return you get per unit of risk. Higher is better. Above 1.0 is good, above 2.0 is great."),
    "Cagr": ("Annual Growth Rate", "How much your portfolio grows per year on average, accounting for compounding."),
    "Annual Volume": ("Yearly Volatility", "How much your portfolio value swings up and down. Lower means smoother ride."),
    "Max Drawdown": ("Worst Drop", "The biggest peak-to-trough decline. e.g. -0.15 means it once dropped 15% from its high."),
    "Total Return": ("Total Return", "Your overall gain or loss over the entire period."),
    "Observations": ("Trading Days", "Number of trading days in the backtest."),
}

FRIENDLY_COMPONENT_NAMES = {
    "regime_consistency": ("Works in All Markets", "Does the strategy hold up in bull markets, bear markets, and crashes?"),
    "param_stability": ("Not Overfit", "Would the strategy still work if you slightly changed its settings?"),
    "cost_resilience": ("Survives Real Costs", "Does it still make money after trading fees and slippage?"),
    "drawdown_penalty": ("Manageable Losses", "Are the worst drops survivable, or would they wipe you out?"),
    "base_performance": ("Raw Performance", "How well does the strategy perform under normal conditions?"),
}


# ── Page setup ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Strategy Shield AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* hide default sidebar toggle */
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    /* hero title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #888;
        margin-top: 2px;
        margin-bottom: 0.5rem;
    }

    /* top nav — spaced-out bold text links */
    .top-nav-radio [role="radiogroup"] {
        gap: 15.0rem !important;
        justify-content: center;
        width: 100%;
        padding: 0.6rem 0;
    }
    .top-nav-radio [data-baseweb="radio"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    /* hide radio circles */
    .top-nav-radio [data-baseweb="radio"] > div:first-child { display: none !important; }
    .top-nav-radio label {
        display: block;
        padding: 0.6rem 0.2rem !important;
        font-weight: 700;
        font-size: 1.15rem;
        color: #555;
        cursor: pointer;
        transition: color 0.15s, border-bottom 0.15s;
        border-bottom: 3px solid transparent;
        white-space: nowrap;
        text-align: center;
    }
    .top-nav-radio label:hover {
        color: #667eea;
    }
    .top-nav-radio label[data-checked="true"],
    .top-nav-radio [aria-checked="true"] + label,
    .top-nav-radio label:has(input:checked) {
        color: #667eea !important;
        -webkit-text-fill-color: #667eea;
        border-bottom-color: #667eea;
        font-weight: 800;
    }

    /* score display */
    .big-score {
        font-size: 3.2rem;
        font-weight: 900;
        text-align: center;
        line-height: 1.1;
    }
    .grade-label {
        font-size: 1.1rem;
        text-align: center;
        color: #888;
        font-weight: 500;
    }
    .help-text {
        font-size: 0.83rem;
        color: #999;
        margin-top: -8px;
    }

    /* section cards */
    .section-card {
        background: #fafbfe;
        border: 1px solid #e8ecf4;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    /* metric polish */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* divider softer */
    hr { border-color: #e8ecf4 !important; }

    /* buttons */
    .stButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.65rem 1.5rem;
    }

    /* tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Top navigation bar (above title, like a website header) ───────
PAGES = [
    "Test Strategy",
    "Describe Strategy",
    "What-If",
    "My Portfolio",
    "Past Results",
]
PAGE_KEY_MAP = {
    "Test Strategy": "Test a New Strategy",
    "Describe Strategy": "Describe Your Strategy",
    "What-If": "What-If Simulator",
    "My Portfolio": "Check My Portfolio",
    "Past Results": "View Past Results",
}

st.markdown('<div class="top-nav-radio">', unsafe_allow_html=True)
_nav_selection = st.radio(
    "nav",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

mode = PAGE_KEY_MAP.get(_nav_selection, "Test a New Strategy")

# ── Hero header (below nav) ──────────────────────────────────────
st.markdown('<div class="hero-title">Strategy Shield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Find out if your trading strategy is built to last — before you risk real money.</div>', unsafe_allow_html=True)
with st.expander("⏱️ App loading slowly?", expanded=False):
    st.caption(
        "Free-hosted apps sleep when idle. The first visit can take 30–60 seconds to wake up. "
        "If the page is blank, wait a moment or refresh once — then it should be snappy."
    )


# =====================================================================
# Helper: render results in beginner-friendly way
# =====================================================================

def _pct(val, fallback="N/A"):
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return fallback


def _fmt(val, decimals=2, fallback="N/A"):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def _dollar(val, fallback="N/A"):
    try:
        v = float(val)
        if abs(v) >= 1_000_000:
            return f"${v:,.0f}"
        return f"${v:,.2f}"
    except (TypeError, ValueError):
        return fallback


def render_audit(audit: dict, prev_audit: dict = None) -> None:
    surv = audit.get("survivability", {})
    score = surv.get("score", 0)
    grade = surv.get("grade", "?")
    base = audit.get("base_metrics", {})
    is_upload = audit.get("mode") == "uploaded_portfolio"
    capital = audit.get("initial_capital", 1.0)

    # Previous run data for delta comparison
    prev_score = None
    prev_grade = None
    if prev_audit:
        prev_surv = prev_audit.get("survivability", {})
        prev_score = prev_surv.get("score")
        prev_grade = prev_surv.get("grade")

    # ── Big score hero ────────────────────────────────────────
    st.markdown("---")
    color = GRADE_COLORS.get(grade, "#888")

    # Build delta HTML if we have a previous run
    delta_html = ""
    if prev_score is not None and float(prev_score) != float(score):
        delta = round(float(score) - float(prev_score), 1)
        delta_sign = "+" if delta > 0 else ""
        delta_color = "#2ecc71" if delta > 0 else "#e74c3c"
        arrow = "▲" if delta > 0 else "▼"
        prev_color = GRADE_COLORS.get(prev_grade, "#888")
        delta_html = f"""
        <div style="margin-top:0.8rem;padding:0.6rem 1.2rem;background:rgba(0,0,0,0.03);
                    border-radius:10px;display:inline-block;">
            <span style="font-size:0.85rem;color:#888;">Previous run:</span>
            <span style="font-size:1rem;font-weight:700;color:{prev_color};margin:0 0.3rem;">{prev_score}/100 ({prev_grade})</span>
            <span style="font-size:1rem;font-weight:800;color:{delta_color};margin-left:0.5rem;">
                {arrow} {delta_sign}{delta} points
            </span>
        </div>
        """

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f8f9fc,#eef1f8);border:1px solid #e0e4ee;
                border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.2rem;text-align:center;">
        <div style="display:flex;justify-content:center;align-items:center;gap:2rem;flex-wrap:wrap;">
            <div>
                <div class="big-score" style="color:{color}">{score}<span style="font-size:1.4rem;color:#aaa">/100</span></div>
                <div class="grade-label">Survivability Score</div>
            </div>
            <div>
                <div class="big-score" style="color:{color}">{grade}</div>
                <div class="grade-label">{GRADE_EMOJI.get(grade, "")}</div>
            </div>
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

    if score >= 80:
        st.success("This strategy looks robust! It holds up across different market conditions and parameter changes.")
    elif score >= 65:
        st.info("Decent strategy with some areas to watch. Review the details below for specific concerns.")
    elif score >= 50:
        st.warning("This strategy has notable weaknesses. It may struggle in certain market environments.")
    else:
        st.error("High risk of failure. This strategy is fragile and may not survive real-world trading conditions.")

    # ── Score comparison breakdown (if previous run exists) ────
    if prev_audit and prev_score is not None:
        prev_base = prev_audit.get("base_metrics", {})
        prev_components = prev_audit.get("survivability", {}).get("components", {})
        curr_components = surv.get("components", {})

        with st.expander("Compare with your previous run", expanded=(float(prev_score) != float(score))):
            cmp1, cmp2, cmp3 = st.columns(3)
            cmp1.metric(
                "Previous Score", f"{prev_score}/100",
                help="Your last audit's survivability score",
            )
            cmp2.metric(
                "Current Score", f"{score}/100",
                help="This audit's survivability score",
            )
            delta_val = round(float(score) - float(prev_score), 1)
            cmp3.metric(
                "Change", f"{'+' if delta_val > 0 else ''}{delta_val} pts",
                delta=f"{'+' if delta_val > 0 else ''}{delta_val}",
                delta_color="normal" if delta_val >= 0 else "inverse",
            )

            st.markdown("**Component-by-component:**")
            comp_rows = []
            for key in ["regime_consistency", "param_stability", "cost_resilience", "drawdown_penalty", "base_performance"]:
                friendly = FRIENDLY_COMPONENT_NAMES.get(key, (key, ""))[0]
                old_val = prev_components.get(key, 0)
                new_val = curr_components.get(key, 0)
                try:
                    old_pct = int(float(old_val) * 100)
                    new_pct = int(float(new_val) * 100)
                    diff = new_pct - old_pct
                    diff_str = f"+{diff}%" if diff > 0 else f"{diff}%"
                    emoji = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
                except (TypeError, ValueError):
                    old_pct, new_pct, diff_str, emoji = "?", "?", "—", "⚪"
                comp_rows.append({
                    "Component": friendly,
                    "Previous": f"{old_pct}%",
                    "Current": f"{new_pct}%",
                    "Change": f"{emoji} {diff_str}",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            # Key metric deltas
            st.markdown("**Key metrics:**")
            km1, km2, km3, km4 = st.columns(4)
            lower_is_better = {"max_drawdown", "annual_vol"}
            for col, label, key in [
                (km1, "Sharpe", "sharpe"), (km2, "Annual Growth", "cagr"),
                (km3, "Max Drop", "max_drawdown"), (km4, "Volatility", "annual_vol"),
            ]:
                old_v = prev_base.get(key)
                new_v = base.get(key)
                try:
                    diff = float(new_v) - float(old_v)
                    good = (diff <= 0) if key in lower_is_better else (diff >= 0)
                    if key in ("cagr", "max_drawdown", "annual_vol"):
                        col.metric(label, _pct(new_v), delta=f"{diff*100:+.1f}%",
                                   delta_color="normal" if good else "inverse")
                    else:
                        col.metric(label, _fmt(new_v), delta=f"{diff:+.2f}",
                                   delta_color="normal" if good else "inverse")
                except (TypeError, ValueError):
                    col.metric(label, _fmt(new_v) if key == "sharpe" else _pct(new_v))

            # Recovery metric deltas
            st.markdown("**Recovery:**")
            rc1, rc2, rc3 = st.columns(3)
            for col, label, key, unit in [
                (rc1, "Avg Recovery", "avg_recovery_days", " days"),
                (rc2, "Worst Recovery", "worst_recovery_days", " days"),
                (rc3, "Time Underwater", "time_underwater_pct", ""),
            ]:
                old_v = prev_base.get(key)
                new_v = base.get(key)
                try:
                    o, n = float(old_v), float(new_v)
                    if np.isnan(o) or np.isnan(n):
                        raise ValueError
                    diff = n - o
                    if key == "time_underwater_pct":
                        col.metric(label, f"{n*100:.1f}%", delta=f"{diff*100:+.1f}%",
                                   delta_color="normal" if diff <= 0 else "inverse")
                    else:
                        col.metric(label, f"{n:.0f}{unit}", delta=f"{diff:+.0f}{unit}",
                                   delta_color="normal" if diff <= 0 else "inverse")
                except (TypeError, ValueError):
                    if key == "time_underwater_pct":
                        col.metric(label, f"{float(new_v)*100:.1f}%" if new_v else "N/A")
                    else:
                        col.metric(label, f"{float(new_v):.0f}{unit}" if new_v else "N/A")

    # ── Key numbers (plain English) ───────────────────────────
    st.markdown("")
    st.markdown("#### At a Glance")

    # Compute dollar P&L from total return
    total_ret = base.get("total_return")
    max_dd = base.get("max_drawdown")
    has_real_capital = capital > 1.5

    if has_real_capital:
        m0, m1, m2, m3, m4 = st.columns(5)
        m0.metric(
            "Starting Capital",
            _dollar(capital),
            help="The amount you said you'd invest.",
        )
    else:
        m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Annual Growth",
        _pct(base.get("cagr")),
        help="How much your money grows each year on average",
    )
    m2.metric(
        "Risk-Adjusted Return",
        _fmt(base.get("sharpe")),
        help="Return per unit of risk (Sharpe ratio). Above 1.0 is good.",
    )
    m3.metric(
        "Worst Drop",
        _pct(max_dd),
        help="The biggest decline from a peak. Smaller (closer to 0%) is better.",
    )
    m4.metric(
        "Volatility",
        _pct(base.get("annual_vol")),
        help="How much your portfolio swings. Lower = smoother ride.",
    )

    # Dollar impact row
    if has_real_capital:
        st.markdown("#### Your Money")
        d1, d2, d3 = st.columns(3)

        try:
            end_value = capital * (1 + float(total_ret)) if total_ret is not None else None
        except (TypeError, ValueError):
            end_value = None

        try:
            dollar_pnl = float(total_ret) * capital if total_ret is not None else None
        except (TypeError, ValueError):
            dollar_pnl = None

        try:
            worst_loss = float(max_dd) * capital if max_dd is not None else None
        except (TypeError, ValueError):
            worst_loss = None

        d1.metric(
            "Final Portfolio Value",
            _dollar(end_value) if end_value is not None else "N/A",
            delta=f"{'+' if dollar_pnl and dollar_pnl >= 0 else ''}{_dollar(dollar_pnl)}" if dollar_pnl is not None else None,
            help="What your portfolio would be worth at the end of the backtest period.",
        )
        d2.metric(
            "Total Profit / Loss",
            _dollar(dollar_pnl) if dollar_pnl is not None else "N/A",
            help="How much money you would have made or lost in dollar terms.",
        )
        d3.metric(
            "Worst Possible Loss",
            _dollar(worst_loss) if worst_loss is not None else "N/A",
            help="The most you could have lost from a peak — the scariest moment.",
        )

    # ── Drawdown & Recovery ──────────────────────────────────
    avg_rec = base.get("avg_recovery_days")
    worst_rec = base.get("worst_recovery_days")
    pct_underwater = base.get("time_underwater_pct")

    has_recovery = any(
        v is not None and not (isinstance(v, float) and np.isnan(v))
        for v in [avg_rec, worst_rec, pct_underwater]
    )

    if has_recovery:
        st.markdown("#### Drawdown & Recovery")
        st.markdown("How quickly does the strategy bounce back after losses?")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(
            "Worst Drop",
            _pct(max_dd),
            help="The deepest peak-to-trough decline.",
        )
        try:
            avg_rec_f = float(avg_rec)
            avg_txt = f"{avg_rec_f:.0f} days" if not np.isnan(avg_rec_f) else "N/A"
        except (TypeError, ValueError):
            avg_txt = "N/A"
        r2.metric(
            "Avg Recovery",
            avg_txt,
            help="On average, how many trading days it takes to climb back to the previous high.",
        )
        try:
            worst_rec_f = float(worst_rec)
            worst_txt = f"{worst_rec_f:.0f} days" if not np.isnan(worst_rec_f) else "N/A"
        except (TypeError, ValueError):
            worst_txt = "N/A"
        r3.metric(
            "Worst Recovery",
            worst_txt,
            help="The longest time it ever took to recover from a drawdown.",
        )
        try:
            uw_pct = float(pct_underwater) * 100
            uw_txt = f"{uw_pct:.1f}%" if not np.isnan(uw_pct) else "N/A"
        except (TypeError, ValueError):
            uw_txt = "N/A"
        r4.metric(
            "Time Underwater",
            uw_txt,
            help="Percentage of all trading days where the portfolio was below its previous peak.",
        )

        # Color-coded verdict
        try:
            uw_val = float(pct_underwater)
        except (TypeError, ValueError):
            uw_val = None

        if uw_val is not None and not np.isnan(uw_val):
            if uw_val < 0.30:
                st.success("Strong recovery profile — the strategy spends most of its time at or near highs.")
            elif uw_val < 0.55:
                st.info("Moderate recovery — some extended drawdown periods, but generally recovers.")
            elif uw_val < 0.75:
                st.warning("Slow recovery — the strategy spends a lot of time below its peak. Patience required.")
            else:
                st.error("Very slow recovery — the strategy is underwater most of the time. This is a red flag.")

    # ── Prepare equity data (used by multiple tabs) ────────────
    equity_data = audit.get("_equity_df")
    if equity_data is None:
        records = audit.get("base_equity", [])
        if records:
            equity_data = pd.DataFrame(records)

    pv = None
    if equity_data is not None and not equity_data.empty and "portfolio_value" in equity_data.columns:
        eq = equity_data.copy()
        if "date" in eq.columns:
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.set_index("date")
        pv = eq["portfolio_value"].astype(float)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    # ── Tabs ──────────────────────────────────────────────────
    gpu_ml = audit.get("gpu_ml", {})
    has_gpu_ml = gpu_ml.get("gpu_used", False) and "error" not in gpu_ml

    tab_names = ["Future Projection", "Market Conditions"]
    if not is_upload:
        tab_names += ["Sensitivity Check", "Stress Test"]
    if has_gpu_ml:
        tab_names += ["AI Model (GPU)"]
    tab_names += ["Detailed Report", "Score Breakdown", "Past Performance"]
    tabs = st.tabs(tab_names)
    ti = 0

    # ── FUTURE PROJECTION (primary tab) ────────────────────────
    with tabs[ti]:
        if pv is not None and len(pv) >= 10:
            end_val = float(pv.iloc[-1])
            end_dt = pv.index[-1]
            daily_rets = pv.pct_change().dropna().values

            # Projection params from the form (or defaults)
            proj_months = audit.get("_projection_months", 12)
            proj_days = int(proj_months * 21)
            n_sims = 2000

            st.markdown(f"""
            <div style="text-align:center;padding:1.2rem 0 0.4rem 0;">
                <div style="font-size:1.6rem;font-weight:800;color:#1a1a2e;letter-spacing:-0.5px;">
                    What Could Your {_dollar(capital)} Become?
                </div>
                <div style="font-size:1rem;color:#666;margin-top:0.3rem;">
                    {proj_months}-month forward projection based on {len(daily_rets)} days of strategy data
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Monte Carlo simulation
            rng = np.random.default_rng(42)
            sampled = rng.choice(daily_rets, size=(n_sims, proj_days), replace=True)
            cum_paths = end_val * np.cumprod(1 + sampled, axis=1)
            p5 = np.percentile(cum_paths, 5, axis=0)
            p10 = np.percentile(cum_paths, 10, axis=0)
            p25 = np.percentile(cum_paths, 25, axis=0)
            p50 = np.percentile(cum_paths, 50, axis=0)
            p75 = np.percentile(cum_paths, 75, axis=0)
            p90 = np.percentile(cum_paths, 90, axis=0)
            p95 = np.percentile(cum_paths, 95, axis=0)
            prob_profit = float((cum_paths[:, -1] >= end_val).sum()) / n_sims * 100
            prob_2x = float((cum_paths[:, -1] >= end_val * 2).sum()) / n_sims * 100
            prob_loss_20 = float((cum_paths[:, -1] <= end_val * 0.8).sum()) / n_sims * 100

            future_dates = pd.bdate_range(start=end_dt + pd.Timedelta(days=1), periods=proj_days)
            final_median = float(p50[-1])
            final_p5 = float(p5[-1])
            final_p25 = float(p25[-1])
            final_p75 = float(p75[-1])
            final_p95 = float(p95[-1])
            med_change = (final_median / end_val - 1) * 100

            # ── Probability hero card ───────────────────────────
            prob_color = "#2ecc71" if prob_profit >= 65 else "#f39c12" if prob_profit >= 45 else "#e74c3c"
            prob_emoji = "Strong odds" if prob_profit >= 65 else "Coin-flip territory" if prob_profit >= 45 else "Risky"
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f0fdf4,#ecfdf5);border:2px solid {prob_color}33;
                        border-radius:16px;padding:1.5rem 2rem;margin:1rem 0;text-align:center;">
                <div style="display:flex;justify-content:center;align-items:center;gap:3rem;flex-wrap:wrap;">
                    <div>
                        <div style="font-size:3rem;font-weight:900;color:{prob_color};line-height:1;">
                            {prob_profit:.0f}%
                        </div>
                        <div style="font-size:0.9rem;color:#666;font-weight:600;margin-top:0.3rem;">
                            Chance of Profit
                        </div>
                    </div>
                    <div style="text-align:left;">
                        <div style="font-size:1.1rem;font-weight:700;color:#333;">
                            {prob_emoji} over {proj_months} months
                        </div>
                        <div style="font-size:0.85rem;color:#888;margin-top:0.25rem;">
                            {n_sims:,} simulated paths &bull; Median outcome: <b style="color:{prob_color}">{_dollar(final_median)}</b>
                            ({'+' if med_change >= 0 else ''}{med_change:.1f}%)
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Projection chart (hero) ─────────────────────────
            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor("#fafbfe")
            ax.set_facecolor("#fafbfe")

            dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")
            ax.yaxis.set_major_formatter(dollar_fmt)

            # Historical tail (faded context)
            hist_tail = pv.iloc[-min(90, len(pv)):]
            ax.plot(hist_tail.index, hist_tail.values, color="#667eea",
                    linewidth=2, alpha=0.5, label="Historical")

            # Confidence bands (layered from widest to narrowest)
            ax.fill_between(future_dates, p5, p95, alpha=0.07, color="#e74c3c", linewidth=0)
            ax.fill_between(future_dates, p10, p90, alpha=0.10, color="#667eea", linewidth=0)
            ax.fill_between(future_dates, p25, p75, alpha=0.18, color="#667eea", linewidth=0, label="50% of outcomes")

            # Median path
            ax.plot(future_dates, p50, color="#2ecc71", linewidth=3, label="Expected (median)", zorder=4)

            # Percentile edge lines
            ax.plot(future_dates, p5, color="#e74c3c", linewidth=1, alpha=0.4, linestyle=":")
            ax.plot(future_dates, p95, color="#2ecc71", linewidth=1, alpha=0.4, linestyle=":")

            # Today divider
            ax.axvline(end_dt, color="#667eea", linestyle="--", linewidth=1.5, alpha=0.5)
            ylim = ax.get_ylim()
            ax.text(end_dt, ylim[1] - (ylim[1] - ylim[0]) * 0.02, "  Today  ",
                    fontsize=10, fontweight="bold", color="#667eea", va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#667eea", alpha=0.85))

            # Starting value line
            ax.axhline(end_val, color="#aaa", linestyle=":", linewidth=1, alpha=0.4)

            # End annotations
            for label_text, val, color, y_off in [
                ("Best case", final_p95, "#27ae60", 30),
                ("Expected", final_median, "#2ecc71", 0),
                ("Worst case", final_p5, "#e74c3c", -30),
            ]:
                pct_chg = (val / end_val - 1) * 100
                sign_str = "+" if pct_chg >= 0 else ""
                ax.scatter([future_dates[-1]], [val], color=color, s=50, zorder=5, edgecolors="white", linewidth=1.5)
                ax.annotate(
                    f"{label_text}\n{_dollar(val)} ({sign_str}{pct_chg:.1f}%)",
                    xy=(future_dates[-1], val),
                    xytext=(12, y_off), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.92, lw=1.5),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1, alpha=0.5),
                )

            ax.set_ylabel("Portfolio Value", fontsize=12, fontweight="700", color="#333")
            ax.set_xlabel("")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
            fig.autofmt_xdate(rotation=0, ha="center")
            ax.tick_params(colors="#666", labelsize=10)
            ax.grid(True, alpha=0.10, color="#ccc")
            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#ddd",
                      fancybox=True, borderpad=0.8)
            fig.tight_layout(pad=1.5)

            st.pyplot(fig)
            plt.close(fig)

            # ── Outcome cards ───────────────────────────────────
            st.markdown(f"""
            <div style="text-align:center;padding:0.8rem 0 0.2rem 0;">
                <div style="font-size:1.2rem;font-weight:800;color:#333;letter-spacing:-0.3px;">
                    Range of Outcomes After {proj_months} Months
                </div>
                <div style="font-size:0.8rem;color:#999;">Starting from {_dollar(end_val)} today</div>
            </div>
            """, unsafe_allow_html=True)

            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            for col, label, val, icon in [
                (oc1, "Worst Case", final_p5, "5th"),
                (oc2, "Pessimistic", final_p25, "25th"),
                (oc3, "Expected", final_median, "50th"),
                (oc4, "Optimistic", final_p75, "75th"),
                (oc5, "Best Case", final_p95, "95th"),
            ]:
                pct_chg = (val / end_val - 1) * 100
                sign_str = "+" if pct_chg >= 0 else ""
                col.metric(
                    f"{label} ({icon})",
                    _dollar(val),
                    delta=f"{sign_str}{pct_chg:.1f}%",
                    delta_color="normal" if pct_chg >= 0 else "inverse",
                )

            # ── Probability details ─────────────────────────────
            st.markdown("")
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Chance of Profit", f"{prob_profit:.0f}%",
                       help="% of simulations that end with more than you started.")
            pc2.metric("Chance of Doubling", f"{prob_2x:.1f}%",
                       help="% of simulations where your money doubles.")
            pc3.metric("Chance of 20%+ Loss", f"{prob_loss_20:.1f}%",
                       help="% of simulations where you lose more than 20%.",
                       delta=f"{prob_loss_20:.1f}%", delta_color="inverse" if prob_loss_20 > 10 else "off")

            # Verdict
            if prob_profit >= 70:
                st.success(f"**{prob_profit:.0f}%** of simulated paths ended in profit over {proj_months} months. This strategy has historically strong forward-looking odds.")
            elif prob_profit >= 50:
                st.info(f"**{prob_profit:.0f}%** of simulated paths ended in profit — roughly a coin flip. Consider adjusting parameters or a longer horizon.")
            elif prob_profit >= 30:
                st.warning(f"Only **{prob_profit:.0f}%** of paths ended in profit. The odds lean against this strategy over {proj_months} months.")
            else:
                st.error(f"Only **{prob_profit:.0f}%** of paths ended in profit. High likelihood of losses — this strategy may not be viable.")

            with st.expander("How does this projection work?"):
                st.markdown(
                    f"We take the **{len(daily_rets)} daily returns** from the historical backtest, "
                    f"randomly resample them **{proj_days} times** (one for each future trading day), "
                    f"and repeat this **{n_sims:,} times** to build a distribution of possible futures.\n\n"
                    "This is called **bootstrapped Monte Carlo simulation**. It assumes future returns "
                    "come from the same statistical distribution as past returns — which is a simplification. "
                    "Real markets can surprise, but this gives you a data-grounded sense of "
                    "the range of possibilities.\n\n"
                    "| Band | Meaning |\n"
                    "|:---|:---|\n"
                    "| **Green line** (median) | The 50/50 outcome — equally likely to be above or below |\n"
                    "| **Dark band** (25th–75th) | Where half of all outcomes land |\n"
                    "| **Light band** (5th–95th) | Where 90% of all outcomes land |\n"
                    "| **Dotted edges** | Extreme best/worst case boundaries |"
                )
        else:
            st.info("Not enough data to generate a future projection. Need at least 10 trading days.")
    ti += 1

    # ── Market conditions ─────────────────────────────────────
    with tabs[ti]:
        st.subheader("How Does It Perform in Different Markets?")
        st.markdown(
            "Markets go through different phases. A good strategy works in most of them — "
            "not just when everything is going up."
        )
        reg = audit.get("regime_metrics", {})
        if reg:
            friendly_regimes = {
                "bull": "Rising Market",
                "bear": "Falling Market",
                "high_vol": "Volatile / Choppy",
                "crisis": "Market Crash",
            }
            rows = []
            for regime, metrics in reg.items():
                sharpe = metrics.get("sharpe")
                ret = metrics.get("total_return")
                dd = metrics.get("max_drawdown")
                days = metrics.get("days", 0)
                status = "Profitable" if (sharpe is not None and sharpe > 0) else "Losing money"
                rows.append({
                    "Market Phase": friendly_regimes.get(regime, regime),
                    "Days": days,
                    "Return": _pct(ret),
                    "Risk-Adjusted": _fmt(sharpe),
                    "Worst Drop": _pct(dd),
                    "Verdict": status,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            neg_regimes = [r for r, m in reg.items() if m.get("sharpe") is not None and m["sharpe"] < 0]
            if neg_regimes:
                labels = [friendly_regimes.get(r, r) for r in neg_regimes]
                st.warning(f"The strategy loses money during: **{', '.join(labels)}**. This is a risk flag.")
            else:
                st.success("The strategy is profitable across all market phases tested.")
        else:
            st.info("No market condition data available.")
    ti += 1

    # ── Sensitivity (strategy only) ───────────────────────────
    if not is_upload:
        with tabs[ti]:
            st.subheader("Is the Strategy Overfit?")
            st.markdown(
                "We tested the strategy with **slightly different settings** to see if it still works. "
                "If small changes break the strategy, it might be overfit to past data."
            )
            psummary = audit.get("param_sweep_summary", {})
            if psummary and psummary.get("num_combos", 0) > 0:
                stability = psummary.get("stability_score", 0)
                combos = psummary.get("num_combos", 0)

                if stability >= 0.7:
                    st.success(f"Stability score: **{_fmt(stability)}** — The strategy is robust to setting changes. ({combos} variations tested)")
                elif stability >= 0.4:
                    st.warning(f"Stability score: **{_fmt(stability)}** — Moderately sensitive to settings. ({combos} variations tested)")
                else:
                    st.error(f"Stability score: **{_fmt(stability)}** — Highly sensitive! Small changes break performance. ({combos} variations tested)")

                sweep_data = audit.get("param_sweep", [])
                if sweep_data:
                    sweep_df = pd.DataFrame(sweep_data)
                    if "fast_window" in sweep_df.columns and "slow_window" in sweep_df.columns and "sharpe" in sweep_df.columns:
                        st.markdown("**Performance across different settings** (green = better):")
                        pivot = sweep_df.pivot_table(index="slow_window", columns="fast_window", values="sharpe", aggfunc="first")
                        st.dataframe(
                            pivot.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}", na_rep="—"),
                            use_container_width=True,
                        )
            else:
                st.info("No sensitivity data available.")
        ti += 1

        # ── Stress test ───────────────────────────────────────
        with tabs[ti]:
            st.subheader("Can It Handle Worst-Case Scenarios?")
            st.markdown(
                "We cranked up trading costs, removed filters, and simulated adverse conditions "
                "to see how the strategy degrades."
            )
            ssummary = audit.get("stress_summary", {})
            if ssummary and ssummary.get("num_scenarios", 0) > 0:
                cost_score = ssummary.get("cost_sensitivity_score")
                if cost_score is not None:
                    if cost_score >= 0.8:
                        st.success(f"Cost resilience: **{_fmt(cost_score)}** — Trading costs barely affect this strategy.")
                    elif cost_score >= 0.5:
                        st.warning(f"Cost resilience: **{_fmt(cost_score)}** — Costs eat into profits noticeably.")
                    else:
                        st.error(f"Cost resilience: **{_fmt(cost_score)}** — Profits mostly disappear under realistic costs.")

                stress_data = audit.get("stress_tests", [])
                if stress_data:
                    stress_df = pd.DataFrame(stress_data)
                    friendly_scenarios = {
                        "baseline": "Normal conditions",
                        "slippage_2.0x": "2x price slippage",
                        "slippage_5.0x": "5x price slippage",
                        "slippage_10.0x": "10x price slippage",
                        "commission_2.0x": "2x trading fees",
                        "commission_5.0x": "5x trading fees",
                        "commission_10.0x": "10x trading fees",
                        "worst_case_costs": "Worst-case costs",
                        "no_ml_filter": "Without AI filter",
                        "small_positions_10pct": "Tiny positions (10%)",
                    }
                    if "scenario" in stress_df.columns:
                        stress_df["Scenario"] = stress_df["scenario"].map(lambda s: friendly_scenarios.get(s, s))
                    display_cols = []
                    renames = {}
                    for orig, friendly in [("Scenario", "Scenario"), ("sharpe", "Risk-Adj Return"), ("total_return", "Total Return"), ("max_drawdown", "Worst Drop")]:
                        if orig in stress_df.columns:
                            display_cols.append(orig)
                            renames[orig] = friendly
                    if display_cols:
                        st.dataframe(stress_df[display_cols].rename(columns=renames), use_container_width=True, hide_index=True)
            else:
                st.info("No stress test data available.")
        ti += 1

    # ── GPU ML tab ─────────────────────────────────────────────
    if has_gpu_ml:
        with tabs[ti]:
            st.subheader("AI Model — Trained on GPU")
            st.markdown(
                "An XGBoost model was trained using "
                f"**{gpu_ml.get('num_features', '?')} features** across "
                f"**{gpu_ml.get('total_samples', gpu_ml.get('training_samples', '?')):,} data points**. "
                "It uses time-series walk-forward validation — no future data leaks into training."
            )

            # Key metrics row
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("GPU Device", gpu_ml.get("device", "N/A"))
            g2.metric("Training Samples", f"{gpu_ml.get('training_samples', 0):,}")
            holdout_n = gpu_ml.get("holdout_samples")
            if holdout_n:
                g3.metric("Holdout Samples", f"{holdout_n:,}")
            else:
                g3.metric("Total Samples", f"{gpu_ml.get('total_samples', gpu_ml.get('training_samples', 0)):,}")
            rounds = gpu_ml.get("num_boost_rounds", 0)
            max_rounds = gpu_ml.get("max_boost_rounds", rounds)
            g4.metric("Boosting Rounds", f"{rounds} / {max_rounds}",
                      help="Early stopping selected optimal round count from training.")

            # Holdout AUC (most important number)
            holdout_auc = gpu_ml.get("holdout_auc")
            cv_mean = gpu_ml.get("cv_auc_mean", 0)
            primary_auc = holdout_auc if holdout_auc is not None else cv_mean

            st.markdown("---")
            a1, a2 = st.columns(2)
            if holdout_auc is not None:
                a1.metric("Holdout AUC (out-of-sample)", f"{holdout_auc:.4f}",
                          help="Tested on the most recent 20% of data the model never saw during training.")
            if cv_mean:
                cv_std = gpu_ml.get("cv_auc_std", 0)
                a2.metric("Walk-Forward CV AUC", f"{cv_mean:.4f} +/- {cv_std:.4f}",
                          help="Average across expanding-window time-series folds.")

            # Interpretation
            if primary_auc >= 0.65:
                st.success(
                    f"**AUC = {primary_auc:.3f}** — The model has meaningful predictive signal. "
                    "It can distinguish profitable from unprofitable setups better than chance (0.50)."
                )
            elif primary_auc >= 0.55:
                st.info(
                    f"**AUC = {primary_auc:.3f}** — Modest predictive power. "
                    "The model finds some edge but shouldn't be the sole decision-maker."
                )
            else:
                st.warning(
                    f"**AUC = {primary_auc:.3f}** — Limited predictive power for these stocks/settings. "
                    "The model struggles to separate winners from losers."
                )

            # CV fold details
            cv_method = gpu_ml.get("cv_method", "5-fold cross-validation")
            cv_scores_list = gpu_ml.get("cv_scores", [])
            if cv_scores_list:
                with st.expander(f"Walk-forward CV details ({cv_method})"):
                    cv_df = pd.DataFrame({
                        "Fold": [f"Window {i+1}" for i in range(len(cv_scores_list))],
                        "AUC Score": cv_scores_list,
                    })
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    st.markdown(
                        "Each fold trains on all data *before* the test window — "
                        "no future information leaks into training. This is the gold standard "
                        "for evaluating time-series models."
                    )

            # Feature importance
            st.markdown("**What the model pays attention to** (feature importance):")
            importance = gpu_ml.get("feature_importance", {})
            if importance:
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in sorted(importance.items(), key=lambda x: -x[1])[:15]
                ])
                st.bar_chart(imp_df.set_index("Feature"))
        ti += 1

    # ── Detailed report ───────────────────────────────────────
    with tabs[ti]:
        st.subheader("Full Analysis Report")
        report_text = audit.get("report", "")
        if report_text:
            st.markdown(report_text)
        else:
            st.info("No detailed report available. Add an OpenAI API key for an AI-written analysis.")
    ti += 1

    # ── Score breakdown ───────────────────────────────────────
    with tabs[ti]:
        st.subheader("What Makes Up the Score?")
        st.markdown("The survivability score is built from 5 factors:")
        components = surv.get("components", {})
        if components:
            for key, val in components.items():
                friendly_name, description = FRIENDLY_COMPONENT_NAMES.get(key, (key, ""))
                pct_val = int(float(val) * 100) if val is not None else 0
                col_name, col_bar = st.columns([1, 3])
                col_name.markdown(f"**{friendly_name}**")
                col_bar.progress(min(pct_val, 100), text=f"{pct_val}%")
                if description:
                    st.markdown(f'<div class="help-text">{description}</div>', unsafe_allow_html=True)

        if is_upload:
            st.info(
                "\"Not Overfit\" and \"Survives Real Costs\" default to 50% for uploaded portfolios "
                "because we can't re-run your strategy with different settings."
            )
    ti += 1

    # ── Past Performance (last tab) ────────────────────────────
    with tabs[ti]:
        st.subheader("Historical Backtest")
        st.markdown(
            "This is how the strategy **actually performed** on real market data. "
            "The projection above is based on these historical returns."
        )

        if pv is not None and len(pv) >= 2:
            start_val = float(pv.iloc[0])
            end_val_hist = float(pv.iloc[-1])
            start_dt = pv.index[0]
            end_dt_hist = pv.index[-1]
            gained = end_val_hist >= start_val

            fig_hist, ax_hist = plt.subplots(figsize=(13, 4.5))
            fig_hist.patch.set_facecolor("#fafbfe")
            ax_hist.set_facecolor("#fafbfe")

            ax_hist.plot(pv.index, pv.values, color="#667eea", linewidth=2)
            ax_hist.fill_between(pv.index, pv.values, alpha=0.06, color="#667eea")

            dollar_fmt_hist = FuncFormatter(lambda x, _: f"${x:,.0f}")
            ax_hist.yaxis.set_major_formatter(dollar_fmt_hist)

            ax_hist.scatter([start_dt], [start_val], color="#667eea", s=80, zorder=5, edgecolors="white", linewidth=1.5)
            ax_hist.annotate(
                f"Start\n{_dollar(start_val)}",
                xy=(start_dt, start_val),
                xytext=(15, 22), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#667eea",
                arrowprops=dict(arrowstyle="->", color="#667eea", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#667eea", alpha=0.92),
            )

            end_color = "#2ecc71" if gained else "#e74c3c"
            ax_hist.scatter([end_dt_hist], [end_val_hist], color=end_color, s=80, zorder=5, edgecolors="white", linewidth=1.5)
            change_pct = (end_val_hist / start_val - 1) * 100 if start_val > 0 else 0
            sign_str = "+" if change_pct >= 0 else ""
            ax_hist.annotate(
                f"End\n{_dollar(end_val_hist)}\n({sign_str}{change_pct:.1f}%)",
                xy=(end_dt_hist, end_val_hist),
                xytext=(-15, 22), textcoords="offset points",
                fontsize=10, fontweight="bold", color=end_color, ha="right",
                arrowprops=dict(arrowstyle="->", color=end_color, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=end_color, alpha=0.92),
            )

            ax_hist.set_ylabel("Portfolio Value", fontsize=11, fontweight="700", color="#333")
            ax_hist.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax_hist.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
            fig_hist.autofmt_xdate(rotation=0, ha="center")
            ax_hist.tick_params(colors="#666", labelsize=10)
            ax_hist.grid(True, alpha=0.10, color="#ccc")
            for spine in ax_hist.spines.values():
                spine.set_visible(False)
            fig_hist.tight_layout(pad=1.5)

            st.pyplot(fig_hist)
            plt.close(fig_hist)

            # Quick summary metrics for the backtest
            hm1, hm2, hm3, hm4 = st.columns(4)
            hm1.metric("Total Return", _pct(base.get("total_return")))
            hm2.metric("Sharpe Ratio", _fmt(base.get("sharpe")))
            hm3.metric("Max Drawdown", _pct(base.get("max_drawdown")))
            hm4.metric("Volatility", _pct(base.get("annual_vol")))
        else:
            st.info("No historical data available.")
    ti += 1

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#aaa;font-size:0.8rem;padding:0.5rem 0 1rem;'>"
        f"Generated: {audit.get('created_at', 'N/A')} &nbsp;|&nbsp; "
        f"Strategy: {audit.get('strategy_name', 'N/A')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Methodology & disclaimer ──────────────────────────────
    with st.expander("Methodology & Important Disclaimer"):
        st.markdown("""
**How this analysis works:**

- **Backtest** — We replay your strategy on real historical market data (Yahoo Finance OHLCV), executing at next-day open prices with realistic slippage and commissions.
- **ML Model** — An XGBoost gradient-boosted tree model is trained on **20+ engineered features** (momentum, volatility, RSI, MACD, Bollinger Bands, volume dynamics, drawdown patterns). It uses **time-series walk-forward cross-validation** — the model is never shown future data during training or evaluation.
- **Out-of-sample holdout** — The most recent 20% of data is held back and never used for training. The holdout AUC is the most honest measure of predictive accuracy.
- **Early stopping** — The model trains up to 500 boosting rounds but stops early if holdout performance plateaus, preventing overfitting.
- **Monte Carlo projection** — Future scenarios are generated by resampling historical daily returns thousands of times (bootstrapped simulation).
- **Survivability score** — Weighted composite of regime consistency, parameter stability, cost resilience, drawdown severity, and base performance.

**What this is NOT:**
- This is **not** financial advice. It is a quantitative research tool.
- Past performance does not guarantee future results.
- The Monte Carlo projection assumes future returns are drawn from the same distribution as past returns — real markets can behave differently.
- The ML model has limited predictive power (typical AUC 0.52–0.65). It identifies statistical tendencies, not certainties.
- This tool does not account for taxes, margin requirements, corporate actions, or liquidity constraints beyond simple slippage estimates.

**Always consult a licensed financial advisor before making investment decisions.**
        """)

    # ── Back / Re-run button ──────────────────────────────────
    bc1, bc2, bc3 = st.columns([1, 2, 1])
    if bc2.button("Adjust Parameters & Run Again", type="secondary", use_container_width=True, key="back_btn_bottom"):
        st.session_state.pop("audit_result", None)
        st.session_state.pop("audit_mode", None)
        st.rerun()


def _render_back_button(key_suffix: str = "top"):
    """Show a compact back-to-form button above results."""
    cols = st.columns([3, 1])
    if cols[1].button("Change Parameters", key=f"back_{key_suffix}", use_container_width=True):
        st.session_state.pop("audit_result", None)
        st.session_state.pop("audit_mode", None)
        st.rerun()


# =====================================================================
# MODE 1: Test a New Strategy
# =====================================================================

if mode == "Test a New Strategy":
    st.markdown("### 🧪 Test a Trading Strategy")
    st.markdown(
        "Pick some stocks, choose your risk level, and we'll **project what could happen** "
        "to your investment going forward — along with the probability of profit, "
        "best-case, and worst-case scenarios."
    )
    st.markdown("---")

    # ── Stock picker ──────────────────────────────────────────
    st.subheader("1. Pick Your Stocks")
    stock_group = st.radio(
        "Start from a group, or pick your own:",
        list(POPULAR_STOCKS.keys()),
        horizontal=True,
    )
    default_symbols = POPULAR_STOCKS.get(stock_group, [])
    symbols = st.multiselect(
        "Stocks to trade (you can add/remove any ticker)",
        ALL_SYMBOLS,
        default=default_symbols,
        help="Type any stock ticker. These are the stocks the strategy will trade.",
    )

    # ── Risk preset ───────────────────────────────────────────
    st.subheader("2. Choose Your Risk Level")
    preset_name = st.radio(
        "How much risk are you comfortable with?",
        list(PRESETS.keys()),
        horizontal=True,
        help="This sets position sizes, exposure limits, and volatility targets for you.",
    )
    preset = PRESETS[preset_name]
    st.info(preset["desc"])

    # ── Starting capital ─────────────────────────────────────
    st.subheader("3. How Much Are You Investing?")
    initial_capital = st.number_input(
        "Starting capital ($)",
        min_value=100,
        max_value=10_000_000,
        value=10_000,
        step=1000,
        help="The dollar amount you'd start with. This affects position sizes, fees, and the real P&L numbers you see.",
    )

    # ── Projection horizon ────────────────────────────────────
    st.subheader("4. How Far Ahead Do You Want to Look?")
    st.markdown("We'll analyze how this strategy performed historically, then **project it forward** to show you what could happen.")
    proj_col_a, proj_col_b = st.columns(2)
    projection_months_input = proj_col_a.select_slider(
        "Projection horizon",
        options=[3, 6, 12, 18, 24, 36],
        value=12,
        format_func=lambda x: f"{x} months" if x < 12 else f"{x // 12} year{'s' if x > 12 else ''}" if x % 12 == 0 else f"{x} months",
        help="How far into the future to simulate.",
    )
    hist_years = proj_col_b.select_slider(
        "Historical data to learn from",
        options=[1, 2, 3, 5, 7, 10],
        value=5,
        format_func=lambda x: f"{x} year{'s' if x > 1 else ''}",
        help="More history = more data for the simulation to learn from. 5 years recommended.",
    )
    start_date = (pd.to_datetime("today") - pd.DateOffset(years=hist_years)).strftime("%Y-%m-%d")
    end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # ── Advanced (hidden) ─────────────────────────────────────
    with st.expander("Advanced settings (for experienced users)", expanded=False):
        st.markdown("*You don't need to change these. The preset above sets good defaults.*")
        adv1, adv2, adv3 = st.columns(3)
        fast_window = adv1.number_input("Fast moving average (days)", value=preset["fast_window"], min_value=5, max_value=100,
                                         help="Short-term trend indicator. Shorter = more reactive.")
        slow_window = adv2.number_input("Slow moving average (days)", value=preset["slow_window"], min_value=20, max_value=300,
                                         help="Long-term trend indicator. The strategy buys when fast crosses above slow.")
        vol_window = adv3.number_input("Volatility lookback (days)", value=preset["vol_window"], min_value=5, max_value=60,
                                        help="How many days to measure recent price swings.")

        adv4, adv5 = st.columns(2)
        position_pct = adv4.slider("Max position size (%)", 5, 50, preset["position_pct"], step=5,
                                    help="Maximum % of your portfolio in a single stock.")
        max_exposure_pct = adv5.slider("Max total investment (%)", 50, 200, preset["max_exposure_pct"], step=10,
                                        help="Total % of portfolio that can be invested (vs cash).")

        adv6, adv7 = st.columns(2)
        slippage = adv6.number_input("Price slippage (bps)", value=preset["slippage"], min_value=0.0, max_value=50.0, step=0.5,
                                      help="How much worse your fill price is vs. the quoted price. 1 bps = 0.01%.")
        commission = adv7.number_input("Trading fee (bps)", value=preset["commission"], min_value=0.0, max_value=50.0, step=0.5,
                                        help="Broker fee per trade. 1 bps = 0.01% of the trade value.")

        ml_enabled = st.checkbox("Use AI to filter trades", value=preset["ml"],
                                  help="An ML model screens out low-confidence trades.")
        vol_target_pct = st.slider("Volatility target (annual %)", 5, 60, preset["vol_target_pct"], step=5,
                                    help="Target yearly volatility. Strategy sizes positions to aim for this.")
        openai_key = st.text_input("OpenAI API key (for AI-written report)", type="password",
                                    help="Optional. Produces a richer narrative analysis.")

    # ── Run button ────────────────────────────────────────────
    if st.button("Analyze This Strategy", type="primary", use_container_width=True):
        if not symbols:
            st.error("Please select at least one stock.")
            st.stop()
        if fast_window >= slow_window:
            st.error("The fast moving average must be shorter than the slow one.")
            st.stop()

        with st.spinner("Running analysis — backtest, param sweep & stress tests. This may take a few minutes ..."):
            from pipeline.config import StrategyConfig
            from audit.orchestrator import run_full_audit

            cfg = StrategyConfig()
            cfg.symbols = list(symbols)
            cfg.benchmark_symbol = "QQQ"
            cfg.fast_window = int(fast_window)
            cfg.slow_window = int(slow_window)
            cfg.vol_window = int(vol_window)
            cfg.position_fraction = float(position_pct) / 100.0
            cfg.max_portfolio_exposure = float(max_exposure_pct) / 100.0
            cfg.vol_target_annual = float(vol_target_pct) / 100.0
            cfg.vol_target_enabled = True
            cfg.slippage_bps = float(slippage)
            cfg.commission_bps = float(commission)
            cfg.ml_enabled = bool(ml_enabled)

            try:
                result = run_full_audit(
                    cfg,
                    start=str(start_date),
                    end=str(end_date) if end_date else None,
                    openai_api_key=openai_key if openai_key else None,
                    initial_capital=float(initial_capital),
                )
            except RuntimeError as e:
                st.error(
                    f"**Could not run the backtest:** {e}\n\n"
                    "Try a longer date range (at least 6 months) or an earlier start date. "
                    "The strategy needs enough historical data to compute moving averages."
                )
                st.stop()

        result["_projection_months"] = int(projection_months_input)
        if "audit_result" in st.session_state:
            st.session_state["prev_audit_result"] = st.session_state["audit_result"]
        st.session_state["audit_result"] = result
        st.session_state["audit_mode"] = "strategy"
        st.rerun()

    if "audit_result" in st.session_state and st.session_state.get("audit_mode") == "strategy":
        _render_back_button("strategy_top")
        render_audit(
            st.session_state["audit_result"],
            prev_audit=st.session_state.get("prev_audit_result"),
        )


# =====================================================================
# MODE: Describe Your Strategy (natural language)
# =====================================================================

elif mode == "Describe Your Strategy":
    st.markdown("### Describe Your Strategy")
    st.markdown(
        "Tell us what you want to do in your own words. We'll interpret it, "
        "build a strategy, and **project what could happen** to your money — "
        "probability of profit, best/worst case scenarios, and a full risk analysis."
    )
    st.markdown("---")

    # ── Input area ─────────────────────────────────────────────
    strategy_text = st.text_area(
        "What's your trading idea?",
        height=140,
        placeholder=(
            "Example: I have $25,000 and want to invest in big tech stocks like Apple, "
            "Nvidia, and Microsoft. I'm not very experienced so keep it safe..."
        ),
    )

    with st.expander("Try one of these examples", expanded=False):
        example_strats = [
            "I want to trade big tech stocks conservatively with $10,000",
            "Aggressive momentum on NVDA, AMD, and AVGO with a 15/80 day crossover, $50k budget",
            "Safe portfolio with index ETFs and gold, low risk, $5,000 to start",
            "Trade the magnificent seven with balanced risk, no AI filter, $100k",
            "I want to invest in semiconductors with 20% position sizes and $20,000",
        ]
        for ex in example_strats:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                st.session_state["_nl_prefill"] = ex
                st.rerun()

    if "_nl_prefill" in st.session_state and not strategy_text:
        strategy_text = st.session_state.pop("_nl_prefill", "")

    # ── Supporting inputs ──────────────────────────────────────
    inp1, inp2 = st.columns(2)
    nl_capital = inp1.number_input(
        "Starting capital ($)", min_value=100, max_value=10_000_000,
        value=10_000, step=1000, key="nl_capital",
        help="How much money you want to invest.",
    )
    nl_proj_months = inp2.select_slider(
        "How far ahead to project",
        options=[3, 6, 12, 18, 24, 36],
        value=12,
        format_func=lambda x: f"{x} months" if x < 12 else f"{x // 12} year{'s' if x > 12 else ''}" if x % 12 == 0 else f"{x} months",
        help="How many months into the future to simulate.",
        key="nl_proj",
    )

    inp3, inp4 = st.columns(2)
    nl_hist_years = inp3.select_slider(
        "Historical data to learn from",
        options=[1, 2, 3, 5, 7, 10],
        value=5,
        format_func=lambda x: f"{x} year{'s' if x > 1 else ''}",
        help="More history = more data for the model to learn from. 5 years recommended.",
        key="nl_hist",
    )
    start_date_nl = (pd.to_datetime("today") - pd.DateOffset(years=nl_hist_years)).strftime("%Y-%m-%d")
    end_date_nl = pd.to_datetime("today").strftime("%Y-%m-%d")

    with st.expander("Advanced", expanded=False):
        openai_key_nl = st.text_input("OpenAI API key (optional — for smarter parsing)", type="password", key="nl_oai",
                                       help="Without a key, we use keyword matching. With a key, GPT interprets your description.")

    # ── Single-click: parse + run ──────────────────────────────
    if strategy_text and st.button("Show Me the Future", type="primary", use_container_width=True):
        from audit.strategy_parser import parse_strategy_text

        with st.spinner("Reading your strategy ..."):
            cfg, explanation = parse_strategy_text(
                strategy_text,
                openai_api_key=openai_key_nl if openai_key_nl else None,
            )

        # ── Styled interpretation card ─────────────────────────
        risk_pct = cfg.position_fraction * 100
        if risk_pct <= 20:
            risk_label, risk_color, risk_icon = "Conservative", "#27ae60", "🛡️"
        elif risk_pct <= 35:
            risk_label, risk_color, risk_icon = "Balanced", "#f39c12", "⚖️"
        else:
            risk_label, risk_color, risk_icon = "Aggressive", "#e74c3c", "🔥"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f8f9fc,#eef1f8);border:1px solid #e0e4ee;
                    border-radius:16px;padding:1.5rem 2rem;margin:1rem 0 1.5rem;">
            <div style="font-size:1.1rem;font-weight:800;margin-bottom:1rem;color:#333;">
                Here's what we understood — now projecting {nl_proj_months} months forward:
            </div>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:1rem;">
                <div style="flex:1;min-width:200px;">
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Stocks</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#333;margin-top:2px;">{', '.join(cfg.symbols)}</div>
                </div>
                <div style="flex:0 0 auto;">
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Risk Level</div>
                    <div style="font-size:1.05rem;font-weight:800;color:{risk_color};margin-top:2px;">{risk_icon} {risk_label}</div>
                </div>
                <div style="flex:0 0 auto;">
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Capital</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#333;margin-top:2px;">${nl_capital:,.0f}</div>
                </div>
                <div style="flex:0 0 auto;">
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Projection</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#667eea;margin-top:2px;">{nl_proj_months} months ahead</div>
                </div>
            </div>
            <div style="display:flex;gap:2rem;flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Strategy Type</div>
                    <div style="font-size:0.95rem;color:#555;margin-top:2px;">{cfg.fast_window}/{cfg.slow_window}-day moving average crossover</div>
                </div>
                <div>
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Position Size</div>
                    <div style="font-size:0.95rem;color:#555;margin-top:2px;">{cfg.position_fraction*100:.0f}% per stock</div>
                </div>
                <div>
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">Max Exposure</div>
                    <div style="font-size:0.95rem;color:#555;margin-top:2px;">{cfg.max_portfolio_exposure*100:.0f}% of portfolio</div>
                </div>
                <div>
                    <div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;color:#888;font-weight:700;">AI Filter</div>
                    <div style="font-size:0.95rem;color:#555;margin-top:2px;">{'Enabled' if cfg.ml_enabled else 'Disabled'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Run the full audit ─────────────────────────────────
        with st.spinner("Crunching the numbers — analyzing history and projecting forward ..."):
            from audit.orchestrator import run_full_audit

            try:
                result = run_full_audit(
                    cfg,
                    start=str(start_date_nl),
                    end=str(end_date_nl),
                    openai_api_key=openai_key_nl if openai_key_nl else None,
                    initial_capital=float(nl_capital),
                )
            except RuntimeError as e:
                st.error(
                    f"**Could not run the analysis:** {e}\n\n"
                    "The strategy needs enough historical data to learn from. "
                    "Try selecting different stocks or increasing the history window."
                )
                st.stop()

        result["_projection_months"] = int(nl_proj_months)
        if "audit_result" in st.session_state:
            st.session_state["prev_audit_result"] = st.session_state["audit_result"]
        st.session_state["audit_result"] = result
        st.session_state["audit_mode"] = "strategy"
        st.rerun()

    if "audit_result" in st.session_state and st.session_state.get("audit_mode") == "strategy":
        _render_back_button("describe_top")
        render_audit(
            st.session_state["audit_result"],
            prev_audit=st.session_state.get("prev_audit_result"),
        )


# =====================================================================
# MODE: What-If Simulator
# =====================================================================

elif mode == "What-If Simulator":
    st.markdown("### 🎛️ What-If Simulator")
    st.markdown(
        "See how your strategy would perform under different market conditions. "
        "Drag the sliders and watch the equity curve change in real time."
    )
    st.markdown("---")

    # Load the most recent audit equity curve
    equity_data = None
    source_label = None

    if "audit_result" in st.session_state:
        audit = st.session_state["audit_result"]
        eq = audit.get("_equity_df")
        if eq is None:
            records = audit.get("base_equity", [])
            if records:
                eq = pd.DataFrame(records)
        if eq is not None and not eq.empty and "portfolio_value" in eq.columns:
            equity_data = eq
            source_label = audit.get("strategy_name", "Last audit")

    if equity_data is None:
        eq_path = AUDIT_DIR / "audit_equity.csv"
        if eq_path.exists():
            equity_data = pd.read_csv(eq_path)
            source_label = "Last saved audit"

    if equity_data is None or equity_data.empty:
        st.warning(
            "No equity curve available yet. Run a strategy audit first "
            "(\"Test a New Strategy\" or \"Describe Your Strategy\"), then come back here."
        )
        st.stop()

    st.info(f"Using equity curve from: **{source_label}**")

    # ── Sliders ───────────────────────────────────────────────
    st.markdown("#### Adjust Market Conditions")

    col1, col2 = st.columns(2)
    vol_mult = col1.slider(
        "Volatility",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="1.0 = normal. 2.0 = double the market swings. 0.5 = half.",
    )
    interest_rate = col2.slider(
        "Interest Rate (annual %)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help="Higher rates = cash is more attractive = drag on your returns.",
    )

    col3, col4 = st.columns(2)
    liquidity = col3.slider(
        "Liquidity Haircut (bps/day)",
        min_value=0.0, max_value=20.0, value=0.0, step=1.0,
        help="Wider spreads = higher hidden costs. Simulates trading in less liquid markets.",
    )
    correlation = col4.slider(
        "Correlation Amplifier",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="1.0 = normal. Higher = your portfolio moves MORE with the market (correlated selloffs hurt more).",
    )

    # ── Apply shocks ──────────────────────────────────────────
    from audit.what_if import apply_what_if, load_benchmark_returns

    bench_rets = load_benchmark_returns("QQQ", str(PROJECT_ROOT))

    stressed = apply_what_if(
        equity_data,
        vol_multiplier=vol_mult,
        interest_rate_annual=interest_rate / 100.0,
        liquidity_haircut_bps=liquidity,
        correlation_amplifier=correlation,
        benchmark_returns=bench_rets,
    )

    # ── Chart ─────────────────────────────────────────────────
    st.markdown("")
    st.markdown("#### Equity Curve: Base vs. Stressed")

    chart_df = pd.DataFrame({
        "Original": stressed["base_value"],
        "With Your Scenario": stressed["portfolio_value"],
    }, index=stressed.index)

    st.line_chart(chart_df)

    # ── Impact summary ────────────────────────────────────────
    base_final = float(stressed["base_value"].iloc[-1])
    stress_final = float(stressed["portfolio_value"].iloc[-1])
    base_start = float(stressed["base_value"].iloc[0])

    base_ret = (base_final / base_start - 1) * 100 if base_start > 0 else 0
    stress_ret = (stress_final / base_start - 1) * 100 if base_start > 0 else 0
    impact = stress_ret - base_ret

    m1, m2, m3 = st.columns(3)
    m1.metric("Original Return", f"{base_ret:.1f}%")
    m2.metric("Stressed Return", f"{stress_ret:.1f}%")
    m3.metric("Impact", f"{impact:+.1f}%",
              delta_color="normal" if impact >= 0 else "inverse")

    if impact < -5:
        st.error(f"Under these conditions, your strategy loses an extra **{abs(impact):.1f}%**. "
                 "Consider reducing exposure or adding hedges.")
    elif impact < 0:
        st.warning(f"Mild impact: **{impact:.1f}%** return reduction. Manageable but worth monitoring.")
    else:
        st.success("Your strategy holds up well (or even improves) under these conditions!")

    # Worst drawdown comparison
    base_dd = float(((stressed["base_value"] / stressed["base_value"].cummax()) - 1).min() * 100)
    stress_dd = float(((stressed["portfolio_value"] / stressed["portfolio_value"].cummax()) - 1).min() * 100)
    d1, d2 = st.columns(2)
    d1.metric("Original Worst Drop", f"{base_dd:.1f}%")
    d2.metric("Stressed Worst Drop", f"{stress_dd:.1f}%")


# =====================================================================
# MODE: Check My Portfolio
# =====================================================================

elif mode == "Check My Portfolio":
    st.markdown("### 📂 Check Your Existing Portfolio")
    st.markdown(
        "Upload a CSV file showing your portfolio value over time, and we'll analyze "
        "how it performed across different market conditions."
    )
    st.markdown("---")

    st.subheader("1. Upload Your Data")
    st.markdown("Your CSV should have two columns: **date** and **portfolio value**. Example:")
    example_csv = "date,portfolio_value\n2024-01-02,100000\n2024-01-03,100250\n2024-01-04,99800\n2024-01-05,100500"
    st.code(example_csv, language="csv")

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    st.subheader("2. A Few Details")
    col_a, col_b = st.columns(2)
    strategy_name = col_a.text_input("Give it a name", value="My Portfolio",
                                      help="Just a label for the report, e.g. 'Tech Growth' or 'Retirement Fund'.")
    benchmark = col_b.selectbox("Compare against", ["QQQ (Nasdaq 100)", "SPY (S&P 500)", "IWM (Small Caps)"],
                                 help="We'll compare your returns to this market index.")
    benchmark_map = {"QQQ (Nasdaq 100)": "QQQ", "SPY (S&P 500)": "SPY", "IWM (Small Caps)": "IWM"}
    benchmark_sym = benchmark_map.get(benchmark, "QQQ")

    with st.expander("Advanced", expanded=False):
        openai_key = st.text_input("OpenAI API key (optional, for richer report)", type="password", key="up_oai")

    if uploaded is not None:
        from audit.portfolio_upload import parse_uploaded_equity

        try:
            equity_df = parse_uploaded_equity(uploaded.getvalue())
        except Exception as exc:
            st.error(f"Couldn't read your file: {exc}")
            st.stop()

        st.subheader("Preview")
        st.dataframe(equity_df.head(5), use_container_width=True, hide_index=True)
        st.line_chart(equity_df.set_index("date")["portfolio_value"])

        if st.button("Analyze My Portfolio", type="primary", use_container_width=True):
            with st.spinner("Analyzing your portfolio ..."):
                from audit.portfolio_upload import run_upload_audit

                result = run_upload_audit(
                    equity_df,
                    benchmark_symbol=benchmark_sym,
                    strategy_name=strategy_name,
                    openai_api_key=openai_key if openai_key else None,
                )

            if "audit_result" in st.session_state:
                st.session_state["prev_audit_result"] = st.session_state["audit_result"]
            st.session_state["audit_result"] = result
            st.session_state["audit_mode"] = "upload"
            st.rerun()

    if "audit_result" in st.session_state and st.session_state.get("audit_mode") == "upload":
        _render_back_button("upload_top")
        render_audit(
            st.session_state["audit_result"],
            prev_audit=st.session_state.get("prev_audit_result"),
        )


# =====================================================================
# MODE 3: View Past Results
# =====================================================================

elif mode == "View Past Results":
    st.markdown("### 📜 Past Analysis Results")
    st.markdown("---")

    available = []
    for d, label in [(AUDIT_DIR, "Strategy Test"), (UPLOAD_AUDIT_DIR, "Portfolio Check")]:
        if d.exists():
            for f in sorted(d.glob("*report.json")):
                available.append((f, label))

    if not available:
        st.info("You haven't run any analyses yet. Use one of the other modes to get started!")
        st.stop()

    selected = st.selectbox(
        "Pick a previous analysis",
        available,
        format_func=lambda x: f"{x[1]}: {x[0].name}",
    )

    if selected:
        audit = json.loads(selected[0].read_text())
        eq_path = selected[0].parent / selected[0].name.replace("_report.json", "_equity.csv")
        if eq_path.exists():
            audit["_equity_df"] = pd.read_csv(eq_path)
        render_audit(audit)
