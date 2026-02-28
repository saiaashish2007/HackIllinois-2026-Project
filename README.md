# Daily Quant Trading Pipeline (Paper Only)

This repo contains a deterministic, daily-updating quant pipeline that:
- Pulls fresh daily OHLCV for liquid ETFs
- Computes indicators (fast/slow SMAs, volatility)
- Generates signals after the close
- Queues trades for next-day open fills
- Simulates executions, marks positions to market, and logs everything

> Paper trading only. No financial advice.

## Strategy (trend-following crossover)
- Universe: `SPY`, `QQQ`, `IWM`
- Bars: daily, decisions after close
- Indicators: fast SMA = 20d, slow SMA = 100d, volatility = 20d stdev of returns
- Entry: go long when fast SMA crosses above slow SMA (with optional debounce)
- Exit: go flat when fast SMA crosses below slow SMA
- Position state: long or flat; no shorts
- Execution assumption: next-day open (logged)

## Data and Persistence
- Raw data saved to `data/raw/<symbol>.csv` (append-only).
- Enriched data with indicators saved to `data/processed/<symbol>.csv`.
- Logs: `logs/daily_signals.csv`, `logs/trades.csv`, `logs/performance.csv`.
- News (best-effort): `logs/latest_news.json` (latest headlines) and `logs/news_impact.csv` (headline → forward returns).
- State (cash, positions, pending orders) in `state/portfolio_state.json` so runs resume seamlessly.

## Running the pipeline
1) Install deps (once):
```
pip install -r requirements.txt
```
2) Run the daily job (after market close):
```
python run_daily.py
```

What one run does:
1. Fetch last ~2 years of daily OHLCV per symbol (Yahoo via `yfinance`).
2. Recompute indicators and persist processed data.
3. Execute any pending orders whose planned fill date now has an open price.
4. Mark to market open positions.
5. Generate new signals; queue next-day orders as needed.
6. Append logs and save state.
7. (Optional) Fetch latest headlines per symbol and compute simple price-impact stats.

## News + “impact” (latest headlines → forward returns)
Each `python run_daily.py` run can also (optionally) pull the latest Yahoo Finance RSS headlines for your symbols and quantify “impact” as:
- Forward close-to-close returns over 1/3/5 trading days after the headline’s event date
- A simple market-adjusted version (subtracting the benchmark’s return over the same horizon)

Outputs:
- `logs/latest_news.json`: latest headlines + sentiment score
- `logs/news_impact.csv`: one row per headline with `ret_1d`, `ret_3d`, `ret_5d`, and `mkt_adj_ret_*`

You can toggle/tune this in `pipeline/config.py` via:
- `news_enabled`, `news_max_items_per_symbol`, `news_horizons`

Notebook usage:
```python
from pathlib import Path
from pipeline.news import load_latest_news_dataframe

df = load_latest_news_dataframe(Path("logs"))
df.head(20)
```

## Automation (run once per day)
- macOS cron example (runs 6:30pm ET):
```
30 18 * * 1-5 cd /Users/saibharadwaj/Desktop/stat107/saib2/project2 && /usr/bin/env python run_daily.py >> logs/cron.log 2>&1
```
- GitHub Actions (if pushed to a repo): schedule a daily workflow that executes `python run_daily.py` on a self-hosted or cloud runner.

## Files and where to look
- `run_daily.py`: entrypoint to run the whole pipeline.
- `pipeline/config.py`: symbols, SMA windows, risk limits, paths.
- `pipeline/core.py`: ingestion, indicators, signals, risk, execution, logging, state.
- `data/`, `logs/`, `state/`: created automatically on first run.

## Assumptions and guardrails
- Deterministic: same inputs + date → same outputs.
- No look-ahead: signals use only data available up to the decision bar; fills assume next-day open and only execute when that bar exists.
- Risk: one long per asset, volatility-aware sizing (downsizes high-vol names), 30% max notional per asset, 100% portfolio cap by default.
- Missing data: if a planned fill date bar is missing, the order stays pending until the bar arrives.

## Realism upgrades (closer to “real” quant backtests)
- Transaction costs: per-side **commission** + **slippage** (configurable in `pipeline/config.py`)
- No accidental leverage: BUY orders are capped by available cash unless `allow_leverage=True`
- Risk-based sizing: optional volatility targeting via `vol_target_*` settings
- ML leakage fix: inference features no longer depend on next-day prices (no look-ahead)

## Historical backtest (recommended research loop)
Instead of relying on forward daily logs, you can replay history event-by-event:

```
python -m pipeline.backtest --start 2024-01-01 --end 2025-12-31 --initial 1.0
```

Outputs (in `logs/` by default):
- `backtest_equity.csv`: daily equity curve + exposure
- `backtest_trades.csv`: fills with costs + realized PnL
- `backtest_report.json`: summary metrics (CAGR/Sharpe/max drawdown, etc.)

## Intraday (5m) backtest (day-trading style)
If you want to experiment with 5-minute bars (intraday), run:

```
python -m pipeline.intraday_backtest --initial 1.0
```

Outputs (in `logs/` by default):
- `intraday_backtest_equity.csv`
- `intraday_backtest_trades.csv`
- `intraday_backtest_report.json`

Important notes / limitations:
- Yahoo 5m data is **lookback-limited** (commonly ~60 trading days). This is controlled by `intraday_period` in `pipeline/config.py`.
- This is still a **toy execution model** (next-bar open fill + bps slippage/commission). Real intraday trading needs spread/volume constraints and better data.

## AI Survivability Audit Layer

On top of the quant pipeline sits an **AI-powered robustness auditor** that stress-tests
the strategy and produces a single **survivability score** (0–100).

### What it does
1. **Regime Detection** — classifies each day as bull / bear / high-vol / crisis using benchmark data
2. **Parameter Sweep** — grid-searches SMA windows and reports Sharpe stability
3. **Stress Tests** — increases slippage, commissions, changes vol-targets, disables ML
4. **Survivability Score** — weighted composite of regime consistency + param stability + cost resilience + drawdown
5. **LLM Report** — feeds structured results to GPT (or a template fallback) for a narrative audit

### Run the full audit (CLI)
```bash
python -m audit.orchestrator --start 2024-01-01
```
Artifacts land in `logs/audit/`:
- `audit_report.json` — structured results
- `audit_report.md` — narrative report
- `audit_equity.csv`, `audit_param_sweep.csv`, `audit_stress_tests.csv`

### Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```
Shows: equity curve, regime breakdown, parameter heatmap, stress-test bar chart, and the AI report.

### Deploy publicly (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, set main file: `app/streamlit_app.py`
4. Add `OPENAI_API_KEY` in Secrets (optional — for LLM reports)

### Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Files
- `audit/regime_detection.py` — market regime classifier
- `audit/param_sweep.py` — parameter grid search + stability score
- `audit/stress_tests.py` — cost / execution stress scenarios
- `audit/survivability_score.py` — composite robustness score
- `audit/llm_report.py` — LLM narrative generator (OpenAI or template)
- `audit/orchestrator.py` — end-to-end audit runner
- `app/streamlit_app.py` — Streamlit dashboard

## Extending
- Add/remove tickers and tweak parameters in `pipeline/config.py`.
- Add more indicators or filters in `pipeline/core.py` (e.g., volatility filter).
- Wire alerts (email/Slack) by hooking into the log-writing functions.
