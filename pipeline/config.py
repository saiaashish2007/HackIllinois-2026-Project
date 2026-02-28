from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List


@dataclass
class StrategyConfig:
    """
    Configuration for the daily quant pipeline.
    Adjust symbols/parameters to tune the strategy without touching core logic.
    """

    symbols: List[str] = field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "NVDA",
            "AVGO",
            "AMD",
            "CRM",
            "ORCL",
            "ADBE",
            "INTC",
            "TSM",
            "QCOM",
            "IBM",
            "SAP",
            "NOW",
            "PANW",
            "NFLX",
        ]
    )
    benchmark_symbol: str = "QQQ"
    fast_window: int = 20
    slow_window: int = 100
    vol_window: int = 20
    debounce_days: int = 1  # require this many consecutive days of condition before acting
    position_fraction: float = 0.30  # target notional per asset when long
    max_portfolio_exposure: float = 1.00  # cap on total gross exposure
    allow_leverage: bool = False  # if False, BUY orders are capped by available cash
    cash_buffer: float = 0.00  # keep this fraction of portfolio in cash (0.02 = 2%)
    lookback_years: int = 10  # data fetch horizon — supports up to 10yr backtests
    execution_price: str = "next_open"  # assumption used in simulation and logs
    warmup_buffer: int = 10  # extra bars beyond indicator windows for safety
    # Execution / transaction cost model (simple but non-zero)
    commission_bps: float = 0.5  # per-side commission in bps of notional
    min_commission: float = 0.0  # absolute minimum commission per fill (in portfolio-value units)
    slippage_bps: float = 1.0  # per-side slippage in bps applied to fill price
    # Filters
    rel_strength_lookback: int = 60  # trading days
    rel_strength_min: float = 0.0  # require outperformance vs benchmark
    max_volatility: float = 0.60  # annualized vol cap for entries
    fundamentals_pe_max: float = 80.0
    fundamentals_ps_max: float = 20.0
    fundamentals_allow_missing: bool = True
    # ML settings
    ml_enabled: bool = True
    ml_years: int = 10
    ml_prob_min: float = 0.52  # minimum predicted prob(up) to allow BUY
    ml_model_name: str = "xgb_classification"

    # News + impact (best-effort enhancement; won't break pipeline if it fails)
    news_enabled: bool = True
    news_max_items_per_symbol: int = 8
    news_horizons: List[int] = field(default_factory=lambda: [1, 3, 5])
    news_timeout_s: int = 10
    news_cache_ttl_minutes: int = 180  # reuse recent cached headlines to reduce rate-limits
    news_sleep_s: float = 0.25  # throttle between network calls

    # Risk-based sizing
    vol_target_enabled: bool = True
    vol_target_annual: float = 0.20  # target annualized vol for a single position (scales DOWN in high-vol names)
    vol_target_floor: float = 0.05  # avoid extreme sizing when vol estimate is tiny

    # Intraday (Yahoo) settings
    intraday_interval: str = "5m"
    intraday_period: str = "60d"  # Yahoo 5m is limited (typically ~60 days)
    intraday_timezone: str = "America/New_York"
    intraday_rth_only: bool = True  # keep regular trading hours only (9:30-16:00 ET)
    intraday_bars_per_day: int = 78  # 6.5 hours * 12 (5m bars/hour)

    @property
    def intraday_bars_per_year(self) -> int:
        return 252 * int(self.intraday_bars_per_day)

    # Paths (resolved relative to repository root)
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def ml_model_path(self) -> Path:
        return self.models_dir / f"{self.ml_model_name}.json"

    @property
    def ml_meta_path(self) -> Path:
        return self.models_dir / f"{self.ml_model_name}_meta.json"

    @property
    def state_path(self) -> Path:
        return self.project_root / "state" / "portfolio_state.json"

    @property
    def lookback_days(self) -> int:
        # Slightly generous to avoid daylight savings / holiday off-by-one issues
        return int(self.lookback_years * 365) + self.warmup_buffer

    @property
    def warmup_period(self) -> int:
        return max(self.fast_window, self.slow_window, self.vol_window) + self.warmup_buffer

    def ensure_directories(self) -> None:
        for path in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.logs_dir,
            self.models_dir,
            self.state_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)
