import json
from datetime import datetime, time, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

# Optional dependency. If missing, we fall back to a tiny rule-based scorer.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _VADER = None


NewsItem = Dict[str, Any]


def _safe_parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = parsedate_to_datetime(dt_str)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _headline_sentiment(text: str) -> Optional[float]:
    """
    Returns a sentiment score in [-1, 1] (higher = more positive).
    """
    if not text:
        return None
    if _VADER is not None:
        try:
            return float(_VADER.polarity_scores(text).get("compound"))
        except Exception:
            pass

    # Simple fallback (kept intentionally small + deterministic)
    t = text.lower()
    pos = ["beats", "surge", "soar", "upgrade", "record", "strong", "profit", "growth"]
    neg = ["miss", "drop", "fall", "downgrade", "lawsuit", "weak", "loss", "cut"]
    score = 0
    score += sum(1 for w in pos if w in t)
    score -= sum(1 for w in neg if w in t)
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 4.0))


def fetch_latest_news_yahoo_rss(
    symbol: str,
    *,
    max_items: int = 10,
    timeout_s: int = 10,
    session: Optional[requests.Session] = None,
) -> List[NewsItem]:
    """
    Fetch latest news via Yahoo Finance RSS (no API key).
    """
    # Example: https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    sess = session or requests.Session()
    resp = sess.get(url, timeout=timeout_s, headers={"User-Agent": "stat107-news-impact/1.0"})
    resp.raise_for_status()

    # Minimal RSS parsing to avoid extra deps:
    # We'll use pandas' read_xml if available? It's brittle across feeds.
    # Instead: rely on feedparser only if installed, else parse with ElementTree.
    items: List[NewsItem] = []
    try:
        import feedparser  # type: ignore

        feed = feedparser.parse(resp.content)
        for entry in (feed.entries or [])[: int(max_items)]:
            published_dt = _safe_parse_datetime(
                getattr(entry, "published", None) or getattr(entry, "updated", None)
            )
            title = getattr(entry, "title", None)
            link = getattr(entry, "link", None)
            summary = getattr(entry, "summary", None)
            items.append(
                {
                    "symbol": symbol,
                    "source": "yahoo_rss",
                    "title": title,
                    "link": link,
                    "published_at_utc": published_dt.isoformat() if published_dt else None,
                    "summary": summary,
                    "sentiment": _headline_sentiment(" ".join([str(title or ""), str(summary or "")]).strip()),
                }
            )
        return items
    except Exception:
        # Fallback XML parsing (best-effort)
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return items
        for item in list(channel.findall("item"))[: int(max_items)]:
            title = (item.findtext("title") or "").strip() or None
            link = (item.findtext("link") or "").strip() or None
            pub = (item.findtext("pubDate") or "").strip() or None
            published_dt = _safe_parse_datetime(pub)
            desc = (item.findtext("description") or "").strip() or None
            items.append(
                {
                    "symbol": symbol,
                    "source": "yahoo_rss",
                    "title": title,
                    "link": link,
                    "published_at_utc": published_dt.isoformat() if published_dt else None,
                    "summary": desc,
                    "sentiment": _headline_sentiment(" ".join([str(title or ""), str(desc or "")]).strip()),
                }
            )
        return items


def fetch_latest_news_yfinance(
    symbol: str,
    *,
    max_items: int = 10,
) -> List[NewsItem]:
    """
    Fetch latest news via yfinance's built-in news endpoint.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []

    items: List[NewsItem] = []
    try:
        t = yf.Ticker(symbol)
        news = getattr(t, "news", None) or []
        for entry in list(news)[: int(max_items)]:
            # yfinance provides providerPublishTime as unix seconds
            pub = entry.get("providerPublishTime")
            pub_dt = None
            try:
                pub_dt = datetime.fromtimestamp(int(pub), tz=timezone.utc) if pub is not None else None
            except Exception:
                pub_dt = None

            title = entry.get("title")
            link = entry.get("link")
            publisher = entry.get("publisher")
            summary = entry.get("summary") or entry.get("content")
            items.append(
                {
                    "symbol": symbol,
                    "source": "yfinance",
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "published_at_utc": pub_dt.isoformat() if pub_dt else None,
                    "summary": summary,
                    "sentiment": _headline_sentiment(" ".join([str(title or ""), str(summary or "")]).strip()),
                }
            )
    except Exception:
        return []
    return items


def _next_trading_index(
    idx: pd.DatetimeIndex, *, day: datetime.date, after_close: bool
) -> Optional[pd.Timestamp]:
    if idx is None or len(idx) == 0:
        return None
    day_ts = pd.Timestamp(day)
    idx_sorted = pd.DatetimeIndex(pd.to_datetime(idx)).sort_values()

    # If headline is after close, treat it as next day's event.
    if after_close:
        # strictly after day
        candidates = idx_sorted[idx_sorted.date > day]
        return candidates.min() if len(candidates) else None

    # Otherwise: first trading day on/after the date
    candidates = idx_sorted[idx_sorted.date >= day]
    return candidates.min() if len(candidates) else None


def _trading_horizon_dt(idx: pd.DatetimeIndex, event_dt: pd.Timestamp, horizon: int) -> Optional[pd.Timestamp]:
    if idx is None or len(idx) == 0 or event_dt is None:
        return None
    idx_sorted = pd.DatetimeIndex(pd.to_datetime(idx)).sort_values()
    try:
        loc = int(idx_sorted.get_loc(event_dt))
    except KeyError:
        return None
    tgt = loc + int(horizon)
    if tgt < 0 or tgt >= len(idx_sorted):
        return None
    return pd.Timestamp(idx_sorted[tgt])


def compute_news_impact(
    news_items: Sequence[NewsItem],
    *,
    prices: Dict[str, pd.DataFrame],
    benchmark: Optional[pd.DataFrame] = None,
    horizons: Sequence[int] = (1, 3, 5),
    market_close_time_et: time = time(16, 0),
) -> pd.DataFrame:
    """
    For each news item, compute forward close-to-close returns over given horizons.
    Also computes a simple market-adjusted return by subtracting benchmark return.
    """
    rows: List[Dict[str, Any]] = []

    bench_idx = None
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        bench_idx = pd.DatetimeIndex(pd.to_datetime(benchmark.index)).sort_values()

    for item in news_items:
        sym = str(item.get("symbol") or "").strip()
        if not sym or sym not in prices:
            continue
        df = prices[sym]
        if df is None or df.empty or "Close" not in df.columns:
            continue

        idx = pd.DatetimeIndex(pd.to_datetime(df.index)).sort_values()
        pub_iso = item.get("published_at_utc")
        pub_dt_utc = None
        try:
            pub_dt_utc = datetime.fromisoformat(pub_iso) if pub_iso else None
        except Exception:
            pub_dt_utc = None

        pub_date = pub_dt_utc.date() if pub_dt_utc else None
        after_close = False
        if pub_dt_utc is not None:
            try:
                from zoneinfo import ZoneInfo

                pub_dt_et = pub_dt_utc.astimezone(ZoneInfo("America/New_York"))
                after_close = pub_dt_et.time() >= market_close_time_et
            except Exception:
                # Fallback heuristic if timezone conversion isn't available
                after_close = pub_dt_utc.time() >= time(20, 0)  # ~16:00 ET in UTC (most of the year)

        if pub_date is None:
            continue

        event_dt = _next_trading_index(idx, day=pub_date, after_close=after_close)
        if event_dt is None or event_dt not in idx:
            continue

        event_close = float(df.loc[event_dt, "Close"])
        if event_close <= 0:
            continue

        base: Dict[str, Any] = {
            "symbol": sym,
            "source": item.get("source"),
            "title": item.get("title"),
            "link": item.get("link"),
            "published_at_utc": item.get("published_at_utc"),
            "event_date": pd.to_datetime(event_dt).date().isoformat(),
            "event_close": event_close,
            "sentiment": item.get("sentiment"),
        }

        for h in horizons:
            end_dt = _trading_horizon_dt(idx, event_dt, int(h))
            if end_dt is None:
                base[f"ret_{h}d"] = None
                base[f"mkt_adj_ret_{h}d"] = None
                continue
            end_close = float(df.loc[end_dt, "Close"])
            ret = float(end_close / event_close - 1.0) if event_close > 0 else None
            base[f"ret_{h}d"] = ret

            # Benchmark-adjusted
            if bench_idx is not None and benchmark is not None and end_dt in bench_idx and event_dt in bench_idx:
                b0 = float(benchmark.loc[event_dt, "Close"])
                b1 = float(benchmark.loc[end_dt, "Close"])
                bret = float(b1 / b0 - 1.0) if b0 > 0 else None
                base[f"mkt_adj_ret_{h}d"] = (ret - bret) if (ret is not None and bret is not None) else None
            else:
                base[f"mkt_adj_ret_{h}d"] = None

        rows.append(dict(base))

    return pd.DataFrame(rows)


def _append_dedup_csv(path: Path, df: pd.DataFrame, dedup_cols: Sequence[str]) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, out], ignore_index=True, sort=False)
    else:
        combined = out

    # Only dedup if the columns exist
    cols = [c for c in dedup_cols if c in combined.columns]
    if cols:
        combined = combined.drop_duplicates(subset=cols, keep="last")

    combined.to_csv(path, index=False)


def write_latest_news_json(path: Path, news_items: Sequence[NewsItem], meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "meta": meta or {},
        "items": list(news_items),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_news_step(
    *,
    symbols: Sequence[str],
    logs_dir: Path,
    prices: Dict[str, pd.DataFrame],
    benchmark: Optional[pd.DataFrame],
    max_items_per_symbol: int = 8,
    horizons: Sequence[int] = (1, 3, 5),
    timeout_s: int = 10,
    cache_ttl_minutes: int = 180,
    sleep_s: float = 0.25,
) -> Tuple[List[NewsItem], pd.DataFrame]:
    sess = requests.Session()
    all_items: List[NewsItem] = []
    per_symbol_status: Dict[str, str] = {}

    cache_dir = logs_dir / "news_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        cache_path = cache_dir / f"{sym}.json"
        try:
            if cache_path.exists():
                age_s = (datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)).total_seconds()
                if age_s <= float(cache_ttl_minutes) * 60.0:
                    with cache_path.open("r", encoding="utf-8") as f:
                        cached = json.load(f) or {}
                    cached_items = cached.get("items") or []
                    if isinstance(cached_items, list) and cached_items:
                        all_items.extend(cached_items)
                        per_symbol_status[sym] = "cached"
                        continue
        except Exception:
            # ignore cache read errors; fall back to fetching
            pass

        try:
            items = fetch_latest_news_yfinance(sym, max_items=int(max_items_per_symbol))
            if not items:
                items = fetch_latest_news_yahoo_rss(
                    sym, max_items=int(max_items_per_symbol), timeout_s=int(timeout_s), session=sess
                )
            if items:
                all_items.extend(items)
                per_symbol_status[sym] = "fetched"
                try:
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {"created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), "items": items},
                            f,
                            indent=2,
                            sort_keys=True,
                        )
                except Exception:
                    pass
            else:
                per_symbol_status[sym] = "empty"
        except Exception:
            # keep pipeline resilient; news is a best-effort enhancement
            per_symbol_status[sym] = "error"
            continue
        try:
            # Gentle throttle to reduce Yahoo-side 429s
            import time as _time

            _time.sleep(max(0.0, float(sleep_s)))
        except Exception:
            pass

    impact_df = compute_news_impact(
        all_items, prices=prices, benchmark=benchmark, horizons=horizons
    )

    # Persist
    write_latest_news_json(
        logs_dir / "latest_news.json",
        all_items,
        meta={
            "symbols": list(symbols),
            "max_items_per_symbol": int(max_items_per_symbol),
            "horizons": list(horizons),
            "per_symbol_status": per_symbol_status,
        },
    )
    impact_path = logs_dir / "news_impact.csv"
    if impact_df is None or impact_df.empty:
        # Ensure the file exists for UX (even if we couldn't compute impact yet).
        if not impact_path.exists():
            cols = [
                "symbol",
                "source",
                "title",
                "link",
                "published_at_utc",
                "event_date",
                "event_close",
                "sentiment",
            ]
            for h in horizons:
                cols.append(f"ret_{int(h)}d")
                cols.append(f"mkt_adj_ret_{int(h)}d")
            pd.DataFrame(columns=cols).to_csv(impact_path, index=False)
    else:
        _append_dedup_csv(
            impact_path,
            impact_df,
            dedup_cols=("symbol", "link", "published_at_utc"),
        )
    return all_items, impact_df


def load_latest_news_dataframe(logs_dir: Path) -> pd.DataFrame:
    """
    Notebook-friendly loader that combines `logs/latest_news.json` with `logs/news_impact.csv` (if present).
    """
    latest_path = logs_dir / "latest_news.json"
    if not latest_path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    items = payload.get("items") or []
    if not isinstance(items, list) or not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)

    if "published_at_utc" in df.columns:
        df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], errors="coerce", utc=True)

    impact_path = logs_dir / "news_impact.csv"
    if impact_path.exists():
        try:
            impact = pd.read_csv(impact_path)
            if "published_at_utc" in impact.columns:
                impact["published_at_utc"] = pd.to_datetime(impact["published_at_utc"], errors="coerce", utc=True)
            key_cols = [c for c in ["symbol", "link", "published_at_utc"] if c in df.columns and c in impact.columns]
            if key_cols:
                df = df.merge(
                    impact,
                    on=key_cols,
                    how="left",
                    suffixes=("", "_impact"),
                )
        except Exception:
            pass

    # Most recent first
    if "published_at_utc" in df.columns:
        df = df.sort_values("published_at_utc", ascending=False)
    return df.reset_index(drop=True)

