"""
tracker/data_collector.py -- Gather all market context needed for analysis.

Fetches price data, calculates indicators, pulls economic calendar
and news headlines, then assembles everything into a prompt-ready context dict.

Usage:
    from tracker.data_collector import DataCollector
    dc = DataCollector(pair="GBPUSD=X")
    context = dc.build_context()
    print(context["price_summary"])
"""

import os
import logging
import requests
import feedparser
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger("data_collector")

# ---- Config ------------------------------------------------------------------
PAIR       = os.getenv("DEFAULT_PAIR", "GBPUSD=X")
AV_KEY     = os.getenv("ALPHA_VANTAGE_API_KEY", "")
CACHE_DIR  = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)


# ---- Price data + indicators -------------------------------------------------

def fetch_price_data(pair: str = PAIR, days: int = 210) -> dict:
    """
    Fetch OHLCV + calculate all technical indicators.
    Returns a dict with price data and indicators, or partial data on failure.
    """
    result = {
        "pair": pair,
        "status": "ok",
        "close": None,
        "open": None,
        "high": None,
        "low": None,
        "ma_50": None,
        "ma_200": None,
        "rsi_14": None,
        "macd_value": None,
        "macd_signal": None,
        "atr_14": None,
        "trend_direction": "unknown",
        "above_200ma": None,
        "nearest_support": None,
        "nearest_resistance": None,
        "df": None,
    }

    try:
        end   = date.today()
        start = end - timedelta(days=days)
        df    = yf.download(pair, start=str(start), end=str(end), progress=False, auto_adjust=True)

        if df.empty or len(df) < 50:
            result["status"] = "partial"
            log.warning(f"Insufficient price data for {pair}")
            return result

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # Moving averages
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Latest values
        latest_close  = float(close.iloc[-1])
        latest_ma50   = float(ma50.iloc[-1])   if not pd.isna(ma50.iloc[-1])   else None
        latest_ma200  = float(ma200.iloc[-1])  if not pd.isna(ma200.iloc[-1])  else None
        latest_rsi    = float(rsi.iloc[-1])    if not pd.isna(rsi.iloc[-1])    else None
        latest_macd   = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
        latest_macd_s = float(macd_sig.iloc[-1])  if not pd.isna(macd_sig.iloc[-1])  else None
        latest_atr    = float(atr.iloc[-1])    if not pd.isna(atr.iloc[-1])    else None

        # Trend direction
        if latest_ma50 and latest_ma200:
            if latest_close > latest_ma50 > latest_ma200:
                trend = "up"
            elif latest_close < latest_ma50 < latest_ma200:
                trend = "down"
            else:
                trend = "sideways"
        else:
            trend = "unknown"

        above_200 = (latest_close > latest_ma200) if latest_ma200 else None

        # Support / resistance (recent swing lows/highs over last 20 bars)
        recent = df.tail(40)
        swing_lows  = recent["Low"].nsmallest(3).values
        swing_highs = recent["High"].nlargest(3).values
        support    = float(swing_lows.mean())  if len(swing_lows)  > 0 else None
        resistance = float(swing_highs.mean()) if len(swing_highs) > 0 else None

        result.update({
            "close":              latest_close,
            "open":               float(df["Open"].iloc[-1]),
            "high":               float(df["High"].iloc[-1]),
            "low":                float(df["Low"].iloc[-1]),
            "ma_50":              latest_ma50,
            "ma_200":             latest_ma200,
            "rsi_14":             latest_rsi,
            "macd_value":         latest_macd,
            "macd_signal":        latest_macd_s,
            "atr_14":             latest_atr,
            "trend_direction":    trend,
            "above_200ma":        above_200,
            "nearest_support":    support,
            "nearest_resistance": resistance,
            "df":                 df,
            "date":               str(df.index[-1].date()),
        })
        log.info(f"Price data OK: {pair} close={latest_close:.5f} trend={trend}")

    except Exception as e:
        result["status"] = "failed"
        log.error(f"Price data failed for {pair}: {e}")

    return result


def save_snapshot(price_data: dict, pair: str = PAIR):
    """Save a market snapshot row to the database."""
    try:
        from tracker.database import get_session, MarketSnapshot
        session = get_session()
        snap = MarketSnapshot(
            snapshot_date      = date.today(),
            pair               = pair,
            open_price         = price_data.get("open"),
            high_price         = price_data.get("high"),
            low_price          = price_data.get("low"),
            close_price        = price_data.get("close"),
            ma_50              = price_data.get("ma_50"),
            ma_200             = price_data.get("ma_200"),
            rsi_14             = price_data.get("rsi_14"),
            macd_value         = price_data.get("macd_value"),
            macd_signal_line   = price_data.get("macd_signal"),
            atr_14             = price_data.get("atr_14"),
            trend_direction    = price_data.get("trend_direction"),
            above_200ma        = price_data.get("above_200ma"),
            nearest_support    = price_data.get("nearest_support"),
            nearest_resistance = price_data.get("nearest_resistance"),
        )
        session.add(snap)
        session.commit()
        session.close()
        log.info("Market snapshot saved to database.")
    except Exception as e:
        log.error(f"Failed to save snapshot: {e}")


# ---- Economic calendar -------------------------------------------------------

KNOWN_EVENTS = [
    ("First Friday of month", "US Non-Farm Payrolls", "USD", "HIGH"),
    ("Mid-month ~15th",       "US CPI Inflation",     "USD", "HIGH"),
    ("Monthly FOMC meetings", "Fed Interest Rate Decision", "USD", "HIGH"),
    ("Monthly MPC meetings",  "BoE Interest Rate Decision", "GBP", "HIGH"),
    ("Monthly ~3rd week",     "UK CPI Inflation",     "GBP", "HIGH"),
    ("Monthly ~late month",   "UK GDP Growth",        "GBP", "MEDIUM"),
    ("Weekly Thursdays",      "US Jobless Claims",    "USD", "MEDIUM"),
]


def fetch_economic_calendar() -> str:
    """
    Attempt to fetch GBP/USD high-impact events.
    Falls back to known recurring events if no live source available.
    """
    try:
        # Try investing.com economic calendar RSS
        url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
        # This is typically blocked without session cookies -- skip to fallback
        raise Exception("Live calendar requires authenticated session -- using fallback")
    except Exception:
        pass

    # Fallback: return known recurring events
    today = date.today()
    lines = [
        f"Economic Calendar Context ({today})",
        "Note: Live calendar unavailable -- showing known recurring events.",
        "",
        "HIGH-IMPACT GBP/USD EVENTS TO MONITOR:",
    ]
    for timing, event, currency, impact in KNOWN_EVENTS:
        lines.append(f"  [{impact}] [{currency}] {event} -- {timing}")

    lines += [
        "",
        "ALWAYS check for events before trading.",
        "Avoid positions within 30 minutes of HIGH-impact releases.",
    ]
    return "\n".join(lines)


# ---- News headlines ----------------------------------------------------------

NEWS_FEEDS = [
    ("FXStreet",   "https://www.fxstreet.com/rss/news"),
    ("Reuters FX", "https://feeds.reuters.com/reuters/businessNews"),
    ("Investing",  "https://www.investing.com/rss/news.rss"),
]

RELEVANCE_KEYWORDS = [
    "gbp", "pound", "usd", "dollar", "fed", "boe", "bank of england",
    "federal reserve", "inflation", "interest rate", "cpi", "nfp",
    "payroll", "gdp", "unemployment", "forex", "sterling", "currency",
    "fomc", "mpc", "uk economy", "us economy", "gbp/usd", "cable",
]


def fetch_news_headlines(max_items: int = 5) -> str:
    """
    Fetch latest forex-relevant headlines from RSS feeds.
    Returns formatted text string.
    """
    headlines = []

    for source_name, url in NEWS_FEEDS:
        if len(headlines) >= max_items:
            break
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "").lower()
                summary = entry.get("summary", "").lower()
                combined = title + " " + summary
                if any(kw in combined for kw in RELEVANCE_KEYWORDS):
                    ts = entry.get("published", str(datetime.now()))
                    headlines.append({
                        "source":  source_name,
                        "title":   entry.get("title", "No title"),
                        "time":    ts[:25],
                        "url":     entry.get("link", ""),
                    })
                    if len(headlines) >= max_items:
                        break
        except Exception as e:
            log.warning(f"News feed {source_name} failed: {e}")
            continue

    if not headlines:
        return "News headlines: No relevant headlines retrieved (feeds may be unavailable)."

    lines = [f"Latest GBP/USD Relevant Headlines ({date.today()}):"]
    for i, h in enumerate(headlines, 1):
        lines.append(f"\n  {i}. [{h['source']}] {h['title']}")
        lines.append(f"     Time: {h['time']}")
    return "\n".join(lines)


# ---- Build context -----------------------------------------------------------

class DataCollector:
    """
    Collects all market data and assembles it into a prompt-ready context dict.

    Usage:
        dc = DataCollector("GBPUSD=X")
        ctx = dc.build_context()
    """

    def __init__(self, pair: str = PAIR):
        self.pair = pair

    def build_context(self) -> dict:
        """
        Fetch all data sources and return a structured context dict.
        Never raises -- marks data as PARTIAL if a source fails.
        """
        context = {
            "pair":           self.pair,
            "date":           str(date.today()),
            "status":         "ok",
            "price_data":     None,
            "price_summary":  "",
            "calendar_text":  "",
            "news_text":      "",
            "full_context":   "",
        }

        # 1. Price data
        try:
            pd_data = fetch_price_data(self.pair)
            context["price_data"] = pd_data
            context["price_summary"] = _format_price_summary(pd_data)
            if pd_data["status"] != "ok":
                context["status"] = "partial"
        except Exception as e:
            log.error(f"Price data collection failed: {e}")
            context["price_summary"] = "Price data unavailable."
            context["status"] = "partial"

        # 2. Economic calendar
        try:
            context["calendar_text"] = fetch_economic_calendar()
        except Exception as e:
            log.error(f"Calendar fetch failed: {e}")
            context["calendar_text"] = "Economic calendar unavailable."
            context["status"] = "partial"

        # 3. News headlines
        try:
            context["news_text"] = fetch_news_headlines()
        except Exception as e:
            log.error(f"News fetch failed: {e}")
            context["news_text"] = "News headlines unavailable."
            context["status"] = "partial"

        # 4. Assemble full context prompt
        context["full_context"] = _assemble_prompt(context)

        if context["status"] == "partial":
            log.warning("Context built with partial data -- some sources failed.")
        else:
            log.info("Full context assembled OK.")

        return context


def _format_price_summary(pd_data: dict) -> str:
    if pd_data.get("close") is None:
        return f"Price data for {pd_data['pair']}: unavailable."

    p = pd_data
    lines = [
        f"GBP/USD Price Data ({p.get('date', 'today')})",
        f"  Close:       {p['close']:.5f}",
        f"  Open:        {p['open']:.5f}",
        f"  High/Low:    {p['high']:.5f} / {p['low']:.5f}",
        f"  50 MA:       {p['ma_50']:.5f}"  if p['ma_50']  else "  50 MA:       N/A",
        f"  200 MA:      {p['ma_200']:.5f}" if p['ma_200'] else "  200 MA:      N/A",
        f"  RSI(14):     {p['rsi_14']:.1f}" if p['rsi_14'] else "  RSI(14):     N/A",
        f"  MACD:        {p['macd_value']:.5f} (signal: {p['macd_signal']:.5f})"
            if p['macd_value'] and p['macd_signal'] else "  MACD:        N/A",
        f"  ATR(14):     {p['atr_14']:.5f}" if p['atr_14'] else "  ATR(14):     N/A",
        f"  Trend:       {p['trend_direction'].upper()}",
        f"  Above 200MA: {'YES' if p['above_200ma'] else 'NO'}",
        f"  Support:     {p['nearest_support']:.5f}"    if p['nearest_support']    else "  Support:     N/A",
        f"  Resistance:  {p['nearest_resistance']:.5f}" if p['nearest_resistance'] else "  Resistance:  N/A",
    ]
    return "\n".join(lines)


def _assemble_prompt(ctx: dict) -> str:
    sections = [
        "=" * 60,
        f"MARKET CONTEXT FOR {ctx['pair']} -- {ctx['date']}",
        "=" * 60,
        "",
        "-- PRICE DATA AND TECHNICAL INDICATORS --",
        ctx["price_summary"],
        "",
        "-- ECONOMIC CALENDAR --",
        ctx["calendar_text"],
        "",
        "-- RECENT NEWS HEADLINES --",
        ctx["news_text"],
        "",
        f"Data status: {ctx['status'].upper()}",
    ]
    return "\n".join(sections)


# ---- Quick test --------------------------------------------------------------
if __name__ == "__main__":
    dc = DataCollector("GBPUSD=X")
    ctx = dc.build_context()
    print(ctx["full_context"])
    print(f"\nStatus: {ctx['status']}")
