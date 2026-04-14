"""
tracker/news_calendar.py -- Economic calendar and news risk assessment.

Data source: Forex Factory public calendar JSON (no API key required)
Fallback: Alpha Vantage news sentiment if FF unavailable

Fetches this week's GBP and USD high-impact events and scores
the news risk for today and tomorrow.
"""

import logging
import requests
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

log = logging.getLogger("news_calendar")

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
LONDON_TZ = ZoneInfo("Europe/London")

# Events that override the signal entirely (too binary/dangerous)
BINARY_EVENTS = [
    "interest rate decision",
    "rate decision",
    "fomc statement",
    "monetary policy statement",
    "boe inflation report",
    "federal funds rate",
    "bank rate",
]

# High-impact event keywords for GBP/USD
HIGH_RISK_KEYWORDS = [
    "cpi", "inflation", "nfp", "non-farm", "unemployment", "gdp",
    "retail sales", "pmi", "manufacturing", "services", "jobs",
    "payroll", "claimant", "trade balance", "current account",
]


def fetch_calendar():
    """
    Fetch this week's economic calendar from Forex Factory.
    Returns list of event dicts or empty list on failure.
    """
    try:
        r = requests.get(
            FF_URL,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
        # Filter to GBP and USD only
        events = [e for e in data if isinstance(e, dict) and e.get("country") in ("USD", "GBP")]
        log.info(f"Calendar fetched: {len(events)} GBP/USD events this week")
        return events
    except Exception as e:
        log.warning(f"Forex Factory calendar unavailable: {e}")
        return []


def parse_event_date(date_str):
    """Parse FF date string to date object. Returns None on failure."""
    try:
        # FF format: "2026-04-14T08:30:00-04:00"
        dt = datetime.fromisoformat(date_str)
        return dt.date()
    except Exception:
        return None


def assess_news_risk(target_date=None):
    """
    Assess news risk for the given date (default: today).

    Returns dict:
    {
        'risk_level': 'high' / 'medium' / 'low' / 'clear',
        'risk_score': 0-3 (for confluence scoring),
        'binary_event_today': bool,
        'high_impact_today': list of event dicts,
        'high_impact_tomorrow': list of event dicts,
        'all_today': list of event dicts,
        'warning_message': str or None,
        'position_multiplier': float (0.0 to 1.0),
    }
    """
    if target_date is None:
        target_date = date.today()

    tomorrow = target_date + timedelta(days=1)
    # Skip weekends for tomorrow
    if tomorrow.weekday() == 5:
        tomorrow += timedelta(days=2)
    elif tomorrow.weekday() == 6:
        tomorrow += timedelta(days=1)

    events = fetch_calendar()

    all_today = []
    high_today = []
    high_tomorrow = []
    binary_today = False

    for e in events:
        event_date = parse_event_date(e.get("date", ""))
        if event_date is None:
            continue
        impact = (e.get("impact") or "").strip()
        title  = (e.get("title") or "").lower()

        if event_date == target_date:
            all_today.append(e)
            if impact == "High":
                high_today.append(e)
                # Check if it's a binary event
                if any(kw in title for kw in BINARY_EVENTS):
                    binary_today = True

        elif event_date == tomorrow:
            if impact == "High":
                high_tomorrow.append(e)

    # Determine risk level
    if binary_today:
        risk_level = "binary"
        risk_score = 0          # Penalises confluence heavily
        position_multiplier = 0.0
        warning = build_warning(high_today, binary=True)
    elif len(high_today) >= 2:
        risk_level = "high"
        risk_score = 0
        position_multiplier = 0.0
        warning = build_warning(high_today, binary=False)
    elif len(high_today) == 1:
        risk_level = "medium"
        risk_score = 1
        position_multiplier = 0.5
        warning = build_warning(high_today, binary=False)
    elif high_tomorrow:
        risk_level = "low"
        risk_score = 2
        position_multiplier = 1.0
        warning = f"Advisory: {len(high_tomorrow)} high-impact event(s) tomorrow -- consider reducing size"
    else:
        risk_level = "clear"
        risk_score = 3
        position_multiplier = 1.0
        warning = None

    return {
        "risk_level":           risk_level,
        "risk_score":           risk_score,
        "binary_event_today":   binary_today,
        "high_impact_today":    high_today,
        "high_impact_tomorrow": high_tomorrow,
        "all_today":            all_today,
        "warning_message":      warning,
        "position_multiplier":  position_multiplier,
        "target_date":          target_date.isoformat(),
    }


def build_warning(events, binary=False):
    """Build a plain English warning string."""
    if not events:
        return None
    names = [e.get("title", "Unknown event") for e in events[:3]]
    names_str = ", ".join(names)
    if binary:
        return f"BINARY EVENT TODAY -- DO NOT TRADE: {names_str}"
    return f"HIGH IMPACT NEWS TODAY: {names_str} -- position size reduced"


def get_week_events():
    """
    Return all this week's GBP+USD events grouped by date.
    Used by the dashboard calendar tab.
    """
    events = fetch_calendar()
    by_date = {}
    for e in events:
        d = parse_event_date(e.get("date", ""))
        if d:
            key = d.isoformat()
            if key not in by_date:
                by_date[key] = []
            by_date[key].append(e)
    return by_date
