"""
scripts/run_daily.py -- Master daily analysis script.

Runs every morning via GitHub Actions (weekdays only).
Can also be triggered manually from the terminal or via .bat file.

Usage:
    python scripts/run_daily.py
    python scripts/run_daily.py --date 2024-01-15   (backtest mode)

Flow:
    1. Check it is a weekday
    2. Collect market data
    3. Run Claude analysis
    4. Run GPT analysis
    5. Compare and combine signals
    6. Calculate trade levels
    7. Save to database
    8. Write daily report file
    9. Print clean terminal summary
"""

import os
import sys
import argparse
import json
import json as json_lib
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

# -- Load env first, before any other imports --
load_dotenv(override=True)

# -- Ensure project root is on path --
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# -- Tracker imports (must come after sys.path setup) --
from tracker.confluence_engine import ConfluenceEngine
from tracker.ensemble import run_ensemble, calculate_consensus

# -- Logging -------------------------------------------------------------------
LOG_DIR = ROOT / "tracker" / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "daily.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("run_daily")

# -- Config -------------------------------------------------------------------
PAIR = os.getenv("DEFAULT_PAIR", "GBPUSD=X")
USD_TO_GBP = 0.79

PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gpt-4o":            (5.00, 15.00),
    "gpt-4o-mini":       (0.15,  0.60),
}


# ---- Helpers ----------------------------------------------------------------

def is_weekday(d: date) -> bool:
    return d.weekday() < 5   # Monday=0 ... Friday=4


def london_session(d: date) -> str:
    hour = datetime.utcnow().hour
    if 7 <= hour < 16:
        return "london"
    if 13 <= hour < 22:
        return "newyork"
    return "overlap" if 13 <= hour < 16 else "off-hours"


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    if model in PRICING:
        i, o = PRICING[model]
    else:
        i, o = 3.0, 15.0
    return round((in_tok * i + out_tok * o) / 1_000_000, 4)


def cost_gbp(model, in_tok, out_tok):
    return round(cost_usd(model, in_tok, out_tok) * USD_TO_GBP, 4)


def get_month_to_date_cost() -> float:
    """Sum GBP costs this calendar month from the costs table."""
    try:
        from tracker.database import get_session, Cost
        session = get_session()
        today = date.today()
        rows = session.query(Cost).filter(
            Cost.run_date >= date(today.year, today.month, 1)
        ).all()
        total = sum(float(r.cost_gbp or 0) for r in rows)
        session.close()
        return round(total, 4)
    except Exception:
        return 0.0


def calculate_trade_levels(price_data: dict, signal: str) -> dict:
    """
    Calculate entry, stop loss, take profit from price data.
    Uses ATR-based minimum stop distance and 1:2 R:R minimum.
    """
    close = price_data.get("close") or 0
    atr   = price_data.get("atr_14") or (close * 0.005)  # fallback 0.5%
    support    = price_data.get("nearest_support")    or (close - atr * 2)
    resistance = price_data.get("nearest_resistance") or (close + atr * 2)

    min_stop = atr * 1.2   # minimum stop distance = 1.2x ATR

    if signal == "BUY":
        entry = close
        sl    = max(support - atr * 0.3, entry - atr * 2)
        sl    = min(sl, entry - min_stop)
        dist  = entry - sl
        tp    = entry + dist * 2.0
    elif signal == "SELL":
        entry = close
        sl    = min(resistance + atr * 0.3, entry + atr * 2)
        sl    = max(sl, entry + min_stop)
        dist  = sl - entry
        tp    = entry - dist * 2.0
    else:  # HOLD
        entry = close
        sl    = close - atr * 2
        tp    = close + atr * 4
        dist  = atr * 2

    pips_stop  = abs(entry - sl) * 10000
    pips_tp    = abs(entry - tp) * 10000
    rr         = round(pips_tp / pips_stop, 2) if pips_stop > 0 else 0.0

    return {
        "entry":      round(entry, 5),
        "stop_loss":  round(sl, 5),
        "take_profit": round(tp, 5),
        "pips_stop":  round(pips_stop, 1),
        "pips_tp":    round(pips_tp, 1),
        "risk_reward": rr,
    }


def combine_signals(claude_result: dict, gpt_result: dict) -> dict:
    """
    Compare two provider results and produce a combined signal.
    If they agree -- use that signal.
    If they disagree -- weight towards higher confidence, flag disagreement.
    """
    c_sig  = claude_result["signal"]
    g_sig  = gpt_result["signal"]
    c_conf = claude_result["confidence"]
    g_conf = gpt_result["confidence"]
    agree  = (c_sig == g_sig)

    if agree:
        combined_signal     = c_sig
        combined_confidence = round((c_conf + g_conf) / 2)
    else:
        # Disagree -- pick higher confidence
        if c_conf >= g_conf:
            combined_signal     = c_sig
            combined_confidence = max(1, c_conf - 2)  # penalise disagreement
        else:
            combined_signal     = g_sig
            combined_confidence = max(1, g_conf - 2)

    market = claude_result.get("technical_summary", "")
    cond   = "trending" if "trend" in market.lower() else "ranging"

    return {
        "signal":           combined_signal,
        "confidence":       combined_confidence,
        "providers_agree":  agree,
        "claude_signal":    c_sig,
        "claude_confidence": c_conf,
        "gpt_signal":       g_sig,
        "gpt_confidence":   g_conf,
        "market_conditions": cond,
    }


def save_signal(combined: dict, claude_r: dict, gpt_r: dict,
                levels: dict, price_data: dict, analysis_date: date,
                scorecard: dict = None, confluence_price_data: dict = None,
                confluence_market_data: dict = None):
    """Save the combined signal record to the database."""
    from tracker.database import get_session, Signal
    from sqlalchemy import text as sa_text
    session = get_session()
    try:
        row = Signal(
            analysis_date       = analysis_date,
            pair                = PAIR,
            signal              = combined["signal"],
            confidence          = combined["confidence"],
            entry_price         = levels["entry"],
            stop_loss           = levels["stop_loss"],
            take_profit         = levels["take_profit"],
            risk_reward         = levels["risk_reward"],
            primary_reason      = (claude_r.get("investment_plan") or "")[:500],
            invalidation        = (claude_r.get("risk_assessment") or "")[:300],
            full_reasoning      = claude_r.get("full_reasoning", ""),
            technical_summary   = claude_r.get("technical_summary", ""),
            fundamental_summary = claude_r.get("fundamental_summary", ""),
            news_summary        = claude_r.get("news_summary", ""),
            sentiment_summary   = claude_r.get("sentiment_summary", ""),
            bull_argument       = claude_r.get("bull_argument", ""),
            bear_argument       = claude_r.get("bear_argument", ""),
            risk_assessment     = claude_r.get("risk_assessment", ""),
            claude_signal       = combined["claude_signal"],
            claude_confidence   = combined["claude_confidence"],
            gpt_signal          = combined["gpt_signal"],
            gpt_confidence      = combined["gpt_confidence"],
            providers_agree     = combined["providers_agree"],
            llm_provider        = "both",
            model_used          = f"{claude_r['model']} + {gpt_r['model']}",
            tokens_used         = claude_r.get("tokens_total", 0) + gpt_r.get("tokens_total", 0),
            estimated_cost_gbp  = claude_r.get("estimated_cost_gbp", 0) + gpt_r.get("estimated_cost_gbp", 0),
            market_conditions   = combined.get("market_conditions", ""),
            trend_direction     = price_data.get("trend_direction", ""),
            above_200ma         = price_data.get("above_200ma"),
            session             = london_session(analysis_date),
        )
        session.add(row)
        session.flush()  # get the row id before setting confluence fields
        flushed_id = int(row.id)  # capture id immediately -- row may expire after commit

        # Add confluence fields via direct UPDATE if scorecard available
        if scorecard and flushed_id:
            pd_c = confluence_price_data or {}
            md_c = confluence_market_data or {}
            try:
                def _safe(v):
                    """Convert numpy scalars to Python native types for psycopg2."""
                    if v is None:
                        return None
                    try:
                        import numpy as np
                        if isinstance(v, (np.integer,)):
                            return int(v)
                        if isinstance(v, (np.floating,)):
                            return float(v)
                        if isinstance(v, np.bool_):
                            return bool(v)
                    except ImportError:
                        pass
                    return v

                factors_json = json.dumps(scorecard.get("factors", {}))
                session.execute(sa_text("""
                    UPDATE signals SET
                        current_price = :current_price,
                        price_50ma = :price_50ma,
                        price_200ma = :price_200ma,
                        above_200ma_conf = :above_200ma_conf,
                        ma_alignment = :ma_alignment,
                        trend_direction_conf = :trend_direction_conf,
                        adx_value = :adx_value,
                        trend_strength = :trend_strength,
                        rsi_value = :rsi_value,
                        rsi_condition = :rsi_condition,
                        rsi_divergence = :rsi_divergence,
                        nearest_support = :nearest_support,
                        nearest_resistance = :nearest_resistance,
                        at_key_level = :at_key_level,
                        key_level_type = :key_level_type,
                        key_level_price = :key_level_price,
                        dxy_trend = :dxy_trend,
                        dxy_1day_change_pct = :dxy_1day_change_pct,
                        us_10yr_yield = :us_10yr_yield,
                        uk_10yr_yield = :uk_10yr_yield,
                        yield_spread = :yield_spread,
                        yield_spread_direction = :yield_spread_direction,
                        confluence_raw_score = :confluence_raw_score,
                        confluence_max_possible = :confluence_max_possible,
                        confluence_pct = :confluence_pct,
                        confluence_grade = :confluence_grade,
                        recommended_position_pct = :recommended_position_pct,
                        confluence_summary = :confluence_summary,
                        confluence_factors = :confluence_factors,
                        confluence_data_completeness_pct = :confluence_data_completeness_pct
                    WHERE id = :signal_id
                """), {
                    "current_price":                _safe(pd_c.get("current_price")),
                    "price_50ma":                   _safe(pd_c.get("price_50ma")),
                    "price_200ma":                  _safe(pd_c.get("price_200ma")),
                    "above_200ma_conf":             _safe(pd_c.get("above_200ma")),
                    "ma_alignment":                 pd_c.get("ma_alignment"),
                    "trend_direction_conf":         pd_c.get("trend_direction"),
                    "adx_value":                    _safe(pd_c.get("adx_value")),
                    "trend_strength":               pd_c.get("trend_strength"),
                    "rsi_value":                    _safe(pd_c.get("rsi_value")),
                    "rsi_condition":                pd_c.get("rsi_condition"),
                    "rsi_divergence":               pd_c.get("rsi_divergence"),
                    "nearest_support":              _safe(pd_c.get("nearest_support")),
                    "nearest_resistance":           _safe(pd_c.get("nearest_resistance")),
                    "at_key_level":                 _safe(pd_c.get("at_key_level")),
                    "key_level_type":               pd_c.get("key_level_type"),
                    "key_level_price":              _safe(pd_c.get("key_level_price")),
                    "dxy_trend":                    md_c.get("dxy_trend"),
                    "dxy_1day_change_pct":          _safe(md_c.get("dxy_1day_change_pct")),
                    "us_10yr_yield":                _safe(md_c.get("us_10yr")),
                    "uk_10yr_yield":                _safe(md_c.get("uk_10yr")),
                    "yield_spread":                 _safe(md_c.get("yield_spread")),
                    "yield_spread_direction":       md_c.get("spread_direction"),
                    "confluence_raw_score":         _safe(scorecard.get("raw_score")),
                    "confluence_max_possible":      _safe(scorecard.get("max_possible")),
                    "confluence_pct":               _safe(scorecard.get("confluence_pct")),
                    "confluence_grade":             scorecard.get("grade"),
                    "recommended_position_pct":     _safe(scorecard.get("position_size_pct")),
                    "confluence_summary":           scorecard.get("summary_text", "")[:2000],
                    "confluence_factors":           factors_json,
                    "confluence_data_completeness_pct": _safe(scorecard.get("data_completeness")),
                    "signal_id":                    flushed_id,
                })
            except Exception as ce:
                log.warning("Could not save confluence fields: %s", ce)

        session.commit()
        signal_id = flushed_id
        session.close()
        log.info(f"Signal saved to database (id={signal_id})")
        return signal_id
    except Exception as e:
        session.rollback()
        session.close()
        log.error(f"Failed to save signal: {e}")
        raise


def save_costs(claude_r: dict, gpt_r: dict, run_date: date):
    """Log costs for both providers to the costs table."""
    from tracker.database import get_session, Cost
    session = get_session()
    try:
        for r in [claude_r, gpt_r]:
            model = r["model"]
            session.add(Cost(
                run_date      = run_date,
                provider      = r["provider"],
                model         = model,
                tokens_input  = r.get("tokens_input", 0),
                tokens_output = r.get("tokens_output", 0),
                cost_usd      = cost_usd(model, r.get("tokens_input", 0), r.get("tokens_output", 0)),
                cost_gbp      = r.get("estimated_cost_gbp", 0),
                run_type      = "daily",
            ))
        session.commit()
        session.close()
        log.info("Costs logged.")
    except Exception as e:
        session.rollback()
        session.close()
        log.error(f"Failed to save costs: {e}")


def write_daily_report(analysis_date: date, combined: dict, claude_r: dict,
                       gpt_r: dict, levels: dict, price_data: dict):
    """Write the daily report text file."""
    report_dir = ROOT / "tracker" / "reports" / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    filepath = report_dir / f"{analysis_date}_GBPUSD.txt"

    agree_str = "YES -- both providers agree" if combined["providers_agree"] else "NO -- providers disagree"

    lines = [
        "=" * 70,
        f"FOREX AI SIGNAL REPORT -- GBP/USD",
        f"Date: {analysis_date}  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        "=" * 70,
        "",
        f"COMBINED SIGNAL:   {combined['signal']}",
        f"CONFIDENCE:        {combined['confidence']}/10",
        f"ENTRY:             {levels['entry']:.5f}",
        f"STOP LOSS:         {levels['stop_loss']:.5f}  (-{levels['pips_stop']:.0f} pips)",
        f"TAKE PROFIT:       {levels['take_profit']:.5f}  (+{levels['pips_tp']:.0f} pips)",
        f"RISK/REWARD:       1:{levels['risk_reward']}",
        "",
        f"Claude:            {combined['claude_signal']}  ({combined['claude_confidence']}/10)",
        f"GPT-4o:            {combined['gpt_signal']}  ({combined['gpt_confidence']}/10)",
        f"Agreement:         {agree_str}",
        "",
        "-- PRICE DATA --",
        f"Close: {price_data.get('close', 'N/A'):.5f}  "
        f"Trend: {price_data.get('trend_direction','?').upper()}  "
        f"RSI: {price_data.get('rsi_14', 0):.1f}  "
        f"Above 200MA: {'YES' if price_data.get('above_200ma') else 'NO'}",
        "",
        "-- CLAUDE ANALYSIS --",
        "",
        "Technical:",
        claude_r.get("technical_summary", "N/A"),
        "",
        "Fundamental:",
        claude_r.get("fundamental_summary", "N/A"),
        "",
        "News:",
        claude_r.get("news_summary", "N/A"),
        "",
        "Sentiment:",
        claude_r.get("sentiment_summary", "N/A"),
        "",
        "Bull argument:",
        claude_r.get("bull_argument", "N/A"),
        "",
        "Bear argument:",
        claude_r.get("bear_argument", "N/A"),
        "",
        "Risk assessment:",
        claude_r.get("risk_assessment", "N/A"),
        "",
        "Investment plan:",
        claude_r.get("investment_plan", "N/A"),
        "",
        "Final decision (Claude):",
        claude_r.get("full_reasoning", "N/A"),
        "",
        "-- GPT-4o ANALYSIS --",
        "",
        "Final decision (GPT):",
        gpt_r.get("full_reasoning", "N/A"),
        "",
        "=" * 70,
        f"Total cost this run: GBP {claude_r.get('estimated_cost_gbp',0) + gpt_r.get('estimated_cost_gbp',0):.4f}",
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"Daily report saved: {filepath}")
    return filepath


def send_signal_email(analysis_date, combined, levels, scorecard, run_cost):
    """
    Send a morning signal email via Gmail SMTP.
    Requires EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO in .env or GitHub secrets.
    Fails silently if credentials are not configured.
    """
    import smtplib
    from email.mime.text import MIMEText

    email_from = os.getenv("EMAIL_FROM", "")
    email_pass = os.getenv("EMAIL_PASSWORD", "")
    email_to   = os.getenv("EMAIL_TO", "")

    if not all([email_from, email_pass, email_to]):
        log.info("Email not configured -- skipping alert (set EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO to enable)")
        return

    try:
        signal    = combined.get("signal", "UNKNOWN")
        conf      = combined.get("confidence", 0)
        agree     = combined.get("providers_agree", False)
        entry     = levels.get("entry", 0)
        sl        = levels.get("stop_loss", 0)
        tp        = levels.get("take_profit", 0)
        rr        = levels.get("risk_reward", 0)

        grade     = scorecard["grade"]          if scorecard else "N/A"
        pct       = scorecard["confluence_pct"] if scorecard else 0
        pos_size  = scorecard["position_size_pct"] if scorecard else 0
        summary   = scorecard["summary_text"]   if scorecard else ""

        agree_str = "YES" if agree else "NO"
        pos_label = {1.5: "Aggressive (1.5%)", 1.0: "Full (1%)",
                     0.5: "Half (0.5%)", 0.25: "Minimum (0.25%)",
                     0.0: "DO NOT TRADE"}.get(float(pos_size), f"{pos_size}%")

        subject = f"GBP/USD Signal {analysis_date}: {signal} | Grade {grade} ({pct}%)"

        body = f"""GBP/USD Daily Signal -- {analysis_date}
{'=' * 50}

SIGNAL:       {signal}
CONFIDENCE:   {conf}/10
PROVIDERS:    Agree = {agree_str}

ENTRY:        {entry:.4f}
STOP LOSS:    {sl:.4f}
TAKE PROFIT:  {tp:.4f}
RISK/REWARD:  1:{rr}

CONFLUENCE:   {pct}% -- Grade {grade}
POSITION:     {pos_label}

{'=' * 50}
{summary}
{'=' * 50}

Run cost: GBP {run_cost:.4f}
Dashboard: check your Streamlit app for full details.
"""

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = email_from
        msg["To"]      = email_to

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_from, email_pass)
            server.send_message(msg)

        log.info(f"Signal email sent to {email_to}")
        print(f"  Email sent to {email_to}")

    except Exception as e:
        log.warning(f"Email send failed (non-critical): {e}")


def print_terminal_summary(analysis_date, combined, levels, price_data,
                            claude_r, gpt_r, signal_id, run_cost, mtd_cost):
    """Print the clean terminal signal box."""
    signal = combined["signal"]
    conf   = combined["confidence"]
    bar    = "#" * conf + "." * (10 - conf)
    agree_str = "YES" if combined["providers_agree"] else "NO"
    rsi   = price_data.get("rsi_14", 0) or 0
    macd_v = price_data.get("macd_value", 0) or 0
    macd_s = price_data.get("macd_signal", 0) or 0
    macd_dir = "BULLISH" if macd_v > macd_s else "BEARISH"
    trend    = (price_data.get("trend_direction") or "unknown").upper()
    above    = "YES" if price_data.get("above_200ma") else "NO"

    print("\n" + "=" * 50)
    print("  FOREX AI SIGNAL -- GBP/USD")
    print(f"  {analysis_date}  London session")
    print("=" * 50)
    print(f"  SIGNAL:      {signal}")
    print(f"  CONFIDENCE:  {conf}/10  [{bar}]")
    print(f"  ENTRY:       {levels['entry']:.5f}")
    print(f"  STOP LOSS:   {levels['stop_loss']:.5f}  (-{levels['pips_stop']:.0f} pips)")
    print(f"  TAKE PROFIT: {levels['take_profit']:.5f}  (+{levels['pips_tp']:.0f} pips)")
    print(f"  RISK/REWARD: 1:{levels['risk_reward']}")
    print("-" * 50)
    print(f"  Claude:    {combined['claude_signal']}  {combined['claude_confidence']}/10")
    print(f"  GPT-4o:    {combined['gpt_signal']}  {combined['gpt_confidence']}/10")
    print(f"  Agreement: {agree_str}")
    print("-" * 50)
    print(f"  Trend:        {trend}")
    print(f"  Above 200MA:  {above}")
    print(f"  RSI:          {rsi:.1f}   MACD: {macd_dir}")
    print("-" * 50)
    print(f"  Cost this run:    GBP {run_cost:.4f}")
    print(f"  Month to date:    GBP {mtd_cost:.4f}")
    print(f"  Signal ID:        {signal_id}")
    print(f"  Saved to database: YES")
    print("=" * 50 + "\n")


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Analysis date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    analysis_date = date.fromisoformat(args.date) if args.date else date.today()

    log.info(f"=== run_daily.py started: {datetime.now()} ===")
    log.info(f"Analysis date: {analysis_date}")

    # Step 1: Weekday check (skip if running backtest with explicit --date)
    if not args.date and not is_weekday(analysis_date):
        log.info("Weekend -- exiting cleanly.")
        print("Today is a weekend -- no analysis run.")
        sys.exit(0)

    # Step 2: DB connection
    from tracker.database import test_connection, create_tables
    if not test_connection():
        log.error("Database connection failed -- exiting.")
        sys.exit(1)
    create_tables()

    # --- Confluence: fetch technical data before AI analysis ---
    price_data_conf = None
    market_data_conf = None
    scorecard = None
    try:
        engine_c = ConfluenceEngine()
        price_data_conf = engine_c.fetch_price_data()
        market_data_conf = engine_c.fetch_market_data()
        log.info("Confluence data fetched OK")
    except Exception as e:
        log.error(f"Confluence data fetch failed (continuing without): {e}")

    # --- Ensemble: run OpenRouter models (uses pre-fetched confluence data) ---
    ensemble_votes = []
    try:
        ensemble_votes = run_ensemble(price_data_conf, market_data_conf)
        if ensemble_votes:
            log.info(f"Ensemble: {len(ensemble_votes)} models voted")
    except Exception as e:
        log.error(f"Ensemble failed (continuing without): {e}")

    # Step 3: Collect data
    log.info("Collecting market data...")
    from tracker.data_collector import DataCollector, save_snapshot
    dc  = DataCollector(PAIR)
    ctx = dc.build_context()
    price_data = ctx.get("price_data") or {}
    save_snapshot(price_data, PAIR)

    # Inject technical confluence context into the AI prompt context
    if price_data_conf:
        p = price_data_conf
        tech_ctx = (
            f"\nTECHNICAL CONTEXT (auto-calculated):\n"
            f"Current price: {p['current_price']:.4f}\n"
            f"50 MA: {p['price_50ma']:.4f} | 200 MA: {p['price_200ma']:.4f}\n"
            f"Trend: {p['trend_direction']}\n"
            f"ADX: {p.get('adx_value', 'N/A')} ({p.get('trend_strength', 'N/A')})\n"
            f"RSI: {p.get('rsi_value', 'N/A')} ({p.get('rsi_condition', 'N/A')})\n"
        )
        if p.get('nearest_support') and p.get('distance_to_support_pips') is not None:
            tech_ctx += f"Nearest support: {p['nearest_support']:.4f} ({p['distance_to_support_pips']:.0f} pips away)\n"
        if p.get('nearest_resistance') and p.get('distance_to_resistance_pips') is not None:
            tech_ctx += f"Nearest resistance: {p['nearest_resistance']:.4f} ({p['distance_to_resistance_pips']:.0f} pips away)\n"
        if market_data_conf and market_data_conf.get('dxy_trend'):
            tech_ctx += f"DXY: {market_data_conf['dxy_trend']} ({market_data_conf.get('dxy_1day_change_pct', 0):.2f}% change today)\n"
        if market_data_conf and market_data_conf.get('spread_direction'):
            tech_ctx += f"UK/US yield spread: {market_data_conf['spread_direction']}\n"
        if isinstance(ctx, dict):
            ctx['technical_context'] = tech_ctx
        elif isinstance(ctx, str):
            ctx = ctx + tech_ctx

    # Step 4: Claude analysis
    log.info("Running Claude (anthropic) analysis...")
    from tracker.agents.forex_agents import ForexTradingAgents
    try:
        claude_agent  = ForexTradingAgents(pair=PAIR, provider="anthropic")
        claude_result = claude_agent.run(date=str(analysis_date), context_data=ctx)
        log.info(f"Claude signal: {claude_result['signal']} ({claude_result['confidence']}/10)")
    except Exception as e:
        log.error(f"Claude analysis failed: {e}")
        sys.exit(1)

    # Step 5: GPT analysis
    log.info("Running GPT (openai) analysis...")
    try:
        gpt_agent   = ForexTradingAgents(pair=PAIR, provider="openai")
        gpt_result  = gpt_agent.run(date=str(analysis_date), context_data=ctx)
        log.info(f"GPT signal: {gpt_result['signal']} ({gpt_result['confidence']}/10)")
    except Exception as e:
        log.error(f"GPT analysis failed: {e}")
        sys.exit(1)

    # Step 6: Compare results
    combined = combine_signals(claude_result, gpt_result)
    log.info(f"Combined signal: {combined['signal']} conf={combined['confidence']} agree={combined['providers_agree']}")

    # Calculate consensus across all models (ensemble + existing providers)
    consensus = calculate_consensus(claude_result, gpt_result, ensemble_votes)
    providers_agree = consensus["providers_agree"]
    # Use consensus confidence if ensemble ran, otherwise keep existing
    if ensemble_votes and consensus.get("avg_confidence"):
        combined_confidence = consensus["avg_confidence"]
    else:
        combined_confidence = combined["confidence"]
    combined["providers_agree"] = providers_agree
    combined["confidence"] = combined_confidence
    final_signal = consensus["final_signal"]
    combined["signal"] = final_signal
    log.info(f"Consensus: {final_signal} {consensus['agreement_pct']}% agreement ({consensus['vote_count']} models)")

    # --- Confluence: score the signal ---
    try:
        scorecard = engine_c.calculate_score(
            signal=combined["signal"],
            ai_confidence=combined["confidence"],
            providers_agree=providers_agree,
            price_data=price_data_conf,
            market_data=market_data_conf,
            agreement_pct=consensus.get("agreement_pct"),
            vote_count=consensus.get("vote_count"),
        )
        log.info(f"Confluence score: {scorecard['confluence_pct']}% grade={scorecard['grade']}")
    except Exception as e:
        log.error(f"Confluence scoring failed (continuing without): {e}")
        scorecard = None

    # Step 7: Trade levels
    levels = calculate_trade_levels(price_data, combined["signal"])

    # Step 8: Save signal
    signal_id = save_signal(
        combined, claude_result, gpt_result, levels, price_data, analysis_date,
        scorecard=scorecard,
        confluence_price_data=price_data_conf,
        confluence_market_data=market_data_conf,
    )

    # Step 8b: Save ensemble model votes to database
    try:
        from tracker.database import get_session, ModelVote
        vsession = get_session()
        for vote in consensus.get("all_votes", []):
            # Skip Claude and GPT -- already tracked in signals table
            if vote.get("provider") == "openrouter":
                mv = ModelVote(
                    signal_id     = signal_id,
                    analysis_date = analysis_date,
                    model_name    = vote["model_name"],
                    provider      = vote["provider"],
                    signal        = vote["signal"],
                    confidence    = vote["confidence"],
                    reasoning     = vote.get("reasoning", ""),
                    cost_usd      = vote.get("cost_usd", 0),
                    cost_gbp      = vote.get("cost_gbp", 0),
                    latency_ms    = vote.get("latency_ms", 0),
                )
                vsession.add(mv)
        vsession.commit()
        vsession.close()
        or_count = len([v for v in consensus.get("all_votes", []) if v.get("provider") == "openrouter"])
        log.info(f"Model votes saved: {or_count} OpenRouter votes")
    except Exception as e:
        log.error(f"Failed to save model votes (non-critical): {e}")

    # Step 8b-ii: Update signals table with ensemble summary fields
    if signal_id and ensemble_votes:
        try:
            from tracker.database import get_engine
            from sqlalchemy import text as sa_text2
            eng = get_engine()
            with eng.connect() as conn:
                conn.execute(sa_text2("""
                    UPDATE signals SET
                        ensemble_vote_count = :vc,
                        ensemble_agreement_pct = :ap
                    WHERE id = :sid
                """), {"vc": consensus["vote_count"], "ap": consensus["agreement_pct"], "sid": signal_id})
                conn.commit()
        except Exception as e:
            log.error(f"Could not update ensemble fields: {e}")

    # Step 8c: Open virtual paper trade
    vtrade = None
    try:
        from tracker.virtual_account import open_trade
        from tracker.database import get_session, initialise_virtual_account
        vsession = get_session()
        initialise_virtual_account(vsession)
        trade_signal = {
            "id":            signal_id,
            "signal":        combined["signal"],
            "entry_price":   levels.get("entry"),
            "stop_loss":     levels.get("stop_loss"),
            "take_profit":   levels.get("take_profit"),
            "analysis_date": analysis_date,
            "ai_confidence": combined["confidence"],
            "providers_agree": combined["providers_agree"],
        }
        vtrade = open_trade(
            vsession,
            trade_signal,
            confluence_grade=scorecard["grade"] if scorecard else None,
            risk_pct=scorecard["position_size_pct"] if scorecard else None,
        )
        if vtrade:
            log.info(
                "Paper trade opened: risk=GBP %.2f value_per_pip=GBP %.4f",
                float(vtrade.risk_gbp),
                float(vtrade.value_per_pip),
            )
        vsession.close()
    except Exception as e:
        log.error("Virtual trade open failed (non-critical): %s", e)

    # Step 9: Save costs
    save_costs(claude_result, gpt_result, analysis_date)
    run_cost = claude_result.get("estimated_cost_gbp", 0) + gpt_result.get("estimated_cost_gbp", 0)
    mtd_cost = get_month_to_date_cost()

    # Step 10: Daily report file
    report_path = write_daily_report(analysis_date, combined, claude_result, gpt_result, levels, price_data)

    # Step 11: Print summary
    print_terminal_summary(analysis_date, combined, levels, price_data,
                           claude_result, gpt_result, signal_id, run_cost, mtd_cost)

    # Step 11b: Print ensemble vote breakdown
    if consensus.get("all_votes"):
        print("-" * 50)
        print(f"  ENSEMBLE VOTE ({consensus['vote_count']} models, {consensus['agreement_pct']}% agree)")
        print("-" * 50)
        for v in consensus["all_votes"]:
            model_short = v["model_name"].split("/")[-1][:20]
            print(f"  {v['signal']:4s} {v['confidence']:2d}/10  {model_short}")
        print(f"  CONSENSUS: {consensus['final_signal']} ({consensus['agreement_pct']}% agreement)")
        print("-" * 50)

    # Step 12: Print confluence output
    if scorecard:
        pct = scorecard['confluence_pct']
        grade = scorecard['grade']
        raw = scorecard['raw_score']
        mx = scorecard['max_possible']
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = '#' * filled + '.' * (bar_len - filled)
        print("-" * 50)
        print(f"  CONFLUENCE: {pct}%  GRADE: {grade}")
        print(f"  Score: {raw}/{mx}  [{bar}]")
        print("-" * 50)
        # supporting/conflicting are label strings in the existing engine
        for lbl in scorecard.get('supporting', [])[:2]:
            print(f"  [+] {lbl}")
        for lbl in scorecard.get('conflicting', [])[:1]:
            print(f"  [-] {lbl}")
        print("-" * 50)
        print(f"  POSITION SIZE: {scorecard['position_size_pct']}% risk")
        print("-" * 50)

    # Step 12b: Print paper trade info
    if vtrade and vtrade.status == "open":
        balance  = float(vtrade.opening_balance)
        risk_gbp = float(vtrade.risk_gbp)
        print(f"  PAPER TRADE: GBP {risk_gbp:.2f} at risk ({vtrade.risk_pct}% of GBP {balance:.2f})")
        print(f"  Spread cost: GBP {float(vtrade.spread_cost_gbp):.4f}")
        print("-" * 50)

    # Step 13: Send email alert
    send_signal_email(analysis_date, combined, levels, scorecard, run_cost)

    log.info(f"=== run_daily.py complete. Signal ID: {signal_id} ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
