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
from tracker.news_calendar import assess_news_risk

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

    min_stop = atr * 0.5   # ~50 pips at typical GBPUSD ATR

    if signal == "BUY":
        entry = close
        sl    = max(support - atr * 0.1, entry - atr * 0.6)
        sl    = min(sl, entry - min_stop)   # never tighter than min_stop
        dist  = entry - sl
        tp    = entry + dist * 2.0          # 1:2 R:R
    elif signal == "SELL":
        entry = close
        sl    = min(resistance + atr * 0.1, entry + atr * 0.6)
        sl    = max(sl, entry + min_stop)
        dist  = sl - entry
        tp    = entry - dist * 2.0
    else:  # HOLD
        entry = close
        sl    = close - atr * 0.6
        tp    = close + atr * 1.2
        dist  = atr * 0.6

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
                confluence_market_data: dict = None, entry_strategy: dict = None,
                mtf_data: dict = None, mtf_aligned: bool = None,
                news_risk: dict = None, news_trade_blocked: bool = False,
                risk_env: dict = None):
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
            primary_reason      = (claude_r.get("investment_plan") or claude_r.get("full_reasoning") or "")[:500],
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

        # Save entry strategy fields
        if entry_strategy and flushed_id:
            try:
                session.execute(sa_text("""
                    UPDATE signals SET
                        order_type          = :order_type,
                        smart_entry_price   = :smart_entry_price,
                        entry_rationale     = :entry_rationale,
                        pips_from_current   = :pips_from_current,
                        order_expires_bars  = :order_expires_bars
                    WHERE id = :signal_id
                """), {
                    "order_type":         entry_strategy.get("order_type"),
                    "smart_entry_price":  entry_strategy.get("entry_price"),
                    "entry_rationale":    (entry_strategy.get("entry_rationale") or "")[:500],
                    "pips_from_current":  entry_strategy.get("pips_from_current"),
                    "order_expires_bars": entry_strategy.get("expires_bars"),
                    "signal_id":          flushed_id,
                })
            except Exception as ee:
                log.warning("Could not save entry strategy fields: %s", ee)

        # Save MTF fields
        if mtf_data and flushed_id:
            try:
                session.execute(sa_text("""
                    UPDATE signals SET
                        weekly_trend        = :weekly_trend,
                        weekly_above_200ma  = :weekly_above_200ma,
                        weekly_rsi          = :weekly_rsi,
                        weekly_ma_alignment = :weekly_ma_alignment,
                        h4_trend            = :h4_trend,
                        h4_above_50ma       = :h4_above_50ma,
                        h4_rsi              = :h4_rsi,
                        h4_ma_alignment     = :h4_ma_alignment,
                        mtf_bias            = :mtf_bias,
                        mtf_aligned         = :mtf_aligned,
                        mtf_notes           = :mtf_notes
                    WHERE id = :signal_id
                """), {
                    "weekly_trend":        mtf_data.get("weekly_trend"),
                    "weekly_above_200ma":  bool(mtf_data.get("weekly_above_200ma")),
                    "weekly_rsi":          float(mtf_data["weekly_rsi"]) if mtf_data.get("weekly_rsi") is not None else None,
                    "weekly_ma_alignment": mtf_data.get("weekly_ma_alignment"),
                    "h4_trend":            mtf_data.get("h4_trend"),
                    "h4_above_50ma":       bool(mtf_data.get("h4_above_50ma")),
                    "h4_rsi":              float(mtf_data["h4_rsi"]) if mtf_data.get("h4_rsi") is not None else None,
                    "h4_ma_alignment":     mtf_data.get("h4_ma_alignment"),
                    "mtf_bias":            mtf_data.get("mtf_bias"),
                    "mtf_aligned":         bool(mtf_aligned),
                    "mtf_notes":           mtf_data.get("mtf_notes"),
                    "signal_id":           flushed_id,
                })
            except Exception as me:
                log.warning("Could not save MTF fields: %s", me)
                try:
                    session.rollback()
                except Exception:
                    pass

        # Save news risk fields
        if flushed_id:
            try:
                event_names = ""
                rl = ""
                if news_risk:
                    rl = news_risk.get("risk_level", "")
                    events = news_risk.get("high_impact_today", [])
                    event_names = ", ".join(e.get("title", "") for e in events[:5])
                session.execute(sa_text("""
                    UPDATE signals SET
                        news_risk_level    = :news_risk_level,
                        news_event_names   = :news_event_names,
                        news_trade_blocked = :news_trade_blocked
                    WHERE id = :signal_id
                """), {
                    "news_risk_level":    rl,
                    "news_event_names":   event_names,
                    "news_trade_blocked": news_trade_blocked,
                    "signal_id":          flushed_id,
                })
            except Exception as ne:
                log.warning("Could not save news fields: %s", ne)
                try:
                    session.rollback()
                except Exception:
                    pass

        # Save risk environment fields (VIX + EURUSD)
        if flushed_id and risk_env:
            try:
                session.execute(sa_text("""
                    UPDATE signals SET
                        vix_current  = :vix_current,
                        vix_level    = :vix_level,
                        vix_signal   = :vix_signal,
                        eurusd_trend = :eurusd_trend,
                        eurusd_rsi   = :eurusd_rsi
                    WHERE id = :signal_id
                """), {
                    "vix_current":  float(risk_env["vix_current"]) if risk_env.get("vix_current") is not None else None,
                    "vix_level":    risk_env.get("vix_level"),
                    "vix_signal":   risk_env.get("vix_signal"),
                    "eurusd_trend": risk_env.get("eurusd_trend"),
                    "eurusd_rsi":   float(risk_env["eurusd_rsi"]) if risk_env.get("eurusd_rsi") is not None else None,
                    "signal_id":    flushed_id,
                })
            except Exception as re_:
                log.warning("Could not save risk env fields: %s", re_)
                try:
                    session.rollback()
                except Exception:
                    pass

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
        (
            f"Close: {price_data['close']:.5f}  " if price_data.get('close') is not None else "Close: N/A  "
        ) + (
            f"Trend: {price_data.get('trend_direction','?').upper()}  "
        ) + (
            f"RSI: {price_data['rsi_14']:.1f}  " if price_data.get('rsi_14') is not None else "RSI: N/A  "
        ) + (
            f"Above 200MA: {'YES' if price_data.get('above_200ma') else 'NO'}"
        ),
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
                            claude_r, gpt_r, signal_id, run_cost, mtd_cost,
                            entry_strategy=None, mtf_data=None, mtf_aligned=False,
                            news_risk=None, news_trade_blocked=False, risk_env=None):
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

    # Entry strategy (smart order type)
    if entry_strategy:
        order_type = entry_strategy.get("order_type", "market").upper()
        entry_px   = entry_strategy.get("entry_price", levels["entry"])
        pips_diff  = entry_strategy.get("pips_from_current", 0)
        rationale  = entry_strategy.get("entry_rationale", "")
        pips_str   = f"  ({pips_diff:+.0f} pips)" if pips_diff else ""
        print(f"  ORDER TYPE:  {order_type}{pips_str}")
        print(f"  ENTRY:       {entry_px:.5f}  [{rationale}]")
    else:
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
    if mtf_data:
        w_trend = (mtf_data.get("weekly_trend") or "unknown").upper()
        h4_trend = (mtf_data.get("h4_trend") or "unknown").upper()
        mtf_bias = mtf_data.get("mtf_bias", "NEUTRAL")
        aligned_str = "YES -- paper trade placed" if (mtf_aligned and not news_trade_blocked) else "NO -- paper trade SKIPPED"
        print(f"  Weekly trend: {w_trend}  |  4H trend: {h4_trend}")
        print(f"  MTF bias:     {mtf_bias}  |  Aligned: {aligned_str}")
        print("-" * 50)
    if news_risk and news_risk.get("risk_level") != "clear":
        rl = news_risk.get("risk_level", "").upper()
        warning = news_risk.get("warning_message", "")
        blocked_str = " *** PAPER TRADE BLOCKED ***" if news_trade_blocked else ""
        print(f"  NEWS RISK:    {rl}{blocked_str}")
        if warning:
            print(f"  {warning}")
        print("-" * 50)
    if risk_env and risk_env.get("vix_current") is not None:
        vix = risk_env.get("vix_current")
        vix_lv = risk_env.get("vix_level", "")
        vix_sig = risk_env.get("vix_signal", "")
        eur_trend = risk_env.get("eurusd_trend", "unknown")
        print(f"  VIX:          {vix:.1f} ({vix_lv}) -- {vix_sig}")
        print(f"  EURUSD:       {eur_trend.upper()}")
        if risk_env.get("risk_notes"):
            print(f"  {risk_env['risk_notes']}")
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
    mtf_data = None
    scorecard = None
    try:
        engine_c = ConfluenceEngine()
        price_data_conf = engine_c.fetch_price_data()
        market_data_conf = engine_c.fetch_market_data()
        log.info("Confluence data fetched OK")
    except Exception as e:
        log.error(f"Confluence data fetch failed (continuing without): {e}")

    # --- Multi-timeframe analysis ---
    try:
        if engine_c:
            mtf_data = engine_c.fetch_multi_timeframe()
            log.info(f"MTF bias: {mtf_data.get('mtf_bias')} -- {mtf_data.get('mtf_notes')}")
    except Exception as e:
        log.error(f"MTF fetch failed (continuing without): {e}")

    # --- Risk environment (VIX + EURUSD) ---
    risk_env = None
    try:
        risk_env = engine_c.fetch_risk_environment()
        log.info(f"VIX: {risk_env.get('vix_current')} ({risk_env.get('vix_level')}) -- {risk_env.get('vix_signal')}")
        log.info(f"EURUSD trend: {risk_env.get('eurusd_trend')} RSI: {risk_env.get('eurusd_rsi')}")
    except Exception as e:
        log.error(f"Risk environment fetch failed (continuing without): {e}")

    # News risk assessment
    news_risk = None
    try:
        news_risk = assess_news_risk(analysis_date)
        log.info(f"News risk: {news_risk['risk_level']} ({len(news_risk['high_impact_today'])} high-impact today)")
        if news_risk['warning_message']:
            log.warning(f"NEWS RISK: {news_risk['warning_message']}")
    except Exception as e:
        log.error(f"News risk assessment failed (continuing): {e}")

    # --- Ensemble: run OpenRouter models (uses pre-fetched confluence data) ---
    ensemble_votes = []   # populated after ctx is built in Step 3

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

    # MTF data is intentionally NOT injected into the agent prompts.
    # Telling models "weekly trend: UP" anchors their daily analysis toward BUY
    # just as effectively as sending the derived bias label directly.
    # Models form independent views from daily data only.
    # MTF is used by the system as a post-consensus filter to gate paper trades.

    # --- Ensemble: run OpenRouter models (now ctx is fully built) ---
    try:
        ensemble_votes = run_ensemble(price_data_conf, market_data_conf, context_data=ctx)
        if ensemble_votes:
            log.info(f"Ensemble: {len(ensemble_votes)} models voted")
    except Exception as e:
        log.error(f"Ensemble failed (continuing without): {e}")

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
            news_risk=news_risk,
            risk_env=risk_env,
        )
        log.info(f"Confluence score: {scorecard['confluence_pct']}% grade={scorecard['grade']}")
    except Exception as e:
        log.error(f"Confluence scoring failed (continuing without): {e}")
        scorecard = None

    # Step 7: Trade levels + smart entry strategy
    levels = calculate_trade_levels(price_data, combined["signal"])

    # Calculate smart entry (limit/stop/market order type and price)
    entry_strategy = None
    try:
        entry_strategy = engine_c.calculate_entry_strategy(price_data_conf, combined["signal"])
        # Override entry price in levels with smart entry
        # Then recalculate SL/TP from the new entry so they remain
        # on the correct side of price (SL below entry for BUY etc.)
        if entry_strategy and entry_strategy["order_type"] != "market":
            limit_entry = entry_strategy["entry_price"]
            levels["entry"] = limit_entry
            # Recalculate from limit entry to keep SL/TP consistent
            atr = price_data.get("atr_14") or (limit_entry * 0.005)
            if not atr:
                atr = limit_entry * 0.005
            atr = float(atr)
            sig = combined["signal"]
            min_stop = atr * 0.5
            if sig == "BUY":
                support = price_data.get("nearest_support") or (limit_entry - atr * 2)
                sl = max(float(support) - atr * 0.1, limit_entry - atr * 0.6)
                sl = min(sl, limit_entry - min_stop)
                dist = limit_entry - sl
                levels["stop_loss"]   = round(sl, 5)
                levels["take_profit"] = round(limit_entry + dist * 2.0, 5)
            elif sig == "SELL":
                resistance = price_data.get("nearest_resistance") or (limit_entry + atr * 2)
                sl = min(float(resistance) + atr * 0.1, limit_entry + atr * 0.6)
                sl = max(sl, limit_entry + min_stop)
                dist = sl - limit_entry
                levels["stop_loss"]   = round(sl, 5)
                levels["take_profit"] = round(limit_entry - dist * 2.0, 5)
            pips_stop = abs(limit_entry - levels["stop_loss"]) * 10000
            pips_tp   = abs(limit_entry - levels["take_profit"]) * 10000
            levels["pips_stop"]   = round(pips_stop, 1)
            levels["pips_tp"]     = round(pips_tp, 1)
            levels["risk_reward"] = round(pips_tp / pips_stop, 2) if pips_stop > 0 else 0.0
        log.info(
            f"Entry strategy: {entry_strategy['order_type']} at {entry_strategy['entry_price']:.5f} "
            f"({entry_strategy['pips_from_current']:+.0f} pips) -- {entry_strategy['entry_rationale']}"
        )
    except Exception as e:
        log.error(f"Entry strategy calculation failed (continuing): {e}")

    # Determine news trade block (binary or high impact = block paper trade)
    news_trade_blocked = False
    if news_risk and news_risk.get("risk_level") in ("binary", "high") and combined["signal"] != "HOLD":
        news_trade_blocked = True

    # Determine MTF alignment with daily signal
    mtf_aligned = False
    if mtf_data:
        mtf_bias = mtf_data.get("mtf_bias", "NEUTRAL")
        daily_signal = combined["signal"]
        mtf_aligned = (
            (daily_signal == "BUY"  and mtf_bias == "BUY")  or
            (daily_signal == "SELL" and mtf_bias == "SELL") or
            (daily_signal == "HOLD")
        )
        if mtf_aligned:
            log.info(f"MTF ALIGNED: daily={daily_signal} mtf={mtf_bias} -- confirmed signal")
        else:
            log.info(f"MTF CONFLICT: daily={daily_signal} vs mtf={mtf_bias} -- daily signal only, no paper trade")

    # Step 8: Save signal
    signal_id = save_signal(
        combined, claude_result, gpt_result, levels, price_data, analysis_date,
        scorecard=scorecard,
        confluence_price_data=price_data_conf,
        confluence_market_data=market_data_conf,
        entry_strategy=entry_strategy,
        mtf_data=mtf_data,
        mtf_aligned=mtf_aligned,
        news_risk=news_risk,
        news_trade_blocked=news_trade_blocked,
        risk_env=risk_env,
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

    # Step 8c: Open virtual paper trade (only when MTF aligned and no news block)
    vtrade = None
    trade_skip_reason = None

    # Check MTF conflict
    if mtf_data and not mtf_aligned and combined["signal"] != "HOLD":
        trade_skip_reason = (
            f"MTF conflict: daily={combined['signal']} vs MTF bias={mtf_data.get('mtf_bias')} "
            f"({mtf_data.get('mtf_notes', '')})"
        )
        log.info(f"Paper trade SKIPPED -- {trade_skip_reason}")

    # Check news risk -- override trade even if MTF aligned
    if news_trade_blocked:
        news_reason = f"News blocked: {news_risk.get('warning_message', 'high-impact event today')}"
        if not trade_skip_reason:
            trade_skip_reason = news_reason
        log.warning(f"Paper trade BLOCKED by news risk -- {news_reason}")

    # Entry quality check -- never skips, only adjusts position size.
    # Grades entry as clean / caution / extended based on RSI stretch.
    # Stored as metadata so data later shows whether extended entries perform worse.
    entry_quality       = "clean"
    entry_quality_note  = ""
    base_risk_pct       = scorecard["position_size_pct"] if scorecard else 0.5
    adjusted_risk_pct   = base_risk_pct

    if combined["signal"] in ("BUY", "SELL") and not trade_skip_reason:
        rsi = price_data.get("rsi_value") if price_data else None
        sig = combined["signal"]
        if rsi is not None:
            rsi = float(rsi)
            if sig == "BUY":
                if rsi > 70:
                    entry_quality      = "extended"
                    entry_quality_note = f"RSI {rsi:.0f} -- overbought, position halved"
                    adjusted_risk_pct  = round(base_risk_pct * 0.5, 2)
                elif rsi > 65:
                    entry_quality      = "caution"
                    entry_quality_note = f"RSI {rsi:.0f} -- elevated, position reduced 25%"
                    adjusted_risk_pct  = round(base_risk_pct * 0.75, 2)
                else:
                    entry_quality_note = f"RSI {rsi:.0f} -- clean entry"
            elif sig == "SELL":
                if rsi < 30:
                    entry_quality      = "extended"
                    entry_quality_note = f"RSI {rsi:.0f} -- oversold, position halved"
                    adjusted_risk_pct  = round(base_risk_pct * 0.5, 2)
                elif rsi < 35:
                    entry_quality      = "caution"
                    entry_quality_note = f"RSI {rsi:.0f} -- extended, position reduced 25%"
                    adjusted_risk_pct  = round(base_risk_pct * 0.75, 2)
                else:
                    entry_quality_note = f"RSI {rsi:.0f} -- clean entry"

        if entry_quality != "clean":
            log.info("Entry quality: %s -- %s (risk %.2f%% -> %.2f%%)",
                     entry_quality, entry_quality_note, base_risk_pct, adjusted_risk_pct)
        else:
            log.info("Entry quality: clean -- %s", entry_quality_note)

    try:
        from tracker.virtual_account import open_trade
        from tracker.database import get_session, initialise_virtual_account
        vsession = get_session()
        initialise_virtual_account(vsession)
        trade_signal = {
            "id":            signal_id,
            "signal":        combined["signal"] if not trade_skip_reason else "HOLD",
            "entry_price":   levels.get("entry"),
            "stop_loss":     levels.get("stop_loss"),
            "take_profit":   levels.get("take_profit"),
            "analysis_date": analysis_date,
            "ai_confidence": combined["confidence"],
            "providers_agree": combined["providers_agree"],
        }
        ot           = entry_strategy.get("order_type", "market") if entry_strategy else "market"
        lim_price    = entry_strategy.get("entry_price")          if entry_strategy else None
        exp_bars     = entry_strategy.get("expires_bars", 1)      if entry_strategy else 1

        vtrade = open_trade(
            vsession,
            trade_signal,
            confluence_grade=scorecard["grade"] if scorecard else None,
            risk_pct=adjusted_risk_pct,
            order_type=ot,
            limit_price=lim_price,
            expires_bars=exp_bars,
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
                           claude_result, gpt_result, signal_id, run_cost, mtd_cost,
                           entry_strategy=entry_strategy,
                           mtf_data=mtf_data, mtf_aligned=mtf_aligned,
                           news_risk=news_risk, news_trade_blocked=news_trade_blocked,
                           risk_env=risk_env)

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

    # News risk warning block
    if news_risk and news_risk.get("warning_message"):
        print("!" * 50)
        print(f"  NEWS: {news_risk['warning_message']}")
        if news_risk.get("binary_event_today"):
            print("  RECOMMENDATION: DO NOT TRADE TODAY")
        print("!" * 50)

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
    if vtrade and vtrade.status in ("open", "pending_entry"):
        balance  = float(vtrade.opening_balance)
        risk_gbp = float(vtrade.risk_gbp)
        print(f"  PAPER TRADE: GBP {risk_gbp:.2f} at risk ({vtrade.risk_pct}% of GBP {balance:.2f})")
        print(f"  Spread cost: GBP {float(vtrade.spread_cost_gbp):.4f}")
        if entry_quality != "clean":
            print(f"  ENTRY QUALITY: {entry_quality.upper()} -- {entry_quality_note}")
        else:
            print(f"  ENTRY QUALITY: CLEAN -- {entry_quality_note}")
        print("-" * 50)

    # Step 13: Send email alert
    send_signal_email(analysis_date, combined, levels, scorecard, run_cost)

    log.info(f"=== run_daily.py complete. Signal ID: {signal_id} ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
