"""
scripts/weekly_report.py -- Generate weekly performance report.

Pulls last 7 days of signals and outcomes, calculates stats,
sends to Claude API for analysis, and saves the report.

Usage:
    python scripts/weekly_report.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR  = ROOT / "tracker" / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weekly.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("weekly_report")


def fetch_week_data():
    """Pull signals + outcomes from the last 7 days."""
    from tracker.database import get_session, Signal, Outcome
    session = get_session()

    week_start = date.today() - timedelta(days=7)
    signals    = session.query(Signal).filter(Signal.analysis_date >= week_start).all()

    outcomes_map = {}
    for sig in signals:
        outcome = session.query(Outcome).filter(Outcome.signal_id == sig.id).first()
        if outcome:
            outcomes_map[sig.id] = outcome

    session.close()
    return signals, outcomes_map


def calculate_stats(signals, outcomes_map):
    """Calculate weekly performance statistics."""
    stats = {
        "total_signals":        len(signals),
        "buy_signals":          sum(1 for s in signals if s.signal == "BUY"),
        "sell_signals":         sum(1 for s in signals if s.signal == "SELL"),
        "hold_signals":         sum(1 for s in signals if s.signal == "HOLD"),
        "outcomes_recorded":    len(outcomes_map),
        "win_rate":             0.0,
        "avg_conf_winners":     0.0,
        "avg_conf_losers":      0.0,
        "total_paper_pips":     0.0,
        "agreement_win_rate":   0.0,
        "disagreement_win_rate": 0.0,
        "best_signal_id":       None,
        "worst_signal_id":      None,
    }

    if not outcomes_map:
        return stats

    wins        = [o for o in outcomes_map.values() if o.signal_correct]
    losses      = [o for o in outcomes_map.values() if not o.signal_correct]
    stats["win_rate"] = round(len(wins) / len(outcomes_map) * 100, 1)

    winner_ids  = {o.signal_id for o in wins}
    loser_ids   = {o.signal_id for o in losses}

    win_confs   = [s.confidence for s in signals if s.id in winner_ids and s.confidence]
    loss_confs  = [s.confidence for s in signals if s.id in loser_ids  and s.confidence]
    stats["avg_conf_winners"] = round(sum(win_confs) / len(win_confs), 1) if win_confs else 0
    stats["avg_conf_losers"]  = round(sum(loss_confs) / len(loss_confs), 1) if loss_confs else 0

    all_pips = [float(o.pips_moved or 0) for o in outcomes_map.values()]
    stats["total_paper_pips"] = round(sum(all_pips), 1)

    # Agreement vs disagreement win rates
    agree_sigs  = [s for s in signals if s.providers_agree  and s.id in outcomes_map]
    disagree_s  = [s for s in signals if not s.providers_agree and s.id in outcomes_map]

    def wr(sigs):
        w = sum(1 for s in sigs if outcomes_map.get(s.id) and outcomes_map[s.id].signal_correct)
        return round(w / len(sigs) * 100, 1) if sigs else 0.0

    stats["agreement_win_rate"]    = wr(agree_sigs)
    stats["disagreement_win_rate"] = wr(disagree_s)

    # Best / worst
    by_pips = sorted(outcomes_map.items(), key=lambda x: float(x[1].pips_moved or 0))
    if by_pips:
        stats["worst_signal_id"] = by_pips[0][0]
        stats["best_signal_id"]  = by_pips[-1][0]

    return stats


def build_data_summary(signals, outcomes_map, stats) -> str:
    """Format structured data for the Claude API prompt."""
    lines = [
        f"Week ending: {date.today()}",
        f"Total signals: {stats['total_signals']}",
        f"  BUY: {stats['buy_signals']}  SELL: {stats['sell_signals']}  HOLD: {stats['hold_signals']}",
        f"Outcomes recorded: {stats['outcomes_recorded']}",
        f"Win rate: {stats['win_rate']}%",
        f"Avg confidence (winners): {stats['avg_conf_winners']}",
        f"Avg confidence (losers):  {stats['avg_conf_losers']}",
        f"Total paper pips: {stats['total_paper_pips']}",
        f"Agreement win rate:    {stats['agreement_win_rate']}%",
        f"Disagreement win rate: {stats['disagreement_win_rate']}%",
        "",
        "Individual signals this week:",
    ]
    for sig in signals:
        o = outcomes_map.get(sig.id)
        outcome_str = f"{'WIN' if o.signal_correct else 'LOSS'} ({o.pips_moved:+.0f} pips)" if o else "PENDING"
        lines.append(
            f"  {sig.analysis_date} | {sig.signal} | conf={sig.confidence}/10 | "
            f"claude={sig.claude_signal} gpt={sig.gpt_signal} | "
            f"agree={sig.providers_agree} | outcome={outcome_str}"
        )
    return "\n".join(lines)


def get_ai_analysis(data_summary: str) -> str:
    """Send data to Claude and get a written weekly analysis."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""You are analysing the historical performance of an AI forex signal system trading GBP/USD.
Here is this week's signal data with outcomes:

{data_summary}

Please provide a weekly performance report covering:
1. Performance summary: was this a good week?
2. Signal quality: what made the best signals stand out? What made poor signals poor?
3. Market conditions: how did the market behave and did the system adapt?
4. Provider analysis: did Claude and GPT perform differently this week?
5. Emerging patterns: any trends developing in the data so far?
6. Next week outlook: based on current conditions, what should I watch for?
7. System health: is this system developing positive expectancy? Be honest.

Write in plain English. Be direct. Maximum 500 words."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def save_report(stats: dict, ai_analysis: str, week_ending: date):
    """Save weekly stats to database."""
    from tracker.database import get_session, WeeklyReport
    session = get_session()
    try:
        row = WeeklyReport(
            week_ending             = week_ending,
            total_signals           = stats["total_signals"],
            buy_signals             = stats["buy_signals"],
            sell_signals            = stats["sell_signals"],
            hold_signals            = stats["hold_signals"],
            outcomes_recorded       = stats["outcomes_recorded"],
            win_rate                = stats["win_rate"],
            avg_confidence_winners  = stats["avg_conf_winners"],
            avg_confidence_losers   = stats["avg_conf_losers"],
            total_paper_pips        = stats["total_paper_pips"],
            agreement_win_rate      = stats["agreement_win_rate"],
            disagreement_win_rate   = stats["disagreement_win_rate"],
            best_signal_id          = stats["best_signal_id"],
            worst_signal_id         = stats["worst_signal_id"],
            ai_performance_analysis = ai_analysis,
        )
        session.add(row)
        session.commit()
        session.close()
        log.info("Weekly report saved to database.")
    except Exception as e:
        session.rollback()
        session.close()
        log.error(f"Failed to save weekly report: {e}")


def write_report_file(stats: dict, ai_analysis: str, data_summary: str) -> Path:
    """Write weekly report to file."""
    report_dir = ROOT / "tracker" / "reports" / "weekly"
    report_dir.mkdir(parents=True, exist_ok=True)
    week_end   = date.today()
    filepath   = report_dir / f"week-ending-{week_end}.txt"

    lines = [
        "=" * 70,
        f"FOREX SIGNAL TRACKER -- WEEKLY PERFORMANCE REPORT",
        f"Week ending: {week_end}  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "-- STATISTICS --",
        data_summary,
        "",
        "-- AI PERFORMANCE ANALYSIS (Claude) --",
        "",
        ai_analysis,
        "",
        "=" * 70,
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"Weekly report saved: {filepath}")
    return filepath


def main():
    log.info(f"=== weekly_report.py started: {datetime.now()} ===")

    from tracker.database import test_connection, create_tables
    if not test_connection():
        log.error("DB connection failed.")
        sys.exit(1)
    create_tables()

    signals, outcomes_map = fetch_week_data()
    stats        = calculate_stats(signals, outcomes_map)
    data_summary = build_data_summary(signals, outcomes_map, stats)

    print("\n-- Weekly Stats --")
    print(f"  Signals:    {stats['total_signals']}  (B:{stats['buy_signals']} S:{stats['sell_signals']} H:{stats['hold_signals']})")
    print(f"  Win rate:   {stats['win_rate']}%  ({stats['outcomes_recorded']} outcomes)")
    print(f"  Paper pips: {stats['total_paper_pips']:+.1f}")
    print(f"  Agree W%:   {stats['agreement_win_rate']}%   Disagree W%: {stats['disagreement_win_rate']}%")

    print("\nRequesting AI analysis from Claude...")
    try:
        ai_analysis = get_ai_analysis(data_summary)
    except Exception as e:
        log.error(f"AI analysis failed: {e}")
        ai_analysis = f"AI analysis unavailable: {e}"

    week_end  = date.today()
    save_report(stats, ai_analysis, week_end)
    filepath  = write_report_file(stats, ai_analysis, data_summary)

    print("\n-- AI Analysis --")
    print(ai_analysis)
    print(f"\nFull report: {filepath}")
    log.info("=== weekly_report.py complete ===")


if __name__ == "__main__":
    main()
