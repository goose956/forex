"""
scripts/update_outcomes.py -- Close out signals from 5 trading days ago.

Runs daily after run_daily.py.
Fetches actual price data for the resolution period and determines
whether each open signal hit TP, hit SL, or expired.

Usage:
    python scripts/update_outcomes.py
"""

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
LOG_FILE = LOG_DIR / "outcomes.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("update_outcomes")


def trading_days_ago(n: int) -> date:
    """Return the date that was n trading days ago (skipping weekends)."""
    d = date.today()
    count = 0
    while count < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


def fetch_period_data(pair: str, start: date, end: date):
    """Fetch OHLCV for a date range. Returns DataFrame or None."""
    try:
        import yfinance as yf
        df = yf.download(pair, start=str(start), end=str(end + timedelta(days=1)),
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df if not df.empty else None
    except Exception as e:
        log.error(f"Price fetch failed for {pair} {start}-{end}: {e}")
        return None


def resolve_signal(signal_row, df) -> dict:
    """
    Given a Signal ORM row and price DataFrame covering the 5-day period,
    determine the outcome.
    """
    if df is None or df.empty:
        return None

    entry = float(signal_row.entry_price or 0)
    sl    = float(signal_row.stop_loss   or 0)
    tp    = float(signal_row.take_profit or 0)
    direction = (signal_row.signal or "HOLD").upper()

    hit_tp = False
    hit_sl = False
    max_fav = 0.0
    max_adv = 0.0

    for _, row in df.iterrows():
        h = float(row["High"])
        l = float(row["Low"])

        if direction == "BUY":
            fav = (h - entry) * 10000
            adv = (entry - l) * 10000
            if h >= tp and not hit_sl:
                hit_tp = True
            if l <= sl and not hit_tp:
                hit_sl = True
        elif direction == "SELL":
            fav = (entry - l) * 10000
            adv = (h - entry) * 10000
            if l <= tp and not hit_sl:
                hit_tp = True
            if h >= sl and not hit_tp:
                hit_sl = True
        else:
            fav, adv = 0.0, 0.0

        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)

        if hit_tp or hit_sl:
            break

    close_price = float(df["Close"].iloc[-1])

    if direction == "BUY":
        pips_moved = (close_price - entry) * 10000
    elif direction == "SELL":
        pips_moved = (entry - close_price) * 10000
    else:
        pips_moved = 0.0

    directionally_correct = (pips_moved > 0)
    signal_correct        = hit_tp or (directionally_correct and not hit_sl)

    return {
        "actual_close_price":    close_price,
        "pips_moved":            round(pips_moved, 1),
        "signal_correct":        signal_correct,
        "directionally_correct": directionally_correct,
        "would_have_hit_tp":     hit_tp,
        "would_have_hit_sl":     hit_sl,
        "max_favorable_pips":    round(max_fav, 1),
        "max_adverse_pips":      round(max_adv, 1),
        "outcome_date":          df.index[-1].date(),
    }


def main():
    log.info(f"=== update_outcomes.py started: {datetime.now()} ===")

    from tracker.database import get_session, Signal, Outcome

    target_date = trading_days_ago(5)
    log.info(f"Resolving signals from: {target_date}")

    session = get_session()
    try:
        # Find unresolved signals from target date
        resolved_ids = [r.signal_id for r in session.query(Outcome.signal_id).all()]
        pending = session.query(Signal).filter(
            Signal.analysis_date == target_date,
            Signal.id.notin_(resolved_ids) if resolved_ids else True,
        ).all()
    except Exception as e:
        log.error(f"Query failed: {e}")
        session.close()
        sys.exit(1)

    if not pending:
        log.info(f"No pending signals from {target_date}.")
        print(f"No pending signals from {target_date}.")
        session.close()
        return

    log.info(f"Found {len(pending)} pending signal(s) from {target_date}.")
    wins = losses = skipped = 0
    total_pips = 0.0

    for sig in pending:
        pair = sig.pair or "GBPUSD=X"
        df   = fetch_period_data(pair, target_date, date.today())
        outcome = resolve_signal(sig, df)

        if outcome is None:
            log.warning(f"Could not resolve signal id={sig.id} -- skipping.")
            skipped += 1
            continue

        try:
            row = Outcome(
                signal_id             = sig.id,
                outcome_date          = outcome["outcome_date"],
                actual_close_price    = outcome["actual_close_price"],
                pips_moved            = outcome["pips_moved"],
                signal_correct        = outcome["signal_correct"],
                directionally_correct = outcome["directionally_correct"],
                would_have_hit_tp     = outcome["would_have_hit_tp"],
                would_have_hit_sl     = outcome["would_have_hit_sl"],
                max_favorable_pips    = outcome["max_favorable_pips"],
                max_adverse_pips      = outcome["max_adverse_pips"],
                notes                 = f"Auto-resolved on {date.today()}",
            )
            session.add(row)
            session.commit()

            if outcome["signal_correct"]:
                wins += 1
            else:
                losses += 1
            total_pips += outcome["pips_moved"]

            log.info(f"Outcome saved: signal id={sig.id} "
                     f"correct={outcome['signal_correct']} "
                     f"pips={outcome['pips_moved']}")

            # Close virtual paper trade
            try:
                from tracker.virtual_account import close_trade
                vtrade = close_trade(session, sig.id, outcome)
                if vtrade:
                    pnl = float(vtrade.net_pnl_gbp or 0)
                    bal = float(vtrade.closing_balance or 0)
                    print(f"  Paper trade: {vtrade.outcome_type} | "
                          f"P&L: GBP {pnl:+.2f} | Balance: GBP {bal:.2f}")
            except Exception as e:
                log.error(f"Virtual trade close failed (non-critical): {e}")
        except Exception as e:
            session.rollback()
            log.error(f"Failed to save outcome for signal id={sig.id}: {e}")
            skipped += 1

    session.close()

    pips_sign = "+" if total_pips >= 0 else ""
    print(f"\nOutcomes recorded: {len(pending)} signals from {target_date}")
    print(f"Results: {wins} wins, {losses} losses, {skipped} skipped")
    print(f"Paper pips this batch: {pips_sign}{total_pips:.1f} pips")
    log.info(f"=== update_outcomes.py complete: {wins}W {losses}L {skipped}skip ===")


if __name__ == "__main__":
    main()
