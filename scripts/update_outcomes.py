"""
scripts/update_outcomes.py -- Resolve open signals using hourly price data.

Runs daily after run_daily.py. Checks ALL unresolved signals, not just
signals from a fixed date in the past.

Resolution logic:
- Downloads hourly OHLCV from signal date to today
- Walks candles in order -- first level hit (TP or SL) wins
- Marks resolved immediately when hit
- Expires after MAX_TRADING_DAYS if neither level is touched

Using hourly data (vs daily) correctly sequences TP/SL hits --
daily candles cannot tell you which was hit first when both levels
are breached within the same day.
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

MAX_TRADING_DAYS = 5   # expire signal if neither TP nor SL hit within this many trading days


def trading_days_between(start: date, end: date) -> int:
    """Count weekday trading days between two dates (exclusive of start, inclusive of end)."""
    count = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return count


def fetch_hourly_data(pair: str, start: date, end: date):
    """
    Fetch hourly OHLCV from start to end (inclusive).
    yfinance 1h interval supports up to 730 days history.
    Returns DataFrame or None.
    """
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download(
            pair,
            start=str(start),
            end=str(end + timedelta(days=1)),
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.dropna(subset=["High", "Low", "Close"])
        return df if not df.empty else None
    except Exception as e:
        log.error("Hourly price fetch failed for %s %s-%s: %s", pair, start, end, e)
        return None


def resolve_signal(signal_row, df) -> dict:
    """
    Walk hourly candles in chronological order.
    First level hit (TP or SL) determines the outcome.
    Returns outcome dict or None if price data unavailable.
    """
    if df is None or df.empty:
        return None

    entry     = float(signal_row.entry_price or 0)
    sl        = float(signal_row.stop_loss   or 0)
    tp        = float(signal_row.take_profit or 0)
    direction = (signal_row.signal or "HOLD").upper()

    if direction == "HOLD" or entry == 0:
        # HOLD signals -- just record directional move at close
        close_price = float(df["Close"].iloc[-1])
        return {
            "actual_close_price":    close_price,
            "pips_moved":            0.0,
            "signal_correct":        True,
            "directionally_correct": True,
            "would_have_hit_tp":     False,
            "would_have_hit_sl":     False,
            "max_favorable_pips":    0.0,
            "max_adverse_pips":      0.0,
            "outcome_date":          df.index[-1].date() if hasattr(df.index[-1], "date") else date.today(),
            "outcome_type":          "expired",
        }

    hit_tp = False
    hit_sl = False
    max_fav = 0.0
    max_adv = 0.0
    resolution_date = None

    for ts, row in df.iterrows():
        h = float(row["High"])
        l = float(row["Low"])

        if direction == "BUY":
            fav = (h - entry) * 10000
            adv = (entry - l) * 10000
            # Check SL first then TP -- on same candle, assume worst case
            if l <= sl:
                hit_sl = True
                resolution_date = ts.date() if hasattr(ts, "date") else date.today()
                break
            if h >= tp:
                hit_tp = True
                resolution_date = ts.date() if hasattr(ts, "date") else date.today()
                break

        elif direction == "SELL":
            fav = (entry - l) * 10000
            adv = (h - entry) * 10000
            # Check SL first then TP -- on same candle, assume worst case
            if h >= sl:
                hit_sl = True
                resolution_date = ts.date() if hasattr(ts, "date") else date.today()
                break
            if l <= tp:
                hit_tp = True
                resolution_date = ts.date() if hasattr(ts, "date") else date.today()
                break
        else:
            fav, adv = 0.0, 0.0

        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)

    close_price = float(df["Close"].iloc[-1])

    if direction == "BUY":
        pips_moved = (close_price - entry) * 10000
    elif direction == "SELL":
        pips_moved = (entry - close_price) * 10000
    else:
        pips_moved = 0.0

    if hit_tp:
        outcome_type = "tp_hit"
        signal_correct = True
        directionally_correct = True
        pips_moved = abs(tp - entry) * 10000  # actual pips gained
    elif hit_sl:
        outcome_type = "sl_hit"
        signal_correct = False
        directionally_correct = False
        pips_moved = -abs(entry - sl) * 10000  # actual pips lost
    else:
        # Neither hit -- expired at max days
        # Not a win or a loss -- TP/SL were never reached.
        # signal_correct=None so analytics exclude it from win rate.
        # pips_moved kept for reference (directional drift only).
        outcome_type = "expired"
        directionally_correct = (pips_moved > 0)
        signal_correct = None

    if resolution_date is None:
        resolution_date = df.index[-1].date() if hasattr(df.index[-1], "date") else date.today()

    return {
        "actual_close_price":    close_price,
        "pips_moved":            round(pips_moved, 1),
        "signal_correct":        signal_correct,
        "directionally_correct": directionally_correct,
        "would_have_hit_tp":     hit_tp,
        "would_have_hit_sl":     hit_sl,
        "max_favorable_pips":    round(max_fav, 1),
        "max_adverse_pips":      round(max_adv, 1),
        "outcome_date":          resolution_date,
        "outcome_type":          outcome_type,
    }


def main():
    log.info("=== update_outcomes.py started: %s ===", datetime.now())

    from tracker.database import get_session, Signal, Outcome

    session = get_session()

    # Step 1: Check if any pending limit/stop orders filled today
    try:
        from tracker.virtual_account import fill_pending_entries
        filled, cancelled = fill_pending_entries(session)
        if filled or cancelled:
            log.info("Limit orders: %d filled, %d cancelled today", filled, cancelled)
            if filled:
                print(f"  Limit orders filled today: {filled}")
            if cancelled:
                print(f"  Limit orders cancelled (unfilled, expired): {cancelled}")
    except Exception as e:
        log.error("fill_pending_entries failed (non-critical): %s", e)

    try:
        # Find ALL unresolved signals that are at least 1 trading day old
        yesterday = date.today() - timedelta(days=1)
        resolved_ids = [r.signal_id for r in session.query(Outcome.signal_id).all()]
        pending = session.query(Signal).filter(
            Signal.analysis_date <= yesterday,
            Signal.id.notin_(resolved_ids) if resolved_ids else True,
            Signal.signal.in_(["BUY", "SELL", "HOLD"]),
        ).order_by(Signal.analysis_date).all()
    except Exception as e:
        log.error("Query failed: %s", e)
        session.close()
        sys.exit(1)

    if not pending:
        log.info("No pending signals to resolve.")
        print("No pending signals to resolve.")
        session.close()
        return

    log.info("Found %d pending signal(s) to check.", len(pending))
    wins = losses = expired_count = skipped = 0
    total_pips = 0.0

    for sig in pending:
        pair            = sig.pair or "GBPUSD=X"
        signal_date     = sig.analysis_date
        trading_days    = trading_days_between(signal_date, date.today())

        # Fetch hourly data from signal date to today
        df = fetch_hourly_data(pair, signal_date, date.today())

        if df is None:
            # Fall back to daily if hourly unavailable
            log.warning("Hourly data unavailable for %s, trying daily.", sig.id)
            try:
                import yfinance as yf
                df = yf.download(pair, start=str(signal_date),
                                 end=str(date.today() + timedelta(days=1)),
                                 progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    import pandas as pd
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                else:
                    df = None
            except Exception:
                df = None

        if df is None:
            log.warning("No price data for signal id=%s -- skipping.", sig.id)
            skipped += 1
            continue

        # Only resolve if TP/SL hit OR we've exceeded MAX_TRADING_DAYS
        outcome = resolve_signal(sig, df)
        if outcome is None:
            skipped += 1
            continue

        tp_hit = outcome["would_have_hit_tp"]
        sl_hit = outcome["would_have_hit_sl"]
        expired = outcome["outcome_type"] == "expired"

        # Skip if still live and within max days
        if expired and trading_days < MAX_TRADING_DAYS:
            log.info("Signal id=%s still live (%d trading days) -- checking tomorrow.", sig.id, trading_days)
            continue

        try:
            row = Outcome(
                signal_id             = sig.id,
                outcome_date          = outcome["outcome_date"],
                actual_close_price    = outcome["actual_close_price"],
                pips_moved            = outcome["pips_moved"],
                signal_correct        = outcome["signal_correct"],
                directionally_correct = outcome["directionally_correct"],
                would_have_hit_tp     = tp_hit,
                would_have_hit_sl     = sl_hit,
                max_favorable_pips    = outcome["max_favorable_pips"],
                max_adverse_pips      = outcome["max_adverse_pips"],
                notes                 = (
                    f"Auto-resolved {date.today()} via "
                    f"{'hourly' if 'h' in str(df.index.freq or '') else 'daily'} data. "
                    f"Type: {outcome['outcome_type']}. "
                    f"Trading days open: {trading_days}."
                ),
            )
            session.add(row)
            session.commit()

            pips = outcome["pips_moved"]
            total_pips += pips
            result_str = outcome["outcome_type"].upper()

            if tp_hit:
                wins += 1
            elif sl_hit:
                losses += 1
            else:
                expired_count += 1

            log.info(
                "Outcome saved: signal id=%s date=%s %s correct=%s pips=%.1f",
                sig.id, signal_date, result_str,
                outcome["signal_correct"], pips,
            )
            print(f"  Signal {sig.id} ({signal_date} {sig.signal}): "
                  f"{result_str} | {pips:+.1f} pips")

            # Update model vote accuracy
            try:
                from tracker.database import ModelVote
                votes = session.query(ModelVote).filter_by(signal_id=sig.id).all()
                for mv in votes:
                    if mv.signal and mv.signal.upper() not in ("HOLD", ""):
                        mv.was_correct = (mv.signal.upper() == sig.signal.upper()) == outcome["signal_correct"]
                        mv.pips_result = outcome["pips_moved"]
                session.commit()
            except Exception as e:
                log.error("Could not update model vote accuracy: %s", e)

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
                log.error("Virtual trade close failed (non-critical): %s", e)

        except Exception as e:
            session.rollback()
            log.error("Failed to save outcome for signal id=%s: %s", sig.id, e)
            skipped += 1

    session.close()

    pips_sign = "+" if total_pips >= 0 else ""
    print(f"\nResolved: {wins} TP hits, {losses} SL hits, {expired_count} expired, {skipped} skipped")
    print(f"Net pips this batch: {pips_sign}{total_pips:.1f}")
    log.info("=== update_outcomes.py complete: %dTP %dSL %dexp %dskip ===",
             wins, losses, expired_count, skipped)


if __name__ == "__main__":
    main()
