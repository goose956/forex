"""
tracker/virtual_account.py -- Paper trading account management.

Opens and closes virtual trades based on signals and outcomes.
All amounts in GBP. Starting balance: 1000 GBP.
Spread default: 1.5 pips (typical GBPUSD retail spread).
"""

import logging
log = logging.getLogger("virtual_account")

STARTING_BALANCE   = 1000.00
DEFAULT_SPREAD_PIPS = 1.5


def open_trade(session, signal, confluence_grade=None, risk_pct=None,
               order_type="market", limit_price=None, expires_bars=1):
    """
    Open a virtual trade for the given signal.

    signal: dict with keys: id, signal, entry_price, stop_loss, take_profit,
            analysis_date, ai_confidence, providers_agree
    risk_pct:     override risk % (if None, derive from confluence_grade)
    order_type:   'market' / 'limit' / 'stop'
    limit_price:  target fill price for limit/stop orders
    expires_bars: cancel unfilled limit/stop after this many trading days

    Market orders open immediately (status='open').
    Limit/stop orders wait for fill (status='pending_entry').
    Returns the VirtualTrade row or None if trade cannot be opened.
    """
    from tracker.database import VirtualTrade, get_virtual_balance

    direction = (signal.get("signal") or "HOLD").upper()

    # Grade C = shadow trade (tracked but balance unaffected)
    # Grade D = skipped entirely
    # Grade B and above = real trade
    grade_to_risk = {"A+": 1.5, "A": 1.0, "B": 0.5, "C": 0.25, "D": 0.0, None: 0.5}
    if risk_pct is None:
        risk_pct = grade_to_risk.get(confluence_grade, 0.5)

    entry = float(signal.get("entry_price") or 0)
    sl    = float(signal.get("stop_loss")   or 0)
    tp    = float(signal.get("take_profit") or 0)

    if entry == 0 or sl == 0 or tp == 0:
        log.warning(
            "open_trade: missing price levels for signal id=%s -- skipping",
            signal.get("id"),
        )
        return None

    sl_pips = abs(entry - sl) * 10000
    tp_pips = abs(entry - tp) * 10000

    if sl_pips < 1:
        log.warning(
            "open_trade: SL too tight (%.1f pips) -- skipping", sl_pips
        )
        return None

    # Check for conflicting open real trade (status='open')
    # Shadow trades don't block — they're separate tracking
    open_trade = session.query(VirtualTrade).filter_by(status="open").first()
    if open_trade:
        if open_trade.direction != direction:
            log.info(
                "open_trade: skipping signal_id=%s (%s) -- conflicts with open %s trade (signal_id=%s)",
                signal.get("id"), direction, open_trade.direction, open_trade.signal_id,
            )
            # Still record a row so we can see skipped conflicts in the dashboard
            skipped = VirtualTrade(
                signal_id        = signal["id"],
                opened_at        = signal.get("analysis_date"),
                direction        = direction,
                entry_price      = entry,
                stop_loss        = sl,
                take_profit      = tp,
                sl_pips          = round(sl_pips, 1),
                tp_pips          = round(tp_pips, 1),
                opening_balance  = get_virtual_balance(session),
                risk_pct         = 0,
                risk_gbp         = 0,
                value_per_pip    = 0,
                spread_pips      = DEFAULT_SPREAD_PIPS,
                spread_cost_gbp  = 0,
                status           = "skipped_conflict",
                confluence_grade = confluence_grade,
                ai_confidence    = signal.get("ai_confidence") or signal.get("confidence"),
                providers_agree  = signal.get("providers_agree"),
            )
            session.add(skipped)
            session.commit()
            return None
        else:
            log.info(
                "open_trade: already in %s trade (signal_id=%s) -- skipping same-direction duplicate",
                direction, open_trade.signal_id,
            )
            return None

    # Check for duplicate (already has a trade for this signal)
    existing = session.query(VirtualTrade).filter_by(signal_id=signal["id"]).first()
    if existing:
        log.info(
            "open_trade: virtual trade already exists for signal_id=%s -- skipping",
            signal["id"],
        )
        return existing

    opening_balance = get_virtual_balance(session)

    risk_gbp        = round(opening_balance * risk_pct / 100, 4)
    value_per_pip   = round(risk_gbp / sl_pips, 6) if sl_pips > 0 else 0
    spread_cost_gbp = round(DEFAULT_SPREAD_PIPS * value_per_pip, 4)

    # Determine trade status:
    # B+ grades = real trade (affects balance)
    # C grade   = shadow trade (tracked, balance unaffected)
    # D / HOLD  = skipped
    # Limit/stop orders = pending_entry until filled
    if confluence_grade == "C":
        trade_status = "shadow_pending" if order_type in ("limit", "stop") else "shadow"
    elif direction == "HOLD" or risk_pct == 0 or confluence_grade == "D":
        trade_status = "skipped"
    elif order_type in ("limit", "stop"):
        trade_status = "pending_entry"
    else:
        trade_status = "open"

    # For limit/stop orders the entry_price stored is the limit price --
    # fill_price will be set when the order actually fills
    effective_entry = limit_price if (order_type in ("limit", "stop") and limit_price) else entry

    trade = VirtualTrade(
        signal_id        = signal["id"],
        opened_at        = signal.get("analysis_date"),
        direction        = direction,
        entry_price      = effective_entry,
        stop_loss        = sl,
        take_profit      = tp,
        sl_pips          = round(sl_pips, 1),
        tp_pips          = round(tp_pips, 1),
        opening_balance  = opening_balance,
        risk_pct         = risk_pct,
        risk_gbp         = risk_gbp,
        value_per_pip    = value_per_pip,
        spread_pips      = DEFAULT_SPREAD_PIPS,
        spread_cost_gbp  = spread_cost_gbp,
        status           = trade_status,
        confluence_grade = confluence_grade,
        ai_confidence    = signal.get("ai_confidence") or signal.get("confidence"),
        providers_agree  = signal.get("providers_agree"),
        order_type       = order_type,
        limit_price      = limit_price if order_type in ("limit", "stop") else None,
        expires_bars     = expires_bars if order_type in ("limit", "stop") else None,
    )

    session.add(trade)
    session.commit()

    if order_type in ("limit", "stop"):
        log.info(
            "Virtual trade pending entry: signal_id=%s %s %s @ %.5f risk=%.2f%% expires=%d bars",
            signal["id"], direction, order_type.upper(), effective_entry, risk_pct, expires_bars,
        )
    else:
        log.info(
            "Virtual trade opened: signal_id=%s %s risk=%.2f%% (GBP %.2f) SL=%.0fpip",
            signal["id"], direction, risk_pct, risk_gbp, sl_pips,
        )
    return trade


def fill_pending_entries(session, today=None):
    """
    Check all pending_entry / shadow_pending trades to see if their
    limit or stop order has been filled today.

    For each pending trade, fetches hourly price data for today and checks:
      - LIMIT BUY:  filled if price traded DOWN to limit_price (low <= limit_price)
      - LIMIT SELL: filled if price traded UP to limit_price (high >= limit_price)
      - STOP BUY:   filled if price traded UP to limit_price (high >= limit_price)
      - STOP SELL:  filled if price traded DOWN to limit_price (low <= limit_price)

    If filled: status -> 'open' (or 'shadow'), filled_at and fill_price set.
    If expires_bars exceeded: status -> 'cancelled', outcome_type -> 'cancelled'.

    Returns (filled_count, cancelled_count).
    """
    from datetime import date as _date, timedelta
    from tracker.database import VirtualTrade, Signal

    if today is None:
        today = _date.today()

    pending = session.query(VirtualTrade).filter(
        VirtualTrade.status.in_(["pending_entry", "shadow_pending"])
    ).all()

    if not pending:
        return 0, 0

    filled = cancelled = 0

    for trade in pending:
        sig = session.query(Signal).filter_by(id=trade.signal_id).first()
        pair = (sig.pair if sig else None) or "GBPUSD=X"
        order_date = trade.opened_at
        limit_px   = float(trade.limit_price or trade.entry_price or 0)
        direction  = (trade.direction or "BUY").upper()
        order_type = (trade.order_type or "limit").lower()
        expires    = int(trade.expires_bars or 2)

        # Count trading days since signal
        trading_days = 0
        d = order_date
        while d < today:
            d = d + timedelta(days=1)
            if d.weekday() < 5:
                trading_days += 1

        # Expired without fill
        if trading_days > expires:
            is_shadow = trade.status == "shadow_pending"
            trade.status       = "shadow_cancelled" if is_shadow else "cancelled"
            trade.outcome_type = "cancelled"
            trade.closed_at    = today
            session.commit()
            cancelled += 1
            log.info(
                "Limit order cancelled (unfilled): signal_id=%s %s %s @ %.5f (%d days elapsed)",
                trade.signal_id, direction, order_type, limit_px, trading_days,
            )
            continue

        # Fetch today's hourly data to check if price touched the limit
        try:
            import yfinance as yf
            import pandas as pd
            df = yf.download(pair, start=str(today), end=str(today + timedelta(days=1)),
                             interval="1h", progress=False, auto_adjust=True)
            if df is None or df.empty:
                log.warning("fill_pending_entries: no hourly data for %s %s", pair, today)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        except Exception as e:
            log.warning("fill_pending_entries: price fetch failed: %s", e)
            continue

        day_high = float(df["High"].max())
        day_low  = float(df["Low"].min())

        # Check fill condition
        filled_now = False
        if order_type == "limit":
            if direction == "BUY"  and day_low  <= limit_px:
                filled_now = True
            elif direction == "SELL" and day_high >= limit_px:
                filled_now = True
        elif order_type == "stop":
            if direction == "BUY"  and day_high >= limit_px:
                filled_now = True
            elif direction == "SELL" and day_low  <= limit_px:
                filled_now = True

        if filled_now:
            is_shadow = trade.status == "shadow_pending"
            trade.status    = "shadow" if is_shadow else "open"
            trade.filled_at = today
            trade.fill_price = limit_px
            # Recalculate sl_pips / value_per_pip based on actual fill price
            sl = float(trade.stop_loss or 0)
            if sl > 0 and limit_px > 0:
                sl_pips = abs(limit_px - sl) * 10000
                if sl_pips > 0:
                    risk_gbp = float(trade.risk_gbp or 0)
                    trade.sl_pips       = round(sl_pips, 1)
                    trade.value_per_pip = round(risk_gbp / sl_pips, 6)
                    trade.spread_cost_gbp = round(DEFAULT_SPREAD_PIPS * float(trade.value_per_pip), 4)
            session.commit()
            filled += 1
            log.info(
                "Limit order filled: signal_id=%s %s %s @ %.5f on %s",
                trade.signal_id, direction, order_type, limit_px, today,
            )

    return filled, cancelled


def close_trade(session, signal_id, outcome):
    """
    Close a virtual trade using the resolved outcome.

    outcome: dict with keys: would_have_hit_tp, would_have_hit_sl,
             pips_moved, actual_close_price, outcome_date

    Updates trade status and account balance.
    Returns the VirtualTrade row or None.
    """
    from tracker.database import VirtualTrade, update_virtual_balance

    trade = session.query(VirtualTrade).filter_by(signal_id=signal_id).first()
    if not trade:
        log.warning(
            "close_trade: no virtual trade found for signal_id=%s", signal_id
        )
        return None

    if trade.status in ("pending_entry", "shadow_pending"):
        # Order never filled -- don't close, let fill_pending_entries handle it
        log.info(
            "close_trade: signal_id=%s is still pending entry -- skipping close",
            signal_id,
        )
        return trade

    if trade.status not in ("open", "shadow"):
        log.info(
            "close_trade: trade %s already closed (status=%s)", trade.id, trade.status
        )
        return trade

    hit_tp = outcome.get("would_have_hit_tp", False)
    hit_sl = outcome.get("would_have_hit_sl", False)
    pips   = float(outcome.get("pips_moved", 0))

    value_per_pip   = float(trade.value_per_pip   or 0)
    spread_cost_gbp = float(trade.spread_cost_gbp or 0)
    risk_gbp        = float(trade.risk_gbp        or 0)
    opening_balance = float(trade.opening_balance or STARTING_BALANCE)

    if hit_tp:
        outcome_type = "tp_hit"
        pips_result  = float(trade.tp_pips or 0)
        gross_pnl    = round(pips_result * value_per_pip, 4)
        status       = "won"
    elif hit_sl:
        outcome_type = "sl_hit"
        pips_result  = -float(trade.sl_pips or 0)
        gross_pnl    = -risk_gbp
        status       = "lost"
    else:
        outcome_type = "expired"
        pips_result  = round(pips, 1)
        gross_pnl    = round(pips * value_per_pip, 4)
        status       = "expired"  # expired is not a win or loss

    net_pnl         = round(gross_pnl - spread_cost_gbp, 4)
    closing_balance = round(opening_balance + net_pnl, 2)

    is_shadow = (trade.status == "shadow")

    trade.closed_at       = outcome.get("outcome_date")
    trade.outcome_type    = outcome_type
    trade.pips_result     = pips_result
    trade.gross_pnl_gbp   = gross_pnl
    trade.net_pnl_gbp     = net_pnl
    # Shadow trades: show what balance WOULD have been, but don't use it
    trade.closing_balance = closing_balance
    # Mark final status as shadow_won / shadow_lost so they're clearly separate
    if is_shadow:
        trade.status = f"shadow_{status}" if status in ("won", "lost") else "shadow_expired"
    else:
        trade.status = status

    session.commit()

    if is_shadow:
        # Shadow trade: record P&L for reference but do NOT update real balance
        log.info(
            "Shadow trade closed: signal_id=%s %s pips=%+.0f shadow_pnl=GBP %+.4f (balance unchanged)",
            signal_id, outcome_type, pips_result, net_pnl,
        )
    else:
        # Real trade: update account balance
        note = (
            f"{outcome_type} on signal_id={signal_id} "
            f"({'+' if net_pnl >= 0 else ''}{net_pnl:.2f} GBP)"
        )
        update_virtual_balance(session, closing_balance, note)
        log.info(
            "Virtual trade closed: signal_id=%s %s pips=%+.0f net_pnl=GBP %+.4f balance=GBP %.2f",
            signal_id, outcome_type, pips_result, net_pnl, closing_balance,
        )

    return trade
