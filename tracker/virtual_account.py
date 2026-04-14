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


def open_trade(session, signal, confluence_grade=None, risk_pct=None):
    """
    Open a virtual trade for the given signal.

    signal: dict with keys: id, signal, entry_price, stop_loss, take_profit,
            analysis_date, ai_confidence, providers_agree
    risk_pct: override risk % (if None, derive from confluence_grade)

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
    if confluence_grade == "C":
        trade_status = "shadow"
    elif direction == "HOLD" or risk_pct == 0 or confluence_grade == "D":
        trade_status = "skipped"
    else:
        trade_status = "open"

    trade = VirtualTrade(
        signal_id        = signal["id"],
        opened_at        = signal.get("analysis_date"),
        direction        = direction,
        entry_price      = entry,
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
    )

    session.add(trade)
    session.commit()
    log.info(
        "Virtual trade opened: signal_id=%s %s risk=%.2f%% (GBP %.2f) SL=%.0fpip",
        signal["id"], direction, risk_pct, risk_gbp, sl_pips,
    )
    return trade


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
        status       = "won" if pips > 0 else "lost" if pips < 0 else "expired"

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
