"""
tracker/baseline_strategy.py -- Simple rule-based strategy for benchmarking.

NO LLMs. NO confluence scoring. NO ensemble voting.
Just a straightforward technical rule that runs in parallel to the AI system.

Purpose: if the AI system can't beat this after 100+ trades, the LLMs
are adding cost without edge. This is the null hypothesis -- if we can't
reject it, we know the fancy stuff isn't working.

Rules:
    BUY  if price > 200MA AND ADX > 25 AND RSI < 70
    SELL if price < 200MA AND ADX > 25 AND RSI > 30
    HOLD otherwise

Levels (identical to AI system for fair comparison):
    SL: 40 pips from entry
    TP: 80 pips from entry  (fixed 1:2 R:R)
    Entry: market at current price
"""

import logging

log = logging.getLogger("baseline_strategy")

# Same parameters as AI system for apples-to-apples comparison
SL_PIPS = 40
TP_PIPS = 80
ADX_THRESHOLD = 25
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30


def generate_baseline_signal(price_data: dict) -> dict:
    """
    Run the baseline rule on price data and return a signal dict.

    price_data should contain at minimum:
        current_price, price_200ma, adx_value, rsi_value

    Returns dict with: signal, rule_name, rule_reason, entry_price,
                       stop_loss, take_profit, current_price,
                       price_200ma, adx_value, rsi_value
    """
    current  = price_data.get("current_price")
    ma_200   = price_data.get("price_200ma")
    adx      = price_data.get("adx_value")
    rsi      = price_data.get("rsi_value")

    # Need at minimum current price and ADX to act
    if current is None or adx is None:
        return {
            "signal":      "HOLD",
            "rule_name":   "200MA+ADX",
            "rule_reason": "Missing required inputs (current price or ADX)",
            "entry_price": current,
            "stop_loss":   None,
            "take_profit": None,
            "current_price": current,
            "price_200ma":   ma_200,
            "adx_value":     adx,
            "rsi_value":     rsi,
        }

    current = float(current)
    adx     = float(adx)
    ma_200  = float(ma_200) if ma_200 is not None else None
    rsi     = float(rsi)    if rsi    is not None else None

    signal = "HOLD"
    reason = ""

    # Not enough trend strength
    if adx < ADX_THRESHOLD:
        reason = f"ADX {adx:.0f} below {ADX_THRESHOLD} -- no clear trend"

    # 200MA required -- if missing, skip the trade
    elif ma_200 is None:
        reason = "200MA not available -- cannot determine trend direction"

    # BUY conditions
    elif current > ma_200:
        if rsi is not None and rsi > RSI_OVERBOUGHT:
            reason = f"Above 200MA, ADX {adx:.0f} strong, but RSI {rsi:.0f} overbought"
        else:
            signal = "BUY"
            reason = f"Price {current:.5f} above 200MA {ma_200:.5f}, ADX {adx:.0f} strong"

    # SELL conditions
    elif current < ma_200:
        if rsi is not None and rsi < RSI_OVERSOLD:
            reason = f"Below 200MA, ADX {adx:.0f} strong, but RSI {rsi:.0f} oversold"
        else:
            signal = "SELL"
            reason = f"Price {current:.5f} below 200MA {ma_200:.5f}, ADX {adx:.0f} strong"

    else:
        reason = "Price exactly at 200MA -- no directional bias"

    # Calculate levels
    entry = current
    if signal == "BUY":
        sl = round(entry - SL_PIPS / 10000, 5)
        tp = round(entry + TP_PIPS / 10000, 5)
    elif signal == "SELL":
        sl = round(entry + SL_PIPS / 10000, 5)
        tp = round(entry - TP_PIPS / 10000, 5)
    else:
        sl = None
        tp = None

    return {
        "signal":      signal,
        "rule_name":   "200MA+ADX",
        "rule_reason": reason,
        "entry_price": round(entry, 5),
        "stop_loss":   sl,
        "take_profit": tp,
        "current_price": round(current, 5),
        "price_200ma":   round(ma_200, 5) if ma_200 is not None else None,
        "adx_value":     round(adx, 2),
        "rsi_value":     round(rsi, 2) if rsi is not None else None,
    }


def save_baseline_signal(session, analysis_date, pair, result):
    """Save a baseline strategy result to the database."""
    from tracker.database import BaselineTrade

    trade = BaselineTrade(
        analysis_date   = analysis_date,
        pair            = pair,
        signal          = result["signal"],
        rule_name       = result["rule_name"],
        rule_reason     = result["rule_reason"],
        entry_price     = result["entry_price"],
        stop_loss       = result["stop_loss"],
        take_profit     = result["take_profit"],
        current_price   = result["current_price"],
        price_200ma     = result["price_200ma"],
        adx_value       = result["adx_value"],
        rsi_value       = result["rsi_value"],
    )
    session.add(trade)
    session.commit()
    log.info(
        "Baseline signal saved: id=%s %s %s entry=%s sl=%s tp=%s",
        trade.id, analysis_date, result["signal"],
        result["entry_price"], result["stop_loss"], result["take_profit"],
    )
    return trade
