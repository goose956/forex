"""
tracker/confluence_engine.py -- Confluence scoring engine for GBP/USD signals.

Calculates a multi-factor technical and macro confluence score to assess signal quality.
All methods are wrapped in try/except so failures never break the main daily run.

Usage:
    from tracker.confluence_engine import ConfluenceEngine
    engine = ConfluenceEngine()
    price_data  = engine.fetch_price_data()
    market_data = engine.fetch_market_data()
    scorecard   = engine.calculate_score('BUY', 8, True, price_data, market_data)
    summary     = engine.generate_summary(scorecard, 'BUY')
"""

import logging
import warnings
from datetime import datetime, date

warnings.filterwarnings("ignore")

log = logging.getLogger("confluence_engine")


class ConfluenceEngine:
    """Multi-factor confluence scoring for GBP/USD forex signals."""

    # ---- Data fetching -------------------------------------------------------

    def fetch_price_data(self, pair: str = "GBPUSD=X") -> dict:
        """
        Fetch last 200 days OHLCV for the pair via yfinance.
        Returns a dict of technical indicators, or an empty dict on failure.
        """
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np

            df = yf.download(pair, period="300d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                log.warning("fetch_price_data: yfinance returned empty data for %s", pair)
                return {}

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            # Use last 200 candles
            df = df.tail(200).copy()
            if len(df) < 50:
                log.warning("fetch_price_data: not enough data (%d candles)", len(df))
                return {}

            closes = df["Close"].values.astype(float)
            highs  = df["High"].values.astype(float)
            lows   = df["Low"].values.astype(float)

            current_price = float(closes[-1])

            # ---- Moving averages ----
            price_50ma  = float(np.mean(closes[-50:])) if len(closes) >= 50 else None
            price_200ma = float(np.mean(closes)) if len(closes) >= 200 else (float(np.mean(closes)) if len(closes) >= 50 else None)

            above_50ma  = bool(current_price > price_50ma)  if price_50ma  is not None else None
            above_200ma = bool(current_price > price_200ma) if price_200ma is not None else None

            # MA alignment
            if price_50ma is not None and price_200ma is not None:
                diff_pips = (price_50ma - price_200ma) * 10000
                if diff_pips > 10:
                    ma_alignment = "bullish"
                elif diff_pips < -10:
                    ma_alignment = "bearish"
                else:
                    ma_alignment = "neutral"
            else:
                ma_alignment = "neutral"

            # Trend direction
            if price_50ma is not None and price_200ma is not None:
                if current_price > price_50ma and price_50ma > price_200ma:
                    trend_direction = "up"
                elif current_price < price_50ma and price_50ma < price_200ma:
                    trend_direction = "down"
                else:
                    trend_direction = "sideways"
            else:
                trend_direction = "sideways"

            # ---- ADX (14 period, manual) ----
            adx_value = None
            trend_strength = "weak"
            try:
                period = 14
                n = len(closes)
                if n >= period + 1:
                    tr_list   = []
                    pdm_list  = []
                    ndm_list  = []

                    for i in range(1, n):
                        h, l, pc = float(highs[i]), float(lows[i]), float(closes[i-1])
                        tr = max(h - l, abs(h - pc), abs(l - pc))
                        pdm = max(h - float(highs[i-1]), 0.0) if (h - float(highs[i-1])) > (float(lows[i-1]) - l) else 0.0
                        ndm = max(float(lows[i-1]) - l, 0.0) if (float(lows[i-1]) - l) > (h - float(highs[i-1])) else 0.0
                        tr_list.append(tr)
                        pdm_list.append(pdm)
                        ndm_list.append(ndm)

                    def wilder_smooth(data, period):
                        result = []
                        s = sum(data[:period])
                        result.append(s)
                        for v in data[period:]:
                            s = s - (s / period) + v
                            result.append(s)
                        return result

                    atr_s  = wilder_smooth(tr_list,  period)
                    pdm_s  = wilder_smooth(pdm_list, period)
                    ndm_s  = wilder_smooth(ndm_list, period)

                    dx_list = []
                    for i in range(len(atr_s)):
                        if atr_s[i] == 0:
                            continue
                        pdi = 100 * pdm_s[i] / atr_s[i]
                        ndi = 100 * ndm_s[i] / atr_s[i]
                        denom = pdi + ndi
                        dx = 100 * abs(pdi - ndi) / denom if denom != 0 else 0.0
                        dx_list.append(dx)

                    if len(dx_list) >= period:
                        # ADX = Wilder smoothing of DX, seeded with the AVERAGE of first N values
                        adx = sum(dx_list[:period]) / period
                        for v in dx_list[period:]:
                            adx = (adx * (period - 1) + v) / period
                        adx_value = round(float(adx), 2)
                    elif dx_list:
                        adx_value = round(float(sum(dx_list) / len(dx_list)), 2)

                if adx_value is not None:
                    if adx_value > 30:
                        trend_strength = "strong"
                    elif adx_value >= 20:
                        trend_strength = "moderate"
                    else:
                        trend_strength = "weak"
            except Exception as e:
                log.warning("ADX calculation failed: %s", e)

            # ---- RSI (14 period, manual) ----
            rsi_value = None
            rsi_condition = "normal"
            try:
                period = 14
                if len(closes) >= period + 1:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains  = [max(d, 0.0) for d in deltas]
                    losses = [max(-d, 0.0) for d in deltas]

                    avg_gain = sum(gains[:period]) / period
                    avg_loss = sum(losses[:period]) / period

                    for i in range(period, len(gains)):
                        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

                    if avg_loss == 0:
                        rsi_value = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi_value = round(100.0 - (100.0 / (1 + rs)), 2)

                    if rsi_value < 35:
                        rsi_condition = "oversold"
                    elif rsi_value > 65:
                        rsi_condition = "overbought"
                    else:
                        rsi_condition = "normal"
            except Exception as e:
                log.warning("RSI calculation failed: %s", e)

            # ---- RSI Divergence (last 10 candles) ----
            rsi_divergence = "none"
            try:
                if rsi_value is not None and len(closes) >= 25:
                    window = 10

                    # Build per-candle RSI for the divergence window
                    rsi_series = []
                    period = 14
                    base = len(closes) - window - period
                    if base >= 0:
                        seg_closes = closes[base:]
                        deltas = [seg_closes[i] - seg_closes[i-1] for i in range(1, len(seg_closes))]
                        gains  = [max(d, 0.0) for d in deltas]
                        losses = [max(-d, 0.0) for d in deltas]
                        avg_gain = sum(gains[:period]) / period
                        avg_loss = sum(losses[:period]) / period
                        rsi_series.append(100 - 100/(1 + avg_gain/avg_loss) if avg_loss != 0 else 100)
                        for i in range(period, len(gains)):
                            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                            rsi_series.append(100 - 100/(1 + avg_gain/avg_loss) if avg_loss != 0 else 100)

                    if len(rsi_series) >= 2:
                        price_window = closes[-window:]
                        rsi_window   = rsi_series[-window:]

                        price_low_start  = price_window[0]
                        price_low_end    = price_window[-1]
                        rsi_low_start    = rsi_window[0]
                        rsi_low_end      = rsi_window[-1]
                        price_high_start = price_window[0]
                        price_high_end   = price_window[-1]
                        rsi_high_start   = rsi_window[0]
                        rsi_high_end     = rsi_window[-1]

                        bullish = (price_low_end < price_low_start) and (rsi_low_end > rsi_low_start)
                        bearish = (price_high_end > price_high_start) and (rsi_high_end < rsi_high_start)

                        if bullish:
                            rsi_divergence = "bullish"
                        elif bearish:
                            rsi_divergence = "bearish"
            except Exception as e:
                log.warning("RSI divergence calculation failed: %s", e)

            # ---- Key Levels (swing highs/lows in last 50 candles) ----
            nearest_support    = None
            nearest_resistance = None
            at_key_level       = False
            key_level_type     = "open_space"
            key_level_price    = None
            distance_to_support_pips    = None
            distance_to_resistance_pips = None

            try:
                window_50 = min(50, len(closes))
                w_highs = highs[-window_50:]
                w_lows  = lows[-window_50:]

                swing_lows  = []
                swing_highs = []

                for i in range(2, len(w_lows) - 2):
                    if w_lows[i] < w_lows[i-1] and w_lows[i] < w_lows[i-2] and \
                       w_lows[i] < w_lows[i+1] and w_lows[i] < w_lows[i+2]:
                        swing_lows.append(float(w_lows[i]))

                for i in range(2, len(w_highs) - 2):
                    if w_highs[i] > w_highs[i-1] and w_highs[i] > w_highs[i-2] and \
                       w_highs[i] > w_highs[i+1] and w_highs[i] > w_highs[i+2]:
                        swing_highs.append(float(w_highs[i]))

                if swing_lows:
                    nearest_support = round(min(swing_lows, key=lambda x: abs(x - current_price)), 5)
                if swing_highs:
                    nearest_resistance = round(max(swing_highs, key=lambda x: abs(x - current_price)), 5)

                # Fallback if no swing detected
                if nearest_support is None:
                    nearest_support = round(float(min(w_lows)), 5)
                if nearest_resistance is None:
                    nearest_resistance = round(float(max(w_highs)), 5)

                distance_to_support_pips    = round(abs(current_price - nearest_support) * 10000, 1)
                distance_to_resistance_pips = round(abs(current_price - nearest_resistance) * 10000, 1)

                at_key_level = (
                    distance_to_support_pips    <= 20 or
                    distance_to_resistance_pips <= 20
                )

                if distance_to_support_pips <= 20:
                    key_level_type  = "at_support"
                    key_level_price = nearest_support
                elif distance_to_resistance_pips <= 20:
                    key_level_type  = "at_resistance"
                    key_level_price = nearest_resistance
                else:
                    key_level_type  = "open_space"
                    key_level_price = None

            except Exception as e:
                log.warning("Key level calculation failed: %s", e)

            return {
                "current_price":              round(current_price, 5),
                "price_50ma":                 round(price_50ma,  5) if price_50ma  is not None else None,
                "price_200ma":                round(price_200ma, 5) if price_200ma is not None else None,
                "above_50ma":                 above_50ma,
                "above_200ma":                above_200ma,
                "ma_alignment":               ma_alignment,
                "trend_direction":            trend_direction,
                "adx_value":                  adx_value,
                "trend_strength":             trend_strength,
                "rsi_value":                  rsi_value,
                "rsi_condition":              rsi_condition,
                "rsi_divergence":             rsi_divergence,
                "nearest_support":            nearest_support,
                "nearest_resistance":         nearest_resistance,
                "distance_to_support_pips":   distance_to_support_pips,
                "distance_to_resistance_pips": distance_to_resistance_pips,
                "at_key_level":               at_key_level,
                "key_level_type":             key_level_type,
                "key_level_price":            key_level_price,
            }

        except Exception as e:
            log.error("fetch_price_data failed: %s", e)
            return {}

    def fetch_market_data(self) -> dict:
        """
        Fetch DXY and UK/US yield spread data.
        Returns a dict — individual fields may be None if data is unavailable.
        """
        result = {
            "dxy_current":          None,
            "dxy_1day_change_pct":  None,
            "dxy_5day_change_pct":  None,
            "dxy_trend":            None,
            "dxy_vs_200ma":         None,
            "us_10yr":              None,
            "uk_10yr":              None,
            "yield_spread":         None,
            "spread_5day_change":   None,
            "spread_direction":     None,
        }

        # ---- DXY ----
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np

            dxy = yf.download("DX-Y.NYB", period="60d", progress=False, auto_adjust=True)
            if dxy is not None and not dxy.empty:
                if isinstance(dxy.columns, pd.MultiIndex):
                    dxy.columns = dxy.columns.droplevel(1)
                dxy = dxy.tail(10)
                closes = dxy["Close"].values.astype(float)
                if len(closes) >= 2:
                    result["dxy_current"]         = round(float(closes[-1]), 3)
                    result["dxy_1day_change_pct"]  = round((closes[-1] - closes[-2]) / closes[-2] * 100, 3)
                if len(closes) >= 6:
                    result["dxy_5day_change_pct"]  = round((closes[-1] - closes[-6]) / closes[-6] * 100, 3)

                pct5 = result["dxy_5day_change_pct"]
                if pct5 is not None:
                    if pct5 > 0.3:
                        result["dxy_trend"] = "rising"
                    elif pct5 < -0.3:
                        result["dxy_trend"] = "falling"
                    else:
                        result["dxy_trend"] = "flat"

                # DXY vs 200MA
                try:
                    dxy_long = yf.download("DX-Y.NYB", period="400d", progress=False, auto_adjust=True)
                    if isinstance(dxy_long.columns, pd.MultiIndex):
                        dxy_long.columns = dxy_long.columns.droplevel(1)
                    if len(dxy_long) >= 200:
                        ma200 = float(np.mean(dxy_long["Close"].values[-200:]))
                        result["dxy_vs_200ma"] = "above" if result["dxy_current"] > ma200 else "below"
                except Exception as e:
                    log.warning("DXY 200MA failed: %s", e)
        except Exception as e:
            log.warning("DXY fetch failed: %s", e)

        # ---- Yield Spread (FRED) ----
        try:
            import requests

            def _parse_fred_csv(url: str, n: int = 10):
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    lines = resp.text.strip().splitlines()
                    # Skip header line
                    data_lines = [l for l in lines[1:] if l and "." in l.split(",")[-1]]
                    values = []
                    for line in data_lines[-n:]:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                val = float(parts[1].strip())
                                values.append(val)
                            except ValueError:
                                pass
                    return values
                except Exception as fe:
                    log.warning("FRED CSV parse failed (%s): %s", url, fe)
                    return []

            us_vals = _parse_fred_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10")
            uk_vals = _parse_fred_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01GBM156N")

            if us_vals and uk_vals:
                result["us_10yr"]  = round(us_vals[-1], 3)
                result["uk_10yr"]  = round(uk_vals[-1], 3)
                result["yield_spread"] = round(uk_vals[-1] - us_vals[-1], 3)

                # 5-day spread change
                min_len = min(len(us_vals), len(uk_vals))
                if min_len >= 6:
                    spread_now  = uk_vals[-1] - us_vals[-1]
                    spread_prev = uk_vals[-6] - us_vals[-6]
                    result["spread_5day_change"] = round(spread_now - spread_prev, 3)

                    change = result["spread_5day_change"]
                    if change is not None:
                        if change > 0.05:
                            result["spread_direction"] = "gbp_positive"
                        elif change < -0.05:
                            result["spread_direction"] = "gbp_negative"
                        else:
                            result["spread_direction"] = "stable"
            else:
                log.warning("FRED yield data unavailable.")
        except Exception as e:
            log.warning("Yield spread fetch failed: %s", e)

        return result

    def fetch_risk_environment(self) -> dict:
        """
        Fetch VIX (fear gauge) and EURUSD trend as additional risk filters.

        Returns dict:
        {
            'vix_current':    float or None   -- VIX index level
            'vix_level':      'low'/'elevated'/'high'/'extreme'
            'vix_signal':     'clear'/'caution'/'avoid'
            'eurusd_trend':   'up'/'down'/'sideways'
            'eurusd_aligned': bool  -- True if EURUSD trend matches GBP signal direction
            'eurusd_rsi':     float or None
            'risk_notes':     str
        }
        """
        result = {
            "vix_current":    None,
            "vix_level":      None,
            "vix_signal":     None,
            "eurusd_trend":   None,
            "eurusd_aligned": None,
            "eurusd_rsi":     None,
            "risk_notes":     "",
        }

        # ---- VIX ----
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np

            vix_df = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
            if vix_df is not None and not vix_df.empty:
                if isinstance(vix_df.columns, pd.MultiIndex):
                    vix_df.columns = vix_df.columns.droplevel(1)
                vix_current = float(vix_df["Close"].iloc[-1])
                result["vix_current"] = round(vix_current, 2)

                if vix_current < 15:
                    result["vix_level"]  = "low"
                    result["vix_signal"] = "clear"
                elif vix_current < 20:
                    result["vix_level"]  = "normal"
                    result["vix_signal"] = "clear"
                elif vix_current < 25:
                    result["vix_level"]  = "elevated"
                    result["vix_signal"] = "caution"
                elif vix_current < 35:
                    result["vix_level"]  = "high"
                    result["vix_signal"] = "avoid"
                else:
                    result["vix_level"]  = "extreme"
                    result["vix_signal"] = "avoid"

                log.info("VIX: %.1f (%s)", vix_current, result["vix_level"])
        except Exception as e:
            log.warning("VIX fetch failed: %s", e)

        # ---- EURUSD trend ----
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np

            eur_df = yf.download("EURUSD=X", period="60d", progress=False, auto_adjust=True)
            if eur_df is not None and not eur_df.empty:
                if isinstance(eur_df.columns, pd.MultiIndex):
                    eur_df.columns = eur_df.columns.droplevel(1)
                eur_df = eur_df.tail(60)
                closes = eur_df["Close"].values.astype(float)

                if len(closes) >= 50:
                    ma50  = float(np.mean(closes[-50:]))
                    ma20  = float(np.mean(closes[-20:]))
                    price = float(closes[-1])

                    if price > ma50 and ma20 > ma50:
                        result["eurusd_trend"] = "up"
                    elif price < ma50 and ma20 < ma50:
                        result["eurusd_trend"] = "down"
                    else:
                        result["eurusd_trend"] = "sideways"

                    # RSI for EURUSD
                    if len(closes) >= 15:
                        period = 14
                        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                        gains  = [max(d, 0.0) for d in deltas]
                        losses = [max(-d, 0.0) for d in deltas]
                        avg_gain = sum(gains[:period]) / period
                        avg_loss = sum(losses[:period]) / period
                        for i in range(period, len(gains)):
                            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                        if avg_loss == 0:
                            result["eurusd_rsi"] = 100.0
                        else:
                            rs = avg_gain / avg_loss
                            result["eurusd_rsi"] = round(100.0 - (100.0 / (1 + rs)), 2)

                log.info("EURUSD trend: %s RSI: %s", result["eurusd_trend"], result["eurusd_rsi"])
        except Exception as e:
            log.warning("EURUSD fetch failed: %s", e)

        # ---- Build risk notes ----
        notes = []
        if result["vix_signal"] == "avoid":
            notes.append(f"VIX {result['vix_current']:.1f} ({result['vix_level']}) -- high fear, reduce exposure")
        elif result["vix_signal"] == "caution":
            notes.append(f"VIX {result['vix_current']:.1f} (elevated) -- trade with caution")
        if result["eurusd_trend"]:
            notes.append(f"EURUSD trend: {result['eurusd_trend']}")
        result["risk_notes"] = " | ".join(notes) if notes else "Risk environment normal"

        return result

    # ---- Scoring ---------------------------------------------------------------

    def calculate_score(
        self,
        signal: str,
        ai_confidence: int,
        providers_agree: bool,
        price_data: dict,
        market_data: dict,
        agreement_pct=None,
        vote_count=None,
        news_risk=None,
        risk_env=None,
    ) -> dict:
        """
        Calculate a multi-factor confluence score for the given signal.
        Returns a full scorecard dict.
        """
        if not price_data:
            price_data = {}
        if not market_data:
            market_data = {}

        is_buy  = (signal == "BUY")
        is_sell = (signal == "SELL")

        factors = {}

        # ---- FACTOR 1: Trend Alignment (max 3) ----
        above_200ma  = price_data.get("above_200ma")
        ma_alignment = price_data.get("ma_alignment", "neutral")
        current_price = price_data.get("current_price")
        price_200ma   = price_data.get("price_200ma")

        f1_score = None
        f1_max   = 3
        f1_label = "Trend Alignment"
        f1_value = f"Above 200MA: {above_200ma}, MA align: {ma_alignment}"

        if above_200ma is not None and ma_alignment is not None:
            if is_buy:
                if above_200ma and ma_alignment == "bullish":
                    f1_score = 3
                elif above_200ma:
                    f1_score = 2
                elif (current_price is not None and price_200ma is not None and
                      abs(current_price - price_200ma) * 10000 <= 20):
                    f1_score = 1
                else:
                    f1_score = 0
            elif is_sell:
                if not above_200ma and ma_alignment == "bearish":
                    f1_score = 3
                elif not above_200ma:
                    f1_score = 2
                elif (current_price is not None and price_200ma is not None and
                      abs(current_price - price_200ma) * 10000 <= 20):
                    f1_score = 1
                else:
                    f1_score = 0
            else:  # HOLD
                f1_score = 1  # neutral
        factors["trend_alignment"] = {
            "score": f1_score, "max": f1_max,
            "value": f1_value, "label": f1_label
        }

        # ---- FACTOR 2: ADX Strength (max 2, direction-neutral) ----
        adx_value     = price_data.get("adx_value")
        trend_strength = price_data.get("trend_strength", "weak")

        f2_score = None
        if adx_value is not None:
            if adx_value > 30:
                f2_score = 2
            elif adx_value >= 20:
                f2_score = 1
            else:
                f2_score = 0
        factors["adx_strength"] = {
            "score": f2_score, "max": 2,
            "value": f"ADX: {adx_value} ({trend_strength})", "label": "ADX Strength"
        }

        # ---- FACTOR 3: RSI Condition (max 2) ----
        rsi_value     = price_data.get("rsi_value")
        rsi_condition = price_data.get("rsi_condition", "normal")

        f3_score = None
        if rsi_value is not None:
            if is_buy:
                if rsi_condition == "oversold":
                    f3_score = 2
                elif rsi_condition == "normal":
                    f3_score = 1
                else:
                    f3_score = 0
            elif is_sell:
                if rsi_condition == "overbought":
                    f3_score = 2
                elif rsi_condition == "normal":
                    f3_score = 1
                else:
                    f3_score = 0
            else:
                f3_score = 1
        factors["rsi_condition"] = {
            "score": f3_score, "max": 2,
            "value": f"RSI: {rsi_value} ({rsi_condition})", "label": "RSI Condition"
        }

        # ---- FACTOR 4: Key Level (max 2) ----
        key_level_type = price_data.get("key_level_type", "open_space")
        dist_support   = price_data.get("distance_to_support_pips")
        dist_resist    = price_data.get("distance_to_resistance_pips")

        f4_score = None
        if key_level_type is not None:
            if is_buy:
                if key_level_type == "at_support":
                    f4_score = 2
                elif key_level_type == "at_resistance":
                    f4_score = 0
                else:
                    # near support within 20 pip
                    if dist_support is not None and dist_support <= 20:
                        f4_score = 1
                    else:
                        f4_score = 1  # open space
            elif is_sell:
                if key_level_type == "at_resistance":
                    f4_score = 2
                elif key_level_type == "at_support":
                    f4_score = 0
                else:
                    if dist_resist is not None and dist_resist <= 20:
                        f4_score = 1
                    else:
                        f4_score = 1
            else:
                f4_score = 1
        factors["key_level"] = {
            "score": f4_score, "max": 2,
            "value": f"Level: {key_level_type}, dist_sup: {dist_support}pip, dist_res: {dist_resist}pip",
            "label": "Key Level"
        }

        # ---- FACTOR 5: DXY Direction (max 2) ----
        dxy_trend   = market_data.get("dxy_trend")
        dxy_vs_200ma = market_data.get("dxy_vs_200ma")

        f5_score = None
        if dxy_trend is not None:
            if is_buy:
                # Want USD weak = DXY falling
                if dxy_trend == "falling" and dxy_vs_200ma == "below":
                    f5_score = 2
                elif dxy_trend == "falling":
                    f5_score = 1
                elif dxy_trend == "flat":
                    f5_score = 1
                else:
                    f5_score = 0
            elif is_sell:
                # Want USD strong = DXY rising
                if dxy_trend == "rising" and dxy_vs_200ma == "above":
                    f5_score = 2
                elif dxy_trend == "rising":
                    f5_score = 1
                elif dxy_trend == "flat":
                    f5_score = 1
                else:
                    f5_score = 0
            else:
                f5_score = 1
        factors["dxy_direction"] = {
            "score": f5_score, "max": 2,
            "value": f"DXY trend: {dxy_trend}, vs 200MA: {dxy_vs_200ma}",
            "label": "DXY Direction"
        }

        # ---- FACTOR 6: Yield Spread (max 3) ----
        spread_direction = market_data.get("spread_direction")

        f6_score = None
        if spread_direction is not None:
            if is_buy:
                if spread_direction == "gbp_positive":
                    f6_score = 3
                elif spread_direction == "stable":
                    f6_score = 1
                else:
                    f6_score = 0
            elif is_sell:
                if spread_direction == "gbp_negative":
                    f6_score = 3
                elif spread_direction == "stable":
                    f6_score = 1
                else:
                    f6_score = 0
            else:
                f6_score = 1
        factors["yield_spread"] = {
            "score": f6_score, "max": 3,
            "value": f"Spread dir: {spread_direction}, UK-US: {market_data.get('yield_spread')}",
            "label": "UK/US Yield Spread"
        }

        # ---- FACTOR 7: RSI Divergence (max 2) ----
        rsi_divergence = price_data.get("rsi_divergence", "none")

        f7_score = None
        if rsi_divergence is not None:
            if is_buy:
                if rsi_divergence == "bullish":
                    f7_score = 2
                elif rsi_divergence == "none":
                    f7_score = 1
                else:
                    f7_score = 0
            elif is_sell:
                if rsi_divergence == "bearish":
                    f7_score = 2
                elif rsi_divergence == "none":
                    f7_score = 1
                else:
                    f7_score = 0
            else:
                f7_score = 1
        factors["rsi_divergence"] = {
            "score": f7_score, "max": 2,
            "value": f"RSI divergence: {rsi_divergence}",
            "label": "RSI Divergence"
        }

        # ---- FACTOR 8: News Risk (max 3 points -- can only reduce score) ----
        if news_risk is not None:
            rs = news_risk.get("risk_score", 3)
            rl = news_risk.get("risk_level", "clear")
            warn = news_risk.get("warning_message", "")
            if rl == "binary":
                f_news = 0; lbl = "BINARY EVENT TODAY -- signal invalid"
            elif rl == "high":
                f_news = 0; lbl = "High-impact news today -- do not trade"
            elif rl == "medium":
                f_news = 1; lbl = "One high-impact event today -- reduce size"
            elif rl == "low":
                f_news = 2; lbl = "High-impact event tomorrow -- advisory"
            else:
                f_news = 3; lbl = "No high-impact events -- clear to trade"
            factors["news_risk"] = {"score": f_news, "max": 3, "value": rl, "label": lbl}
        else:
            factors["news_risk"] = {"score": None, "max": 3, "value": None, "label": "Calendar data unavailable"}

        # ---- FACTOR 9: VIX Risk Environment (max 2) ----
        if risk_env is not None:
            vix_signal = risk_env.get("vix_signal")
            vix_current = risk_env.get("vix_current")
            if vix_signal == "clear":
                f_vix = 2
                vix_lbl = f"VIX {vix_current} -- low fear, normal conditions"
            elif vix_signal == "caution":
                f_vix = 1
                vix_lbl = f"VIX {vix_current} -- elevated fear, trade with caution"
            elif vix_signal == "avoid":
                f_vix = 0
                vix_lbl = f"VIX {vix_current} -- high fear, unfavourable for GBP"
            else:
                f_vix = None
                vix_lbl = "VIX data unavailable"
            factors["vix_environment"] = {
                "score": f_vix, "max": 2,
                "value": f"VIX: {vix_current} ({risk_env.get('vix_level')})",
                "label": vix_lbl,
            }
        else:
            factors["vix_environment"] = {"score": None, "max": 2, "value": None, "label": "VIX data unavailable"}

        # ---- FACTOR 10: EURUSD Alignment (max 1) ----
        if risk_env is not None:
            eurusd_trend = risk_env.get("eurusd_trend")
            if eurusd_trend is not None:
                if is_buy:
                    # BUY GBPUSD -- want EURUSD also trending up (GBP/EUR correlated)
                    if eurusd_trend == "up":
                        f_eur = 1
                        eur_lbl = "EURUSD trending up -- aligned with BUY"
                    elif eurusd_trend == "sideways":
                        f_eur = 1
                        eur_lbl = "EURUSD sideways -- neutral"
                    else:
                        f_eur = 0
                        eur_lbl = "EURUSD trending down -- divergence warning"
                elif is_sell:
                    if eurusd_trend == "down":
                        f_eur = 1
                        eur_lbl = "EURUSD trending down -- aligned with SELL"
                    elif eurusd_trend == "sideways":
                        f_eur = 1
                        eur_lbl = "EURUSD sideways -- neutral"
                    else:
                        f_eur = 0
                        eur_lbl = "EURUSD trending up -- divergence warning"
                else:
                    f_eur = 1
                    eur_lbl = f"EURUSD: {eurusd_trend}"
            else:
                f_eur = None
                eur_lbl = "EURUSD data unavailable"
            factors["eurusd_alignment"] = {
                "score": f_eur, "max": 1,
                "value": f"EURUSD trend: {eurusd_trend}",
                "label": eur_lbl,
            }
        else:
            factors["eurusd_alignment"] = {"score": None, "max": 1, "value": None, "label": "EURUSD data unavailable"}

        # ---- AI Confidence (max 3) ----
        if ai_confidence is not None:
            if ai_confidence >= 9:
                f_ai = 3
            elif ai_confidence >= 7:
                f_ai = 2
            elif ai_confidence >= 5:
                f_ai = 1
            else:
                f_ai = 0
        else:
            f_ai = None
        factors["ai_confidence"] = {
            "score": f_ai, "max": 3,
            "value": f"AI confidence: {ai_confidence}/10",
            "label": "AI Confidence"
        }

        # ---- Provider Agreement (max 2) ----
        conf = ai_confidence or 0
        if agreement_pct is not None and vote_count and vote_count >= 3:
            # Multi-model: score based on agreement percentage
            if agreement_pct >= 80:
                f_agree = 2
                lbl = f'{agreement_pct}% of {vote_count} models agree -- strong consensus'
            elif agreement_pct >= 60:
                f_agree = 1
                lbl = f'{agreement_pct}% of {vote_count} models agree -- majority'
            else:
                f_agree = 0
                lbl = f'Models split ({agreement_pct}% agreement) -- low conviction'
        elif providers_agree is not None:
            # Fallback to binary agree/disagree
            if providers_agree and conf >= 7:
                f_agree = 2
                lbl = 'Both AI providers agree with high confidence'
            elif providers_agree:
                f_agree = 1
                lbl = 'Both AI providers agree'
            else:
                f_agree = 0
                lbl = 'AI providers disagree -- treat with caution'
        else:
            f_agree = None
            lbl = 'Provider agreement unknown'
        factors["provider_agreement"] = {
            "score": f_agree, "max": 2,
            "value": lbl if f_agree is not None else f"Providers agree: {providers_agree}",
            "label": "Provider Agreement"
        }

        # ---- Totals ----
        available   = [(k, v) for k, v in factors.items() if v["score"] is not None]
        unavailable = [(k, v) for k, v in factors.items() if v["score"] is None]

        raw_score    = sum(v["score"] for _, v in available)
        max_possible = sum(v["max"]   for _, v in available)

        confluence_pct = round((raw_score / max_possible) * 100, 2) if max_possible > 0 else 0.0

        if confluence_pct >= 85:
            grade = "A+"
        elif confluence_pct >= 70:
            grade = "A"
        elif confluence_pct >= 55:
            grade = "B"
        elif confluence_pct >= 40:
            grade = "C"
        else:
            grade = "D"

        position_map = {"A+": 1.5, "A": 1.0, "B": 0.5, "C": 0.25, "D": 0.0}
        position_size_pct = position_map[grade]

        # Classify each factor
        supporting  = []
        conflicting = []
        neutral_f   = []
        unavail_f   = []

        for k, v in factors.items():
            s = v["score"]
            m = v["max"]
            label = v["label"]
            if s is None:
                unavail_f.append(label)
            elif m > 0 and s == m:
                supporting.append(label)
            elif s == 0:
                conflicting.append(label)
            else:
                neutral_f.append(label)

        data_completeness = round(len(available) / len(factors) * 100, 1) if factors else 0.0

        summary_text = self.generate_summary(
            {
                "raw_score":          raw_score,
                "max_possible":       max_possible,
                "confluence_pct":     confluence_pct,
                "grade":              grade,
                "position_size_pct":  position_size_pct,
                "factors":            factors,
                "supporting":         supporting,
                "conflicting":        conflicting,
                "neutral":            neutral_f,
                "unavailable":        unavail_f,
                "data_completeness":  data_completeness,
            },
            signal,
        )

        return {
            "raw_score":           raw_score,
            "max_possible":        max_possible,
            "confluence_pct":      confluence_pct,
            "grade":               grade,
            "position_size_pct":   position_size_pct,
            "factors":             factors,
            "supporting":          supporting,
            "conflicting":         conflicting,
            "neutral":             neutral_f,
            "unavailable":         unavail_f,
            "data_completeness":   data_completeness,
            "summary_text":        summary_text,
        }

    # ---- Summary ---------------------------------------------------------------

    def calculate_entry_strategy(self, price_data: dict, signal: str) -> dict:
        """
        Calculate a smart entry price rather than just using current price.

        Returns a dict:
        {
          'order_type':    'market' / 'limit' / 'stop',
          'entry_price':   float,
          'entry_rationale': str,
          'pips_from_current': float,  -- how far entry is from current price
          'expires_bars':  int,        -- cancel unfilled order after N daily bars
        }

        Order types:
          market -- enter now (price already at key level)
          limit  -- wait for a pullback to better price (most common)
          stop   -- enter on a breakout above/below current price

        For BUY signals:
          - At support (within 15 pips): market order now
          - Within 50 pips of support: limit order at support + small buffer
          - Strong ADX trend (>30), no nearby support: stop order above current
          - Otherwise: limit order at nearest support

        For SELL signals: mirror logic with resistance.
        For HOLD: return current price as market order (informational only).
        """
        try:
            if not price_data:
                return {
                    'order_type': 'market',
                    'entry_price': 0.0,
                    'entry_rationale': 'No price data available -- using current price',
                    'pips_from_current': 0.0,
                    'expires_bars': 1,
                }

            current    = float(price_data.get('current_price', 0))
            support    = price_data.get('nearest_support')
            resistance = price_data.get('nearest_resistance')
            adx        = float(price_data.get('adx_value') or 20)
            rsi        = float(price_data.get('rsi_value') or 50)
            at_level   = price_data.get('at_key_level', False)
            level_type = price_data.get('key_level_type', 'open_space')

            sig = (signal or 'HOLD').upper()

            # Buffer to add/subtract from key level (5 pips)
            BUFFER = 0.0005

            # Limit orders only make sense if price is close enough to fill within a session.
            # Beyond 30 pips away, use market entry instead.
            LIMIT_MAX_PIPS = 30

            if sig == 'BUY':
                if at_level and level_type == 'at_support':
                    # Already at support -- enter now
                    entry = current
                    order_type = 'market'
                    rationale = f'Price at support {support:.4f} -- market entry now'
                    expires = 1

                elif support and (current - support) * 10000 <= LIMIT_MAX_PIPS:
                    # Support within 30 pips -- realistic intraday pullback, use limit
                    entry = round(support + BUFFER, 5)
                    order_type = 'limit'
                    pips_away = (current - entry) * 10000
                    rationale = f'Limit order: wait for pullback to support {support:.4f} ({pips_away:.0f} pips below current)'
                    expires = 1

                elif adx > 30 and rsi < 65:
                    # Strong uptrend -- buy the breakout
                    entry = round(current + BUFFER, 5)
                    order_type = 'stop'
                    rationale = f'Strong trend (ADX {adx:.0f}) -- stop entry on breakout above {entry:.4f}'
                    expires = 1

                else:
                    # Support too far away or not found -- enter at market
                    entry = current
                    order_type = 'market'
                    pips_to_support = round((current - support) * 10000, 0) if support else None
                    rationale = (
                        f'Support {support:.4f} is {pips_to_support:.0f} pips away -- market entry at current price'
                        if support else 'No key support identified -- entering at current price'
                    )
                    expires = 1

            elif sig == 'SELL':
                if at_level and level_type == 'at_resistance':
                    # Already at resistance -- enter now
                    entry = current
                    order_type = 'market'
                    rationale = f'Price at resistance {resistance:.4f} -- market entry now'
                    expires = 1

                elif resistance and (resistance - current) * 10000 <= LIMIT_MAX_PIPS:
                    # Resistance within 30 pips -- realistic intraday rally, use limit
                    entry = round(resistance - BUFFER, 5)
                    order_type = 'limit'
                    pips_away = (entry - current) * 10000
                    rationale = f'Limit order: wait for rally to resistance {resistance:.4f} ({pips_away:.0f} pips above current)'
                    expires = 1

                elif adx > 30 and rsi > 35:
                    # Strong downtrend -- sell the breakdown
                    entry = round(current - BUFFER, 5)
                    order_type = 'stop'
                    rationale = f'Strong trend (ADX {adx:.0f}) -- stop entry on breakdown below {entry:.4f}'
                    expires = 1

                else:
                    # Resistance too far away or not found -- enter at market
                    entry = current
                    order_type = 'market'
                    pips_to_resistance = round((resistance - current) * 10000, 0) if resistance else None
                    rationale = (
                        f'Resistance {resistance:.4f} is {pips_to_resistance:.0f} pips away -- market entry at current price'
                        if resistance else 'No key resistance identified -- entering at current price'
                    )
                    expires = 1

            else:
                # HOLD
                entry = current
                order_type = 'market'
                rationale = 'HOLD signal -- entry shown for reference only'
                expires = 0

            pips_from_current = round((entry - current) * 10000, 1)

            return {
                'order_type':         order_type,
                'entry_price':        round(entry, 5),
                'entry_rationale':    rationale,
                'pips_from_current':  pips_from_current,
                'expires_bars':       expires,
            }

        except Exception as e:
            log.warning("calculate_entry_strategy failed: %s", e)
            return {
                'order_type': 'market',
                'entry_price': float(price_data.get('current_price', 0)) if price_data else 0.0,
                'entry_rationale': f'Entry calculation failed: {e}',
                'pips_from_current': 0.0,
                'expires_bars': 1,
            }

    def fetch_multi_timeframe(self, pair: str = "GBPUSD=X") -> dict:
        """
        Fetch weekly and 4H trend data for multi-timeframe analysis.

        Returns a dict with:
            weekly_trend        : "up" / "down" / "sideways"
            weekly_above_200ma  : bool
            weekly_rsi          : float
            weekly_ma_alignment : "bullish" / "bearish" / "neutral"
            h4_trend            : "up" / "down" / "sideways"
            h4_above_50ma       : bool
            h4_rsi              : float
            h4_ma_alignment     : "bullish" / "bearish" / "neutral"
            mtf_bias            : "BUY" / "SELL" / "NEUTRAL"
            mtf_notes           : str  (human-readable summary)
        """
        import warnings
        warnings.filterwarnings("ignore")
        result = {}

        try:
            import yfinance as yf
            import numpy as np

            # ---- Weekly ----
            try:
                wk = yf.download(pair, period="400wk", interval="1wk",
                                 progress=False, auto_adjust=True)
                if wk is not None and not wk.empty:
                    import pandas as pd
                    if isinstance(wk.columns, pd.MultiIndex):
                        wk.columns = wk.columns.droplevel(1)
                    wk = wk.dropna()
                    closes = wk["Close"].values.astype(float)
                    if len(closes) >= 50:
                        price     = float(closes[-1])
                        ma20      = float(np.mean(closes[-20:]))
                        ma50      = float(np.mean(closes[-50:]))
                        ma200     = float(np.mean(closes[-200:])) if len(closes) >= 200 else float(np.mean(closes))

                        if price > ma20 and ma20 > ma50:
                            w_trend = "up"
                        elif price < ma20 and ma20 < ma50:
                            w_trend = "down"
                        else:
                            w_trend = "sideways"

                        w_above_200 = bool(price > ma200)

                        diff = (ma20 - ma50) * 10000
                        if diff > 10:
                            w_ma_align = "bullish"
                        elif diff < -10:
                            w_ma_align = "bearish"
                        else:
                            w_ma_align = "neutral"

                        # Weekly RSI (14)
                        w_rsi = None
                        if len(closes) >= 15:
                            deltas = np.diff(closes[-15:])
                            gains  = np.where(deltas > 0, deltas, 0)
                            losses = np.where(deltas < 0, -deltas, 0)
                            avg_g  = np.mean(gains[-14:])
                            avg_l  = np.mean(losses[-14:])
                            if avg_l > 0:
                                rs    = avg_g / avg_l
                                w_rsi = round(100 - 100 / (1 + rs), 1)

                        result["weekly_trend"]       = w_trend
                        result["weekly_above_200ma"] = w_above_200
                        result["weekly_rsi"]         = w_rsi
                        result["weekly_ma_alignment"] = w_ma_align
                        log.info("Weekly trend: %s above_200=%s rsi=%s", w_trend, w_above_200, w_rsi)
            except Exception as e:
                log.warning("Weekly fetch failed: %s", e)

            # ---- 4H ----
            try:
                h4 = yf.download(pair, period="60d", interval="4h",
                                 progress=False, auto_adjust=True)
                if h4 is not None and not h4.empty:
                    import pandas as pd
                    if isinstance(h4.columns, pd.MultiIndex):
                        h4.columns = h4.columns.droplevel(1)
                    h4 = h4.dropna()
                    closes4 = h4["Close"].values.astype(float)
                    if len(closes4) >= 20:
                        price4   = float(closes4[-1])
                        ma20_4h  = float(np.mean(closes4[-20:]))
                        ma50_4h  = float(np.mean(closes4[-50:])) if len(closes4) >= 50 else float(np.mean(closes4))

                        if price4 > ma20_4h and ma20_4h > ma50_4h:
                            h4_trend = "up"
                        elif price4 < ma20_4h and ma20_4h < ma50_4h:
                            h4_trend = "down"
                        else:
                            h4_trend = "sideways"

                        h4_above_50 = bool(price4 > ma50_4h)

                        diff4 = (ma20_4h - ma50_4h) * 10000
                        if diff4 > 5:
                            h4_ma_align = "bullish"
                        elif diff4 < -5:
                            h4_ma_align = "bearish"
                        else:
                            h4_ma_align = "neutral"

                        # 4H RSI (14)
                        h4_rsi = None
                        if len(closes4) >= 15:
                            deltas4 = np.diff(closes4[-15:])
                            gains4  = np.where(deltas4 > 0, deltas4, 0)
                            losses4 = np.where(deltas4 < 0, -deltas4, 0)
                            avg_g4  = np.mean(gains4[-14:])
                            avg_l4  = np.mean(losses4[-14:])
                            if avg_l4 > 0:
                                rs4    = avg_g4 / avg_l4
                                h4_rsi = round(100 - 100 / (1 + rs4), 1)

                        result["h4_trend"]       = h4_trend
                        result["h4_above_50ma"]  = h4_above_50
                        result["h4_rsi"]         = h4_rsi
                        result["h4_ma_alignment"] = h4_ma_align
                        log.info("4H trend: %s above_50=%s rsi=%s", h4_trend, h4_above_50, h4_rsi)
            except Exception as e:
                log.warning("4H fetch failed: %s", e)

        except Exception as e:
            log.warning("fetch_multi_timeframe failed: %s", e)

        # ---- Derive MTF bias ----
        w_trend  = result.get("weekly_trend",  "sideways")
        h4_trend = result.get("h4_trend",      "sideways")

        bullish_count = sum([
            w_trend  == "up",
            h4_trend == "up",
            result.get("weekly_ma_alignment") == "bullish",
            result.get("h4_ma_alignment")     == "bullish",
        ])
        bearish_count = sum([
            w_trend  == "down",
            h4_trend == "down",
            result.get("weekly_ma_alignment") == "bearish",
            result.get("h4_ma_alignment")     == "bearish",
        ])

        if bullish_count >= 3:
            mtf_bias = "BUY"
        elif bearish_count >= 3:
            mtf_bias = "SELL"
        else:
            mtf_bias = "NEUTRAL"

        notes_parts = []
        if result.get("weekly_trend"):
            notes_parts.append(f"Weekly: {w_trend}")
        if result.get("weekly_above_200ma") is not None:
            notes_parts.append(f"{'above' if result['weekly_above_200ma'] else 'below'} weekly 200MA")
        if result.get("h4_trend"):
            notes_parts.append(f"4H: {h4_trend}")
        if result.get("weekly_rsi"):
            notes_parts.append(f"W-RSI {result['weekly_rsi']}")

        result["mtf_bias"]  = mtf_bias
        result["mtf_notes"] = " | ".join(notes_parts) if notes_parts else "No MTF data"

        log.info("MTF bias: %s (%s)", mtf_bias, result["mtf_notes"])
        return result

    def generate_summary(self, scorecard: dict, signal: str) -> str:
        """
        Generate a plain English summary (under 200 words).
        Uses ASCII characters only -- no fancy Unicode.
        """
        try:
            grade     = scorecard.get("grade", "?")
            pct       = scorecard.get("confluence_pct", 0)
            pos_size  = scorecard.get("position_size_pct", 0)
            raw       = scorecard.get("raw_score", 0)
            max_p     = scorecard.get("max_possible", 0)
            factors   = scorecard.get("factors", {})
            supporting   = scorecard.get("supporting", [])
            conflicting  = scorecard.get("conflicting", [])
            unavailable  = scorecard.get("unavailable", [])
            completeness = scorecard.get("data_completeness", 0)

            lines = [
                f"CONFLUENCE SUMMARY -- {signal} signal",
                f"Score: {raw}/{max_p} ({pct:.1f}%)  Grade: {grade}  Position: {pos_size}% risk",
                f"Data completeness: {completeness:.0f}%",
                "",
            ]

            if supporting:
                lines.append("Supporting factors:")
                for f in supporting:
                    lines.append(f"  [+] {f}")

            if conflicting:
                lines.append("Conflicting factors:")
                for f in conflicting:
                    lines.append(f"  [-] {f}")

            if unavailable:
                lines.append("Unavailable data:")
                for f in unavailable:
                    lines.append(f"  [?] {f}")

            # Key price context
            price_info = []
            rsi = factors.get("rsi_condition", {}).get("value", "")
            if rsi:
                price_info.append(rsi)
            adx = factors.get("adx_strength", {}).get("value", "")
            if adx:
                price_info.append(adx)
            kl = factors.get("key_level", {}).get("value", "")
            if kl:
                price_info.append(kl)

            if price_info:
                lines.append("")
                lines.append("Technical context:")
                for info in price_info:
                    lines.append(f"  {info}")

            lines.append("")
            if pos_size > 0:
                lines.append(f"Recommendation: Enter {signal} with {pos_size}% risk.")
            else:
                lines.append("Recommendation: Grade D -- skip this signal.")

            return "\n".join(lines)[:1500]

        except Exception as e:
            log.warning("generate_summary failed: %s", e)
            return f"Summary unavailable: {e}"
