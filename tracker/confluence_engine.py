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
