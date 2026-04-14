"""
tracker/agents/forex_agents.py -- Forex-optimised wrapper around TradingAgents.

This module wraps the existing TradingAgents installation with forex-specific
system prompts and returns structured signal dictionaries.

Usage:
    from tracker.agents.forex_agents import ForexTradingAgents

    fa = ForexTradingAgents(pair="GBPUSD=X", provider="anthropic")
    result = fa.run(date="2024-01-15", context_data=context)
    print(result["signal"], result["confidence"])
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# TradingAgents is installed via pip from GitHub into this venv.
# It is imported normally below -- no sys.path manipulation needed.

log = logging.getLogger("forex_agents")

# ---- Forex-specific system prompts ------------------------------------------

FOREX_PROMPTS = {
    "market": """You are a professional forex technical analyst specialising in currency pairs, primarily GBP/USD.
Analyse the provided price data and indicators:
- Identify trend direction on Daily and 4H timeframes
- Assess price position relative to 50 MA and 200 MA
- Evaluate RSI(14) for overbought/oversold conditions
- Assess MACD momentum and any recent crossovers
- Identify the nearest key support and resistance levels
- Use ATR(14) for volatility context
- Assess higher highs/higher lows price structure
Provide a clear technical bias: BULLISH, BEARISH or NEUTRAL with specific price levels and reasoning.
IMPORTANT: Do not reference earnings, P/E ratios, revenue, or any company/stock metrics. This is a currency pair analysis only.""",

    "fundamentals": """You are a professional forex fundamental analyst specialising in GBP/USD macroeconomic drivers.
Using the provided fundamental context, analyse:
- Bank of England vs Federal Reserve interest rate differential and forward guidance trajectory
- Recent UK economic data and surprises: CPI inflation, employment, GDP growth, PMI surveys
- Recent US economic data and surprises: Non-Farm Payrolls, CPI, FOMC decisions, retail sales, ISM surveys
- Any upcoming high-impact economic events this week flagged in the provided calendar data
- Relative strength of GBP and USD against other major currencies
- Current risk-on vs risk-off market environment
Provide a clear fundamental bias: BULLISH GBP, BEARISH GBP, or NEUTRAL with key drivers.
IMPORTANT: Do not reference stock market valuations, company earnings, or equity metrics.""",

    "news": """You are a professional forex news analyst assessing the impact of recent news on GBP/USD.
Using the provided headlines and news context:
- Identify UK news that impacts GBP positively or negatively
- Identify US news that impacts USD positively or negatively
- Flag any central bank speeches or policy signals
- Identify any global risk events affecting safe haven USD demand
- Note any trade, geopolitical or political developments affecting either currency
For each significant news item state:
  Impact: HIGH / MEDIUM / LOW
  Direction: GBP POSITIVE / GBP NEGATIVE / NEUTRAL
  Timeframe: IMMEDIATE / SHORT-TERM / LONG-TERM
Provide overall news sentiment: BULLISH GBP, BEARISH GBP, or MIXED""",

    "social": """You are a professional forex market sentiment analyst.
Using the provided context, assess current market sentiment for GBP/USD:
- Evaluate recent price momentum and follow-through
- Consider whether retail sentiment is typically contrarian to price direction
- Assess whether current move has conviction (volume, follow-through) or looks exhausted
- Consider broader market risk appetite and its effect on GBP/USD
- Note any significant positioning extremes if known
State overall sentiment: BULLISH / BEARISH / NEUTRAL
State whether this is a CONTRARIAN or CONFIRMING signal relative to the technical picture.""",

    "risk": """You are a conservative professional forex risk manager. Your job is to protect capital above all else.
Assess the proposed GBP/USD trade:
- Verify stop loss is placed at a logical structure level, not an arbitrary distance
- Confirm minimum risk/reward ratio of 1:2 is met
- Check for any major news events in next 24 hours that create unacceptable gap or spike risk
- Assess whether current volatility (ATR) makes the trade viable or oversized
- Consider whether multiple timeframes confirm the signal direction
- Evaluate overall trade quality on scale of 1-10 where 10 is highest quality
Rate overall trade risk: LOW / MEDIUM / HIGH / EXTREME
REJECT and recommend HOLD for any EXTREME risk rating.
State clearly whether you APPROVE or REJECT the trade.""",
}

# ---- Cost constants (USD per 1M tokens) -------------------------------------
PRICING = {
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-sonnet-4-5":         (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80,  4.00),
    "claude-haiku-4-5-20251001": (0.80,  4.00),
    "gpt-4o":                  (5.00, 15.00),
    "gpt-4o-mini":             (0.15,  0.60),
}
USD_TO_GBP = 0.79


def _estimate_cost(model, input_tokens, output_tokens):
    if model in PRICING:
        in_r, out_r = PRICING[model]
    else:
        in_r, out_r = 3.00, 15.00
    usd = (input_tokens * in_r + output_tokens * out_r) / 1_000_000
    return round(usd * USD_TO_GBP, 4)


def _rough_tokens(text):
    return max(100, len(str(text)) // 4)


# ---- ForexTradingAgents class -----------------------------------------------

class ForexTradingAgents:
    """
    Wraps TradingAgentsGraph with forex-specific system prompts.

    Args:
        pair:     yfinance ticker, e.g. "GBPUSD=X"
        provider: "anthropic" or "openai"
        debug:    pass True to stream agent output to terminal
    """

    def __init__(self, pair: str = "GBPUSD=X", provider: str = "anthropic", debug: bool = False):
        self.pair = pair
        self.provider = provider.lower()
        self.debug = debug
        self._graph = None
        self._config = None

    def _build_graph(self):
        """Lazy-initialise the TradingAgents graph."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"]      = 1
        config["max_risk_discuss_rounds"] = 1
        config["data_vendors"] = {
            "core_stock_apis":      "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data":     "yfinance",
            "news_data":            "yfinance",
        }

        if self.provider == "anthropic":
            config["llm_provider"]    = "anthropic"
            config["deep_think_llm"]  = "claude-sonnet-4-6"
            config["quick_think_llm"] = "claude-3-5-haiku-20241022"
            config["backend_url"]     = None
        else:
            config["llm_provider"]    = "openai"
            config["deep_think_llm"]  = "gpt-4o-mini"
            config["quick_think_llm"] = "gpt-4o-mini"
            config["backend_url"]     = "https://api.openai.com/v1"

        self._config = config
        self._graph = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=self.debug,
            config=config,
        )
        log.info(f"TradingAgentsGraph built: provider={self.provider}")

    def run(self, date: str, context_data: dict = None) -> dict:
        """
        Run analysis for self.pair on the given date.

        Args:
            date:         "YYYY-MM-DD"
            context_data: optional dict with 'calendar_text', 'news_text', etc.
                          injected into the initial state message if provided

        Returns:
            Structured signal dict with all agent outputs.
        """
        if self._graph is None:
            self._build_graph()

        log.info(f"Running {self.provider} analysis: {self.pair} {date}")
        start = datetime.now()

        try:
            final_state, processed_signal = self._graph.propagate(self.pair, date)
        except Exception as e:
            log.error(f"Analysis failed: {e}")
            raise

        elapsed = (datetime.now() - start).total_seconds()

        # Extract fields
        decision_text = final_state.get("final_trade_decision", "")
        signal_action = _parse_signal(decision_text)
        confidence    = _parse_confidence(decision_text)

        # Investment debate
        debate = final_state.get("investment_debate_state", {})
        bull_arg  = debate.get("bull_history",  "") if isinstance(debate, dict) else ""
        bear_arg  = debate.get("bear_history",  "") if isinstance(debate, dict) else ""

        # Risk debate
        risk = final_state.get("risk_debate_state", {})
        risk_text = risk.get("history", "") if isinstance(risk, dict) else ""

        # Token estimate
        all_text    = " ".join(str(v) for v in final_state.values() if v)
        out_tokens  = _rough_tokens(all_text)
        in_tokens   = out_tokens * 5
        model       = self._config["deep_think_llm"]
        cost_gbp    = _estimate_cost(model, in_tokens, out_tokens)

        return {
            "pair":                 self.pair,
            "date":                 date,
            "provider":             self.provider,
            "model":                model,
            "signal":               signal_action,
            "confidence":           confidence,
            "processed_signal":     processed_signal,
            "full_reasoning":       decision_text,
            "technical_summary":    final_state.get("market_report",        ""),
            "fundamental_summary":  final_state.get("fundamentals_report",  ""),
            "news_summary":         final_state.get("news_report",          ""),
            "sentiment_summary":    final_state.get("sentiment_report",     ""),
            "bull_argument":        bull_arg,
            "bear_argument":        bear_arg,
            "risk_assessment":      risk_text,
            "investment_plan":      final_state.get("investment_plan",      ""),
            "tokens_input":         in_tokens,
            "tokens_output":        out_tokens,
            "tokens_total":         in_tokens + out_tokens,
            "estimated_cost_gbp":   cost_gbp,
            "elapsed_seconds":      round(elapsed, 1),
        }


# ---- Helpers -----------------------------------------------------------------

def _parse_signal(text: str) -> str:
    t = text.upper()
    if "BUY" in t or "LONG" in t:
        return "BUY"
    if "SELL" in t or "SHORT" in t:
        return "SELL"
    return "HOLD"


def _parse_confidence(text: str) -> int:
    """Return confidence as integer 1-10."""
    high   = ["strongly", "confident", "clear", "definitive", "strong", "high confidence", "clearly"]
    medium = ["likely", "suggests", "indicates", "moderate", "cautious", "probably"]
    low    = ["uncertain", "unclear", "mixed", "weak", "limited", "inconclusive"]
    t = text.lower()
    if any(w in t for w in high):
        return 8
    if any(w in t for w in low):
        return 4
    if any(w in t for w in medium):
        return 6
    return 6


# ---- Quick test --------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ForexTradingAgents with GBPUSD=X / 2024-01-15 / anthropic...")
    fa = ForexTradingAgents(pair="GBPUSD=X", provider="anthropic", debug=False)
    result = fa.run(date="2024-01-15")
    print(f"\nSignal:     {result['signal']}")
    print(f"Confidence: {result['confidence']}/10")
    print(f"Provider:   {result['provider']} / {result['model']}")
    print(f"Cost:       GBP {result['estimated_cost_gbp']:.4f}")
    print(f"Duration:   {result['elapsed_seconds']}s")
    print(f"\nProcessed signal: {result['processed_signal']}")
