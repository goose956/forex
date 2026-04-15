"""
tracker/agents/forex_agents.py -- Direct LLM forex analysis (no TradingAgents dependency).

Calls Anthropic and OpenAI APIs directly with forex-specific prompts.
Returns the same structured signal dict as before so nothing else needs changing.
"""

import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger("forex_agents")

# ---- Models ------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL    = "gpt-4o-mini"

# ---- Cost constants (USD per 1M tokens) -------------------------------------
PRICING = {
    "claude-sonnet-4-6":  (3.00, 15.00),
    "claude-sonnet-4-5":  (3.00, 15.00),
    "gpt-4o":             (5.00, 15.00),
    "gpt-4o-mini":        (0.15,  0.60),
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


def _parse_signal(text: str) -> str:
    text_upper = text.upper()
    # Look for explicit signal keywords
    for keyword in ["STRONG BUY", "STRONG SELL"]:
        if keyword in text_upper:
            return keyword.split()[-1]  # BUY or SELL
    if "BUY" in text_upper and "SELL" not in text_upper:
        return "BUY"
    if "SELL" in text_upper and "BUY" not in text_upper:
        return "SELL"
    # Both mentioned -- pick last clear decision
    buy_pos  = text_upper.rfind("BUY")
    sell_pos = text_upper.rfind("SELL")
    if buy_pos > sell_pos:
        return "BUY"
    if sell_pos > buy_pos:
        return "SELL"
    return "HOLD"


def _parse_confidence(text: str) -> int:
    # Look for explicit confidence score e.g. "confidence: 7/10" or "7 out of 10"
    patterns = [
        r'confidence[:\s]+(\d+)\s*/\s*10',
        r'(\d+)\s*/\s*10\s+confidence',
        r'confidence[:\s]+(\d+)',
        r'(\d+)\s+out\s+of\s+10',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            return max(1, min(10, val))
    # Fallback: infer from language
    text_lower = text.lower()
    if any(w in text_lower for w in ["strong", "high conviction", "clear", "confident"]):
        return 7
    if any(w in text_lower for w in ["moderate", "mixed", "uncertain"]):
        return 5
    return 5


# ---- Master analysis prompt --------------------------------------------------

SYSTEM_PROMPT = """You are a professional forex trader and analyst specialising in GBP/USD (British Pound / US Dollar).
You combine technical analysis, fundamental analysis, news sentiment, and market psychology to generate high-quality trade signals.
Your job is to analyse all available data and produce a structured trading recommendation.
Be direct, specific, and use precise price levels. Do not reference equities, earnings, P/E ratios or stock metrics."""

def _build_user_prompt(pair: str, date: str, context_data: dict) -> str:
    ctx = context_data or {}

    # Build context sections
    technical = ctx.get("technical_context", "")
    news      = ctx.get("news_text", "No news data available.")
    calendar  = ctx.get("calendar_text", "No calendar data available.")
    price_ctx = ctx.get("price_context", "")

    prompt = f"""Analyse GBP/USD for {date} and produce a trading signal.

{'=' * 60}
TECHNICAL CONTEXT
{'=' * 60}
{technical or price_ctx or 'Standard technical analysis required.'}

{'=' * 60}
NEWS & SENTIMENT
{'=' * 60}
{news}

{'=' * 60}
ECONOMIC CALENDAR
{'=' * 60}
{calendar}

{'=' * 60}
REQUIRED OUTPUT FORMAT
{'=' * 60}
Provide your analysis in the following exact sections:

TECHNICAL SUMMARY:
[2-3 sentences on trend, key levels, indicator readings]

FUNDAMENTAL SUMMARY:
[2-3 sentences on rate differentials, macro backdrop]

NEWS SUMMARY:
[Key news items and their GBP/USD impact]

SENTIMENT SUMMARY:
[Market sentiment and positioning assessment]

BULL ARGUMENT:
[Top 3 reasons to buy GBP/USD]

BEAR ARGUMENT:
[Top 3 reasons to sell GBP/USD]

RISK ASSESSMENT:
[Key risks and trade invalidation levels]

INVESTMENT PLAN:
[Specific entry rationale, key levels to watch]

FINAL DECISION:
[Signal: BUY / SELL / HOLD]
[Confidence: X/10]
[Reasoning: 2-3 sentences explaining the final call]
"""
    return prompt


# ---- ForexTradingAgents class -----------------------------------------------

class ForexTradingAgents:
    """
    Direct LLM forex analysis using Anthropic or OpenAI APIs.

    Args:
        pair:     yfinance ticker, e.g. "GBPUSD=X"
        provider: "anthropic" or "openai"
        debug:    unused, kept for API compatibility
    """

    def __init__(self, pair: str = "GBPUSD=X", provider: str = "anthropic", debug: bool = False):
        self.pair     = pair
        self.provider = provider.lower()
        self.debug    = debug

    def run(self, date: str, context_data: dict = None) -> dict:
        """
        Run analysis for self.pair on the given date.

        Returns:
            Structured signal dict compatible with the existing pipeline.
        """
        log.info(f"Running {self.provider} analysis: {self.pair} {date}")
        start = datetime.now()

        user_prompt = _build_user_prompt(self.pair, date, context_data or {})
        response_text = ""

        if self.provider == "anthropic":
            response_text, in_tok, out_tok = self._call_anthropic(user_prompt)
            model = ANTHROPIC_MODEL
        else:
            response_text, in_tok, out_tok = self._call_openai(user_prompt)
            model = OPENAI_MODEL

        elapsed = (datetime.now() - start).total_seconds()
        log.info(f"{self.provider} analysis complete in {elapsed:.1f}s "
                 f"({in_tok} in / {out_tok} out tokens)")

        signal     = _parse_signal(response_text)
        confidence = _parse_confidence(response_text)
        cost_gbp   = _estimate_cost(model, in_tok, out_tok)

        def _extract(label):
            pattern = rf'{label}:\s*\n?(.*?)(?=\n[A-Z ]+:|$)'
            m = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        return {
            "pair":                 self.pair,
            "date":                 date,
            "provider":             self.provider,
            "model":                model,
            "signal":               signal,
            "confidence":           confidence,
            "processed_signal":     signal,
            "full_reasoning":       _extract("FINAL DECISION"),
            "technical_summary":    _extract("TECHNICAL SUMMARY"),
            "fundamental_summary":  _extract("FUNDAMENTAL SUMMARY"),
            "news_summary":         _extract("NEWS SUMMARY"),
            "sentiment_summary":    _extract("SENTIMENT SUMMARY"),
            "bull_argument":        _extract("BULL ARGUMENT"),
            "bear_argument":        _extract("BEAR ARGUMENT"),
            "risk_assessment":      _extract("RISK ASSESSMENT"),
            "investment_plan":      _extract("INVESTMENT PLAN"),
            "tokens_input":         in_tok,
            "tokens_output":        out_tok,
            "tokens_total":         in_tok + out_tok,
            "estimated_cost_gbp":   cost_gbp,
            "elapsed_seconds":      round(elapsed, 1),
        }

    def _call_anthropic(self, user_prompt: str) -> tuple:
        """Call Anthropic API with extended thinking enabled.
        Returns (response_text, in_tokens, out_tokens).
        Thinking tokens are included in out_tokens for cost tracking.
        """
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=16000,           # must exceed budget_tokens
            thinking={
                "type": "enabled",
                "budget_tokens": 8000,  # internal reasoning scratchpad
            },
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Extract only text blocks -- thinking blocks are internal and not part of the signal
        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        in_tok  = message.usage.input_tokens
        # thinking tokens count as output tokens for billing
        out_tok = message.usage.output_tokens
        return text, in_tok, out_tok

    def _call_openai(self, user_prompt: str) -> tuple:
        """Call OpenAI API directly. Returns (response_text, in_tokens, out_tokens)."""
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        text    = response.choices[0].message.content
        in_tok  = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        return text, in_tok, out_tok
