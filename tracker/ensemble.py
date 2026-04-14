"""
tracker/ensemble.py -- Multi-model ensemble voting for forex signals.

Calls 7 cheap models via OpenRouter (OpenAI-compatible API).
Each model gets the pre-calculated technical context and votes independently.
Graceful failure -- if OpenRouter unavailable, returns empty list.

Current models (9 total including Claude + GPT):
  openrouter: llama-3.3-70b, deepseek-chat, qwen-2.5-72b, mistral-nemo,
              gemini-2.0-flash, gemini-2.0-flash-lite, mistral-small-3.1
"""

import os
import time
import json
import logging
log = logging.getLogger("ensemble")

USD_TO_GBP = 0.79

# OpenRouter model pricing (USD per 1M tokens, input/output)
OPENROUTER_PRICING = {
    "meta-llama/llama-3.3-70b-instruct":   (0.100, 0.100),  # Meta -- confirmed working
    "deepseek/deepseek-chat":              (0.270, 1.100),  # DeepSeek -- confirmed working
    "qwen/qwen-2.5-72b-instruct":          (0.180, 0.180),  # Alibaba -- confirmed working
    "mistralai/mistral-nemo":              (0.035, 0.080),  # Mistral -- confirmed working
    "google/gemini-2.0-flash-001":         (0.100, 0.400),  # Google Gemini 2.0 Flash
    "google/gemini-2.0-flash-lite-001":    (0.075, 0.300),  # Google Gemini 2.0 Flash Lite
    "mistralai/mistral-small-3.1-24b-instruct": (0.100, 0.300),  # Mistral Small 3.1
}

OPENROUTER_MODELS = list(OPENROUTER_PRICING.keys())

SYSTEM_PROMPT = """You are an expert forex analyst specialising in GBP/USD.
You will be given technical and macro data for GBP/USD and must provide a trading signal.

Respond with ONLY valid JSON in this exact format:
{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": <integer 1-10>,
  "reasoning": "<one concise sentence explaining your decision>"
}

Rules:
- Signal must be exactly BUY, SELL, or HOLD
- Confidence 1-10 where 10 is highest conviction
- Reasoning must be under 100 words
- No other text outside the JSON"""


def build_prompt(price_data, market_data, context_data=None):
    """Build the analysis prompt from pre-calculated data."""
    lines = ["Analyse GBP/USD and provide a trading signal for today.\n"]

    if price_data:
        lines.append("TECHNICAL DATA:")
        lines.append(f"Current price: {price_data.get('current_price', 'N/A'):.4f}")
        lines.append(f"50 MA: {price_data.get('price_50ma', 0):.4f} | 200 MA: {price_data.get('price_200ma', 0):.4f}")
        lines.append(f"Above 200MA: {price_data.get('above_200ma', False)}")
        lines.append(f"MA alignment: {price_data.get('ma_alignment', 'N/A')}")
        lines.append(f"Trend: {price_data.get('trend_direction', 'N/A')}")
        lines.append(f"ADX: {price_data.get('adx_value', 0):.1f} ({price_data.get('trend_strength', 'N/A')})")
        lines.append(f"RSI: {price_data.get('rsi_value', 0):.1f} ({price_data.get('rsi_condition', 'N/A')})")
        lines.append(f"RSI divergence: {price_data.get('rsi_divergence', 'none')}")
        lines.append(f"Nearest support: {price_data.get('nearest_support', 0):.4f} ({price_data.get('distance_to_support_pips', 0):.0f} pips)")
        lines.append(f"Nearest resistance: {price_data.get('nearest_resistance', 0):.4f} ({price_data.get('distance_to_resistance_pips', 0):.0f} pips)")
        lines.append(f"At key level: {price_data.get('at_key_level', False)} ({price_data.get('key_level_type', 'N/A')})")

    if market_data:
        lines.append("\nMACRO DATA:")
        if market_data.get('dxy_trend'):
            lines.append(f"DXY (Dollar Index): {market_data.get('dxy_trend')} ({market_data.get('dxy_1day_change_pct', 0):.2f}% today, {market_data.get('dxy_vs_200ma', 'N/A')} 200MA)")
        if market_data.get('yield_spread') is not None:
            lines.append(f"UK/US 10yr yield spread: {market_data.get('yield_spread', 0):.3f}% ({market_data.get('spread_direction', 'N/A')})")

    if context_data and isinstance(context_data, dict) and context_data.get('news_headlines'):
        lines.append("\nRECENT NEWS:")
        for h in context_data.get('news_headlines', [])[:3]:
            lines.append(f"- {h}")

    lines.append("\nProvide your signal as JSON only.")
    return "\n".join(lines)


def call_model(model_name, prompt, api_key):
    """Call a single OpenRouter model. Returns dict or None."""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        start = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        latency_ms = int((time.time() - start) * 1000)

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Strip markdown code blocks if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data = json.loads(content.strip())

        signal = str(data.get("signal", "HOLD")).upper()
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"
        confidence = max(1, min(10, int(data.get("confidence", 5))))
        reasoning  = str(data.get("reasoning", ""))[:300]

        # Estimate cost
        in_tokens  = response.usage.prompt_tokens     if response.usage else 0
        out_tokens = response.usage.completion_tokens if response.usage else 0
        price_in, price_out = OPENROUTER_PRICING.get(model_name, (0.5, 1.5))
        cost_usd = (in_tokens * price_in + out_tokens * price_out) / 1_000_000
        cost_gbp = round(cost_usd * USD_TO_GBP, 6)

        return {
            "model_name":  model_name,
            "provider":    "openrouter",
            "signal":      signal,
            "confidence":  confidence,
            "reasoning":   reasoning,
            "cost_usd":    round(cost_usd, 6),
            "cost_gbp":    cost_gbp,
            "latency_ms":  latency_ms,
            "input_tokens":  in_tokens,
            "output_tokens": out_tokens,
        }

    except Exception as e:
        log.warning(f"Model {model_name} failed: {e}")
        return None


def run_ensemble(price_data, market_data, context_data=None):
    """
    Run all OpenRouter models and return list of vote dicts.
    Returns empty list if OPENROUTER_API_KEY not set or all models fail.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.info("OPENROUTER_API_KEY not set -- ensemble skipped")
        return []

    prompt = build_prompt(price_data, market_data, context_data)
    votes = []

    for model in OPENROUTER_MODELS:
        log.info(f"Calling ensemble model: {model}")
        result = call_model(model, prompt, api_key)
        if result:
            votes.append(result)
            log.info(f"  {model}: {result['signal']} {result['confidence']}/10")
        else:
            log.warning(f"  {model}: failed -- skipping")

    return votes


def calculate_consensus(claude_result, gpt_result, ensemble_votes):
    """
    Calculate consensus across all available models.

    Returns dict with:
    - final_signal: majority vote signal
    - agreement_pct: % of models agreeing on final signal
    - vote_count: total models that voted
    - all_votes: list of (model, signal, confidence) for display
    - providers_agree: True if all voted the same (backward compatible)
    """
    all_votes = []

    # Add existing providers
    if claude_result and claude_result.get("signal"):
        all_votes.append({
            "model_name": "claude-sonnet-4-6",
            "provider":   "anthropic",
            "signal":     claude_result["signal"].upper(),
            "confidence": claude_result.get("confidence", 5),
        })

    if gpt_result and gpt_result.get("signal"):
        all_votes.append({
            "model_name": "gpt-4o-mini",
            "provider":   "openai",
            "signal":     gpt_result["signal"].upper(),
            "confidence": gpt_result.get("confidence", 5),
        })

    # Add ensemble votes
    for v in ensemble_votes:
        all_votes.append(v)

    if not all_votes:
        return {
            "final_signal": "HOLD",
            "agreement_pct": 0,
            "vote_count": 0,
            "all_votes": [],
            "providers_agree": False,
        }

    # Count votes per signal
    from collections import Counter
    signal_counts = Counter(v["signal"] for v in all_votes)

    # Majority vote (weighted by confidence for tiebreaking)
    final_signal = signal_counts.most_common(1)[0][0]
    agreeing = signal_counts[final_signal]
    total = len(all_votes)
    agreement_pct = round(agreeing / total * 100, 1)

    # Backward-compatible providers_agree (True if >= 60% agree)
    providers_agree = agreement_pct >= 60

    # Combined confidence = avg confidence of models voting for final signal
    agreeing_votes = [v for v in all_votes if v["signal"] == final_signal]
    avg_confidence = round(sum(v["confidence"] for v in agreeing_votes) / len(agreeing_votes))

    return {
        "final_signal":    final_signal,
        "agreement_pct":   agreement_pct,
        "vote_count":      total,
        "all_votes":       all_votes,
        "providers_agree": providers_agree,
        "avg_confidence":  avg_confidence,
        "signal_counts":   dict(signal_counts),
    }
