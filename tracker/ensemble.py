"""
tracker/ensemble.py -- Multi-model ensemble voting with weighted consensus.

9 models total (Claude + GPT + 7 OpenRouter).
All models receive the same full context (technical, macro, news, calendar).

Weighting tiers:
  Claude sonnet-4-6          weight 3.0  -- primary, extended thinking
  GPT-4o                     weight 2.0  -- strong secondary
  Llama 70B, DeepSeek, Qwen,
  Gemini Flash, Grok-3-mini  weight 1.0  -- large capable models
  Mistral Nemo,
  Mistral Small              weight 0.5  -- smaller models
"""

import os
import time
import json
import logging
log = logging.getLogger("ensemble")

USD_TO_GBP = 0.79

# OpenRouter model pricing (USD per 1M tokens, input/output)
OPENROUTER_PRICING = {
    "meta-llama/llama-3.3-70b-instruct":        (0.100, 0.100),
    "deepseek/deepseek-chat":                   (0.270, 1.100),
    "qwen/qwen-2.5-72b-instruct":               (0.180, 0.180),
    "google/gemini-2.0-flash-001":              (0.100, 0.400),
    "x-ai/grok-3-mini-beta":                    (0.300, 0.500),
    "mistralai/mistral-nemo":                   (0.035, 0.080),
    "mistralai/mistral-small-3.1-24b-instruct": (0.100, 0.300),
}

# Vote weights per model
# Claude and GPT weights are applied in calculate_consensus()
OPENROUTER_WEIGHTS = {
    "meta-llama/llama-3.3-70b-instruct":        1.0,   # 70B -- full vote
    "deepseek/deepseek-chat":                   1.0,   # strong reasoner -- full vote
    "qwen/qwen-2.5-72b-instruct":               1.0,   # 72B -- full vote
    "google/gemini-2.0-flash-001":              1.0,   # Gemini Flash -- full vote
    "x-ai/grok-3-mini-beta":                    1.0,   # Grok, real-time data -- full vote
    "mistralai/mistral-nemo":                   0.5,   # smaller -- half vote
    "mistralai/mistral-small-3.1-24b-instruct": 0.5,   # smaller -- half vote
}

MAIN_MODEL_WEIGHTS = {
    "claude-sonnet-4-6": 3.0,
    "gpt-4o":            2.0,
}

OPENROUTER_MODELS = list(OPENROUTER_PRICING.keys())

SYSTEM_PROMPT = """You are an expert forex analyst specialising in GBP/USD (British Pound / US Dollar).
You have been given comprehensive technical, macro-economic and news data.
Analyse all the information provided and give a trading signal.

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
    """
    Build the full context prompt -- same data Claude and GPT receive.
    Technical indicators + macro data + news + economic calendar.
    """
    ctx  = context_data or {}
    lines = ["Analyse GBP/USD and provide a trading signal.\n"]

    # --- Technical data ---
    if price_data:
        lines.append("=== TECHNICAL DATA ===")
        lines.append(f"Current price:      {price_data.get('current_price', 0):.5f}")
        lines.append(f"50 MA:              {price_data.get('price_50ma', 0):.5f}")
        lines.append(f"200 MA:             {price_data.get('price_200ma', 0):.5f}")
        lines.append(f"Above 200MA:        {price_data.get('above_200ma', False)}")
        lines.append(f"MA alignment:       {price_data.get('ma_alignment', 'N/A')}")
        lines.append(f"Trend:              {price_data.get('trend_direction', 'N/A')}")
        lines.append(f"ADX:                {price_data.get('adx_value', 0):.1f} ({price_data.get('trend_strength', 'N/A')})")
        lines.append(f"RSI(14):            {price_data.get('rsi_value', 0):.1f} ({price_data.get('rsi_condition', 'N/A')})")
        lines.append(f"RSI divergence:     {price_data.get('rsi_divergence', 'none')}")
        lines.append(f"Nearest support:    {price_data.get('nearest_support', 0):.5f} ({price_data.get('distance_to_support_pips', 0):.0f} pips)")
        lines.append(f"Nearest resistance: {price_data.get('nearest_resistance', 0):.5f} ({price_data.get('distance_to_resistance_pips', 0):.0f} pips)")
        if price_data.get('at_key_level'):
            lines.append(f"At key level:       {price_data.get('key_level_type', 'N/A')} at {price_data.get('key_level_price', 0):.5f}")

    # --- Macro / market data ---
    if market_data:
        lines.append("\n=== MACRO DATA ===")
        if market_data.get('dxy_trend'):
            lines.append(f"DXY (Dollar Index): {market_data['dxy_trend']} ({market_data.get('dxy_1day_change_pct', 0):.2f}% today)")
        if market_data.get('yield_spread') is not None:
            lines.append(f"UK/US yield spread: {market_data.get('yield_spread', 0):.3f}% ({market_data.get('spread_direction', 'N/A')})")
        if market_data.get('us_10yr'):
            lines.append(f"US 10yr yield:      {market_data['us_10yr']:.3f}%")
        if market_data.get('uk_10yr'):
            lines.append(f"UK 10yr yield:      {market_data['uk_10yr']:.3f}%")

    # --- News and sentiment ---
    news_text = ctx.get('news_text') or ctx.get('technical_context', '')
    if news_text and len(str(news_text)) > 20:
        lines.append("\n=== NEWS & SENTIMENT ===")
        # Trim to avoid token bloat -- keep first 800 chars
        lines.append(str(news_text)[:800])

    # --- Economic calendar ---
    calendar_text = ctx.get('calendar_text', '')
    if calendar_text and len(str(calendar_text)) > 20:
        lines.append("\n=== ECONOMIC CALENDAR ===")
        lines.append(str(calendar_text)[:400])

    lines.append("\nRespond with JSON only.")
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

        # Strip markdown code blocks if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        data       = json.loads(content.strip())
        signal     = str(data.get("signal", "HOLD")).upper()
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"
        confidence = max(1, min(10, int(data.get("confidence", 5))))
        reasoning  = str(data.get("reasoning", ""))[:300]

        in_tokens  = response.usage.prompt_tokens     if response.usage else 0
        out_tokens = response.usage.completion_tokens if response.usage else 0
        price_in, price_out = OPENROUTER_PRICING.get(model_name, (0.5, 1.5))
        cost_usd   = (in_tokens * price_in + out_tokens * price_out) / 1_000_000
        cost_gbp   = round(cost_usd * USD_TO_GBP, 6)
        weight     = OPENROUTER_WEIGHTS.get(model_name, 1.0)

        return {
            "model_name":    model_name,
            "provider":      "openrouter",
            "signal":        signal,
            "confidence":    confidence,
            "reasoning":     reasoning,
            "cost_usd":      round(cost_usd, 6),
            "cost_gbp":      cost_gbp,
            "latency_ms":    latency_ms,
            "input_tokens":  in_tokens,
            "output_tokens": out_tokens,
            "weight":        weight,
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
    log.info(f"Ensemble prompt: ~{len(prompt)//4} tokens")
    votes = []

    for model in OPENROUTER_MODELS:
        log.info(f"Calling ensemble model: {model} (weight={OPENROUTER_WEIGHTS.get(model, 1.0)})")
        result = call_model(model, prompt, api_key)
        if result:
            votes.append(result)
            log.info(f"  {model}: {result['signal']} {result['confidence']}/10  weight={result['weight']}")
        else:
            log.warning(f"  {model}: failed -- skipping")

    return votes


def calculate_consensus(claude_result, gpt_result, ensemble_votes):
    """
    Weighted consensus across all models.

    Weights:
      Claude sonnet-4-6  = 3.0
      GPT-4o-mini        = 2.0
      OpenRouter Tier 1  = 1.0  (llama, deepseek, qwen, gemini-flash)
      OpenRouter Tier 2  = 0.5  (mistral-nemo, flash-lite, mistral-small)

    Returns:
      final_signal, agreement_pct (weighted), vote_count, all_votes,
      providers_agree, avg_confidence, signal_counts, weighted_scores
    """
    all_votes = []

    if claude_result and claude_result.get("signal"):
        all_votes.append({
            "model_name": "claude-sonnet-4-6",
            "provider":   "anthropic",
            "signal":     claude_result["signal"].upper(),
            "confidence": claude_result.get("confidence", 5),
            "weight":     MAIN_MODEL_WEIGHTS["claude-sonnet-4-6"],
        })

    if gpt_result and gpt_result.get("signal"):
        all_votes.append({
            "model_name": "gpt-4o-mini",
            "provider":   "openai",
            "signal":     gpt_result["signal"].upper(),
            "confidence": gpt_result.get("confidence", 5),
            "weight":     MAIN_MODEL_WEIGHTS["gpt-4o-mini"],
        })

    for v in ensemble_votes:
        if "weight" not in v:
            v["weight"] = OPENROUTER_WEIGHTS.get(v.get("model_name", ""), 1.0)
        all_votes.append(v)

    if not all_votes:
        return {
            "final_signal":    "HOLD",
            "agreement_pct":   0,
            "vote_count":      0,
            "all_votes":       [],
            "providers_agree": False,
            "avg_confidence":  5,
            "signal_counts":   {},
            "weighted_scores": {},
        }

    # --- Weighted vote tally ---
    weighted_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    total_weight    = 0.0

    for v in all_votes:
        w = float(v.get("weight", 1.0))
        sig = v["signal"]
        weighted_scores[sig] = weighted_scores.get(sig, 0.0) + w
        total_weight += w

    final_signal   = max(weighted_scores, key=weighted_scores.get)
    winning_weight = weighted_scores[final_signal]
    agreement_pct  = round(winning_weight / total_weight * 100, 1) if total_weight > 0 else 0

    # Raw vote counts for display
    from collections import Counter
    raw_counts = Counter(v["signal"] for v in all_votes)

    providers_agree = agreement_pct >= 60

    agreeing_votes  = [v for v in all_votes if v["signal"] == final_signal]
    avg_confidence  = round(
        sum(v["confidence"] * v.get("weight", 1.0) for v in agreeing_votes) /
        sum(v.get("weight", 1.0) for v in agreeing_votes)
    ) if agreeing_votes else 5

    log.info(
        f"Weighted consensus: {final_signal} ({agreement_pct}% weighted) "
        f"scores={weighted_scores} total_weight={total_weight:.1f}"
    )

    return {
        "final_signal":    final_signal,
        "agreement_pct":   agreement_pct,
        "vote_count":      len(all_votes),
        "all_votes":       all_votes,
        "providers_agree": providers_agree,
        "avg_confidence":  avg_confidence,
        "signal_counts":   dict(raw_counts),
        "weighted_scores": {k: round(v, 2) for k, v in weighted_scores.items()},
    }
