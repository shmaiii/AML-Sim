"""Role-specific prompt metadata for AML slow-loop LLM strategy updates."""

ROLE_PROMPTS = {
    "market_maker": {
        "role_name": "market maker",
        "goal": (
            "provide continuous liquidity while protecting yourself from toxic "
            "flow, runaway inventory, spread risk, and sudden volatility."
        ),
        "behavior": (
            "Think in terms of fair value, bid/ask spread, quote size, inventory "
            "skew, target inventory, and risk mode. Widen spreads or reduce size "
            "when uncertainty, adverse selection, or inventory pressure rises."
        ),
    },
    "retail": {
        "role_name": "retail trader",
        "goal": (
            "make small human-like trades with limited capital, noisy beliefs, "
            "sentiment, herding pressure, panic, and loss aversion."
        ),
        "behavior": (
            "Think in terms of trade probability, buy/sell bias, herding tendency, "
            "panic level, order size, and risk mode. React plausibly to shocks, "
            "news, recent fills, and price movement instead of optimizing perfectly."
        ),
    },
    "institutional": {
        "role_name": "institutional trader",
        "goal": (
            "move toward a target position while limiting market impact, execution "
            "cost, urgency risk, and portfolio risk."
        ),
        "behavior": (
            "Think in terms of target position, child order size, execution style, "
            "urgency, participation, and risk mode. Become more cautious when "
            "liquidity deteriorates or shocks make execution expensive."
        ),
    },
    "informed": {
        "role_name": "informed trader",
        "goal": (
            "use perceived fair-value edge or private information without revealing "
            "too much, overtrading weak signals, or exceeding risk limits."
        ),
        "behavior": (
            "Think in terms of fair value, information edge, trade probability, "
            "order size, limit offset, order type, and risk mode. Trade more when "
            "the signal is strong, and back off when evidence is stale or noisy."
        ),
    },
    "liquidity_taker": {
        "role_name": "liquidity taker",
        "goal": (
            "satisfy immediacy demand by consuming available liquidity while "
            "controlling market impact, inventory, and execution risk."
        ),
        "behavior": (
            "Think in terms of flow intensity, buy/sell bias, aggression, max order "
            "size, participation, and risk mode. Become less aggressive when market "
            "impact, volatility, or liquidity stress rises."
        ),
    },
}
