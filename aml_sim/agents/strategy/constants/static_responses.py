"""Static LLM-shaped responses for offline slow-loop tests."""

STATIC_MARKET_MAKER_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "spread": 0.25,
        "quote_size": 100,
        "inventory_skew": 0.0015,
    },
    "confidence": 0.75,
    "reason": "Static market-maker LLM test response: quote slightly wider and manage inventory conservatively.",
}


STATIC_RETAIL_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "trade_probability": 0.35,
        "buy_bias": 0.52,
        "herding_tendency": 0.15,
        "panic_level": 0.05,
    },
    "confidence": 0.7,
    "reason": "Static retail LLM test response: slightly active, mildly bullish, low panic.",
}


STATIC_INSTITUTIONAL_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "child_order_size": 100,
        "execution_style": "sliced",
        "urgency": 0.6,
    },
    "confidence": 0.78,
    "reason": "Static institutional LLM test response: keep sliced execution with moderate urgency.",
}


STATIC_INFORMED_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "trade_probability": 0.38,
        "information_edge": 0.72,
    },
    "confidence": 0.74,
    "reason": "Static informed-trader LLM test response: keep trading only when the private value signal is strong.",
}


STATIC_LIQUIDITY_TAKER_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "flow_intensity": 0.38,
        "aggression": 0.75,
    },
    "confidence": 0.7,
    "reason": "Static liquidity-taker LLM test response: maintain steady aggressive flow with bounded size.",
}


STATIC_RESPONSES_BY_ROLE = {
    "market_maker": STATIC_MARKET_MAKER_RESPONSE,
    "retail": STATIC_RETAIL_RESPONSE,
    "institutional": STATIC_INSTITUTIONAL_RESPONSE,
    "informed": STATIC_INFORMED_RESPONSE,
    "liquidity_taker": STATIC_LIQUIDITY_TAKER_RESPONSE,
}
