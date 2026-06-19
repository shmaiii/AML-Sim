"""AML-specific trading agents owned by AML-Sim."""

__all__ = [
    "AMLInstitutionalTrader",
    "AMLMarketMakerTrader",
    "AMLRetailTrader",
]


def __getattr__(name: str):
    if name == "AMLInstitutionalTrader":
        from aml_sim.agents.institutional_trader import AMLInstitutionalTrader

        return AMLInstitutionalTrader
    if name == "AMLMarketMakerTrader":
        from aml_sim.agents.market_maker_trader import AMLMarketMakerTrader

        return AMLMarketMakerTrader
    if name == "AMLRetailTrader":
        from aml_sim.agents.retail_trader import AMLRetailTrader

        return AMLRetailTrader

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
