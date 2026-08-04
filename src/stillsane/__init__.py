"""stillsane -- a drift canary for deployed LLM apps and agents."""

#: Single source of truth. `pyproject.toml` reads this via hatchling's version
#: hook, so the packaged version and `stillsane --version` cannot disagree.
__version__ = "0.0.5"
