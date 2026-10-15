"""组合层：把因子得分变成带约束、可执行的目标权重。"""

from __future__ import annotations

from engine.portfolio.allocator import (
    SUPPORTED_SCHEMES,
    allocate,
    normalize,
    realized_volatility,
    top_k,
)
from engine.portfolio.builder import PortfolioBuilder, PortfolioDecision
from engine.portfolio.constraints import (
    PortfolioConstraints,
    apply_no_trade_band,
    cap_weights,
    fit_to_budget,
    limit_turnover,
    scale_to_budget,
    turnover,
)

__all__ = [
    "SUPPORTED_SCHEMES",
    "PortfolioBuilder",
    "PortfolioConstraints",
    "PortfolioDecision",
    "allocate",
    "apply_no_trade_band",
    "cap_weights",
    "fit_to_budget",
    "limit_turnover",
    "normalize",
    "realized_volatility",
    "scale_to_budget",
    "top_k",
    "turnover",
]
