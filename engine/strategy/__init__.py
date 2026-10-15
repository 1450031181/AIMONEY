"""策略层：基类、内置模板与用户策略加载器。"""

from __future__ import annotations

from engine.strategy.base import BaseStrategy
from engine.strategy.loader import LoadedStrategy, load_strategy
from engine.strategy.templates.dual_moving_average import DualMovingAverageStrategy
from engine.strategy.templates.momentum_rotation import MomentumRotationStrategy
from engine.strategy.templates.multi_factor_enhance import MultiFactorEnhanceStrategy

__all__ = [
    "BaseStrategy",
    "DualMovingAverageStrategy",
    "LoadedStrategy",
    "MomentumRotationStrategy",
    "MultiFactorEnhanceStrategy",
    "load_strategy",
]
