"""内置策略模板。"""

from __future__ import annotations

from engine.strategy.templates.dual_moving_average import DualMovingAverageStrategy
from engine.strategy.templates.momentum_rotation import MomentumRotationStrategy
from engine.strategy.templates.multi_factor_enhance import MultiFactorEnhanceStrategy

__all__ = ["DualMovingAverageStrategy", "MomentumRotationStrategy", "MultiFactorEnhanceStrategy"]
