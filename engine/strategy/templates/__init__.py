"""内置策略模板。"""

from __future__ import annotations

from engine.strategy.templates.dual_moving_average import DualMovingAverageStrategy
from engine.strategy.templates.momentum_rotation import MomentumRotationStrategy

__all__ = ["DualMovingAverageStrategy", "MomentumRotationStrategy"]
