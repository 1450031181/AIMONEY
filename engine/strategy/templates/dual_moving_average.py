"""双均线策略模板（P1-B5）。

逻辑：对股票池内每只标的独立判断——
- 快线上穿慢线（金叉）→ 买入 ``target_weight``；
- 快线下穿慢线（死叉）→ 清仓；
- 其余情况不动。

这是一个**教学/冒烟模板**：逐标的独立判断、不看组合集中度，
实盘意义有限，但足够验证回测引擎的每一环。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from engine.errors import StrategyError
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class DualMovingAverageStrategy(BaseStrategy):
    """双均线金叉/死叉策略。"""

    def __init__(self, *, strategy_id: str = "dual_ma", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            strategy_id=strategy_id,
            params={"fast": 5, "slow": 20, "target_weight": 0.95, **(params or {})},
        )

    def on_bar(self, context: Any, bars: Any) -> list[Signal]:
        fast = int(self.param("fast"))
        slow = int(self.param("slow"))
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise StrategyError(
                "双均线参数非法：要求 0 < fast < slow",
                detail={"fast": fast, "slow": slow},
            )
        target_weight = float(self.param("target_weight"))

        signals: list[Signal] = []
        for code in sorted(bars):
            closes = self.closes(context, code, window=slow + 1)
            if len(closes) < slow + 1:
                continue  # 历史不足：金叉判定不可信，跳过而不是用短窗口硬算
            fast_now = self.mean(closes[-fast:])
            slow_now = self.mean(closes[-slow:])
            fast_prev = self.mean(closes[-fast - 1 : -1])
            slow_prev = self.mean(closes[-slow - 1 : -1])

            golden_cross = fast_prev <= slow_prev and fast_now > slow_now
            death_cross = fast_prev >= slow_prev and fast_now < slow_now
            if golden_cross:
                signals.append(self._signal(code, context, Direction.BUY, target_weight, f"金叉 fast={fast} slow={slow}"))
            elif death_cross:
                signals.append(self._signal(code, context, Direction.SELL, -1.0, f"死叉 fast={fast} slow={slow}"))
        return signals

    @staticmethod
    def _signal(code: str, context: Any, direction: Direction, weight: float, reason: str) -> Signal:
        return Signal(
            code=code,
            strategy_id=context.strategy_id,
            trade_date=context.trade_date,
            direction=direction,
            target_weight=weight,
            reason=reason,
        )


__all__ = ["DualMovingAverageStrategy"]
