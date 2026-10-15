"""动量轮动策略模板（P1-B5）。

逻辑：每 ``rebalance_interval`` 个交易日调仓一次——
按 ``lookback`` 日动量（区间收益）对股票池排序，取动量为正的前 ``top_k`` 名等权持有；
原持仓中跌出名单的清仓；动量全部为负则空仓等待。

这是最接近实盘可用的模板：有调仓节奏（控制换手）、有组合层面约束（top_k 等权）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from engine.errors import StrategyError
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class MomentumRotationStrategy(BaseStrategy):
    """动量轮动策略（定期调仓、top_k 等权）。"""

    def __init__(self, *, strategy_id: str = "momentum_rotation", params: Mapping[str, Any] | None = None) -> None:
        super().__init__(
            strategy_id=strategy_id,
            params={
                "lookback": 10,
                "top_k": 2,
                "rebalance_interval": 5,
                "total_weight": 0.95,
                **(params or {}),
            },
        )
        self._day_counter = 0

    def on_bar(self, context: Any, bars: Any) -> list[Signal]:
        lookback = int(self.param("lookback"))
        top_k = int(self.param("top_k"))
        interval = int(self.param("rebalance_interval"))
        total_weight = float(self.param("total_weight"))
        if lookback <= 0 or top_k <= 0 or interval <= 0:
            raise StrategyError(
                "动量轮动参数必须为正",
                detail={"lookback": lookback, "top_k": top_k, "rebalance_interval": interval},
            )

        self._day_counter += 1
        if (self._day_counter - 1) % interval != 0:
            return []  # 非调仓日不动（控制换手率）

        scored: list[tuple[str, float]] = []
        for code in sorted(bars):
            closes = self.closes(context, code, window=lookback + 1)
            if len(closes) < lookback + 1:
                continue  # 历史不足不参与排名（次新股天然被排除）
            momentum = float(closes[-1] / closes[0] - 1)
            if momentum > 0:
                scored.append((code, momentum))

        selected = [code for code, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]]
        selected_set = set(selected)

        signals: list[Signal] = []
        per_weight = total_weight / len(selected) if selected else 0.0
        for code in selected:
            signals.append(
                Signal(
                    code=code,
                    strategy_id=context.strategy_id,
                    trade_date=context.trade_date,
                    direction=Direction.BUY,
                    target_weight=per_weight,
                    reason=f"动量前 {top_k}（等权 {per_weight:.2%}）",
                )
            )
        # 已持有但跌出名单的清仓
        for code in sorted(context.positions()):
            if code not in selected_set:
                signals.append(
                    Signal(
                        code=code,
                        strategy_id=context.strategy_id,
                        trade_date=context.trade_date,
                        direction=Direction.SELL,
                        target_weight=-1.0,
                        reason="跌出动量名单，清仓",
                    )
                )
        return signals


__all__ = ["MomentumRotationStrategy"]
