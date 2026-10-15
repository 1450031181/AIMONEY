"""信号 → 订单计划（P1-B3）。

语义：**目标权重再平衡**。
- ``target_weight < 0``（SELL 信号）：清仓该标的（A 股不允许做空，权重绝对值仅作提示）；
- ``target_weight > 0``（BUY 信号）：把持仓调整到 ``权重 × 总资产``，差额为正则买入、
  为负则减仓；
- ``HOLD`` 忽略。

用"当日收盘价"做数量测算是安全的：信号在 T 日收盘后产生、T+1 日开盘成交，
收盘价在决策时点已知——这不是未来函数。真正的成交价由撮合器用 T+1 开盘价决定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from engine.num import floor_to_lot, to_decimal
from engine.types import Account, Bar, Direction, OrderPlan, Position, Side, Signal

_logger = logging.getLogger(__name__)

_ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    lot_size: int = 100
    #: 低于该金额的调仓忽略（避免产生"买 1 手还付最低佣金"的碎单）
    min_trade_value: Decimal = Decimal("1000")
    min_sell_value: Decimal = Decimal("100")


class TargetWeightPlanner:
    """把目标权重信号转成订单计划（纯函数，不改账户状态）。"""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def build(
        self,
        signals: list[Signal],
        *,
        trade_date: date,
        account: Account,
        positions: dict[str, Position],
        bars: dict[str, Bar],
    ) -> list[OrderPlan]:
        plans: list[OrderPlan] = []
        total_asset = account.total_asset
        if total_asset <= 0:
            _logger.warning("总资产非正，跳过全部信号")
            return plans

        for signal in signals:
            if signal.direction is Direction.HOLD:
                continue
            bar = bars.get(signal.code)
            if bar is None or bar.close <= 0:
                _logger.warning("信号被跳过（无有效行情）：%s %s", signal.code, signal.direction)
                continue

            price = bar.close
            position = positions.get(signal.code)
            current_qty = position.qty if position is not None else 0
            current_value = price * current_qty

            if signal.direction is Direction.SELL:
                plan = self._plan_reduce_to_zero(signal, trade_date, current_qty, price)
            else:
                # 权重是 float，必须先转 Decimal 再与金额相乘（float × Decimal 会直接 TypeError）
                plan = self._plan_rebalance_to_weight(
                    signal,
                    trade_date,
                    to_decimal(abs(signal.target_weight)) * total_asset,
                    current_qty,
                    current_value,
                    price,
                )
            if plan is not None:
                plans.append(plan)
        return plans

    # -------------------------------------------------- 内部
    def _plan_reduce_to_zero(
        self,
        signal: Signal,
        trade_date: date,
        current_qty: int,
        price: Decimal,
    ) -> OrderPlan | None:
        if current_qty <= 0:
            return None
        if price * current_qty < self.config.min_sell_value:
            _logger.info("清仓金额低于最小阈值，忽略：%s", signal.code)
            return None
        return OrderPlan(
            code=signal.code,
            strategy_id=signal.strategy_id,
            trade_date=trade_date,
            side=Side.SELL,
            target_qty=current_qty,
            reason=signal.reason or "目标权重为负，清仓",
        )

    def _plan_rebalance_to_weight(
        self,
        signal: Signal,
        trade_date: date,
        target_value: Decimal,
        current_qty: int,
        current_value: Decimal,
        price: Decimal,
    ) -> OrderPlan | None:
        delta = target_value - current_value
        if abs(delta) < self.config.min_trade_value:
            return None

        if delta > 0:
            qty = floor_to_lot(int(delta / price), self.config.lot_size)
            if qty <= 0 or price * qty < self.config.min_trade_value:
                return None
            return OrderPlan(
                code=signal.code,
                strategy_id=signal.strategy_id,
                trade_date=trade_date,
                side=Side.BUY,
                target_qty=qty,
                reason=signal.reason or f"调仓至目标权重 {signal.target_weight:.2%}",
            )

        qty = min(current_qty, int(-delta / price))
        if qty <= 0 or price * qty < self.config.min_sell_value:
            return None
        return OrderPlan(
            code=signal.code,
            strategy_id=signal.strategy_id,
            trade_date=trade_date,
            side=Side.SELL,
            target_qty=qty,
            reason=signal.reason or "调仓减仓",
        )


__all__ = ["PlannerConfig", "TargetWeightPlanner"]
