"""回测撮合器（P1-B1）—— 回测可信度的核心。

设计要点：

1. **撮合器是纯函数式的**：给定"订单计划 + 当根 bar + 账户状态"，返回一次成交尝试。
   时间推进、信号与成交错开一个 bar、顺延队列都由引擎主循环（P1-B3）负责，
   撮合器不持有跨 bar 状态——这样每条 A 股规则都能被独立构造用例验证。
2. **它只决定"能否成交、成交多少、什么价"**，不改账户、不落库。
3. 五类约束全部内建，且每条都有专属测试：
   - 停牌 / 零成交 → 顺延
   - 一字涨停买不进、一字跌停卖不出 → 顺延
   - 单笔不超过当日成交量 X%（余量由上层顺延）
   - 资金约束：买入按整手缩量，绝不超可用资金
   - 可卖约束：卖出不超过可用持仓（T+1 由 ``Position.apply_buy`` 保证，这里只看 ``avail_qty``）
4. 限价单语义：限价未触及当根不成交、**不追价**（与实盘挂单一致），
   由上层决定是否撤单重挂。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Final, Literal

from engine.backtest.cost import affordable_qty
from engine.backtest.slippage import NoSlippage, SlippageModel
from engine.num import floor_to_lot, to_decimal
from engine.types import Account, Bar, OrderPlan, Position, Side

_ZERO = Decimal(0)

ExecutionPriceMode = Literal["open", "close", "vwap"]


class FillStatus(StrEnum):
    FILLED = "FILLED"  # 计划全部成交
    PARTIAL = "PARTIAL"  # 部分成交（余量由上层顺延）
    DEFERRED = "DEFERRED"  # 本根不成交，保留至下一根
    EXPIRED = "EXPIRED"  # 顺延超限，放弃（由上层判定并调用）


@dataclass(frozen=True, slots=True)
class FillAttempt:
    """一次撮合尝试的结果。``qty == 0`` 且状态为 DEFERRED 时 ``reason`` 必填。"""

    status: FillStatus
    qty: int = 0
    price: Decimal | None = None
    reason: str = ""

    @property
    def has_fill(self) -> bool:
        return self.qty > 0 and self.price is not None


@dataclass(frozen=True, slots=True)
class MatcherConfig:
    max_vol_ratio: float = 0.10
    price_mode: ExecutionPriceMode = "open"
    lot_size: int = 100


class MatchEngine:
    """无状态撮合器（跨 bar 状态由引擎管理）。"""

    def __init__(
        self,
        config: MatcherConfig | None = None,
        *,
        slippage: SlippageModel | None = None,
    ) -> None:
        self.config = config or MatcherConfig()
        self.slippage = slippage or NoSlippage()

    def attempt(
        self,
        plan: OrderPlan,
        bar: Bar | None,
        *,
        account: Account,
        positions: dict[str, Position],
    ) -> FillAttempt:
        if bar is None:
            return FillAttempt(FillStatus.DEFERRED, reason="无行情数据")
        if bar.is_suspended or bar.volume <= 0:
            return FillAttempt(FillStatus.DEFERRED, reason="停牌或零成交")

        price = self._execution_price(bar, plan.side)
        if plan.side is Side.BUY:
            return self._attempt_buy(plan, bar, price, account=account)
        return self._attempt_sell(plan, bar, price, positions=positions)

    # ============================================================ 内部
    def _execution_price(self, bar: Bar, side: Side) -> Decimal:
        if self.config.price_mode == "close":
            base = bar.close
        elif self.config.price_mode == "vwap" and bar.volume > 0 and bar.amount > 0:
            base = (bar.amount / to_decimal(bar.volume)).quantize(Decimal("0.0001"))
        else:
            base = bar.open
        return self.slippage.apply(base, side)

    def _volume_cap(self, bar: Bar) -> int:
        return int(to_decimal(bar.volume) * to_decimal(str(self.config.max_vol_ratio)))

    def _attempt_buy(
        self,
        plan: OrderPlan,
        bar: Bar,
        price: Decimal,
        *,
        account: Account,
    ) -> FillAttempt:
        if bar.sealed_limit_up:
            return FillAttempt(FillStatus.DEFERRED, reason="一字涨停，无法买入")
        if plan.limit_price is not None and price > plan.limit_price:
            return FillAttempt(FillStatus.DEFERRED, reason="价格未触及限价")

        qty = min(plan.target_qty, self._volume_cap(bar))
        if qty <= 0:
            return FillAttempt(FillStatus.DEFERRED, reason="当日成交量不足，无法买入")
        qty = min(qty, affordable_qty(account.available_cash, price, lot=self.config.lot_size))
        qty = floor_to_lot(qty, self.config.lot_size)
        if qty <= 0:
            return FillAttempt(FillStatus.DEFERRED, reason="可用资金不足")

        if qty >= plan.target_qty:
            return FillAttempt(FillStatus.FILLED, qty=plan.target_qty, price=price)
        return FillAttempt(FillStatus.PARTIAL, qty=qty, price=price)

    def _attempt_sell(
        self,
        plan: OrderPlan,
        bar: Bar,
        price: Decimal,
        *,
        positions: dict[str, Position],
    ) -> FillAttempt:
        if bar.sealed_limit_down:
            return FillAttempt(FillStatus.DEFERRED, reason="一字跌停，无法卖出")
        if plan.limit_price is not None and price < plan.limit_price:
            return FillAttempt(FillStatus.DEFERRED, reason="价格未触及限价")

        qty = min(plan.target_qty, self._volume_cap(bar))
        if qty <= 0:
            return FillAttempt(FillStatus.DEFERRED, reason="当日成交量不足，无法卖出")

        position = positions.get(plan.code)
        avail = position.avail_qty if position is not None else 0
        if avail <= 0:
            # T+1 未解冻 / 全部冻结都可能在次日恢复，因此顺延而不是拒绝
            return FillAttempt(FillStatus.DEFERRED, reason="可卖数量不足（T+1 或冻结）")
        qty = min(qty, avail)

        if qty >= plan.target_qty:
            return FillAttempt(FillStatus.FILLED, qty=plan.target_qty, price=price)
        return FillAttempt(FillStatus.PARTIAL, qty=qty, price=price)


#: 单笔计划顺延的最大天数（超过即 EXPIRED，由引擎调用）
MAX_DEFER_DAYS: Final[int] = 2


__all__ = ["MAX_DEFER_DAYS", "ExecutionPriceMode", "FillAttempt", "FillStatus", "MatchEngine", "MatcherConfig"]
