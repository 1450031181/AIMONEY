"""风控规则实现（P3-2）。

分两类（对应 ``engine.ports`` 里的两个协议）：

- **事前**（``check``，逐条订单计划）：急停、风险等级、单票集中度、单笔金额、
  当日委托笔数、限价合法性、可用资金、可卖数量。
- **事中**（``check_portfolio``，组合级）：最大回撤、单日亏损，触线降级为
  ``HALT``（只平不开）或 ``FLATTEN``（清仓）。

三条贯穿全局的原则：

1. **能缩量就不要拒单**：集中度/金额类约束本质是"买太多"，缩到合规数量即可完成交易；
   直接拒单会让策略以为"信号无效"，而实际只是仓位太大。
2. **卖出尽量放行**：风控的目标是控制风险暴露，**降低暴露的操作不该被拦**
   （急停与 FLATTEN 都允许卖出，只有买入被拒）。
3. **数据缺失时不做假设**：拿不到价格/涨跌停就不做该条校验，并在裁定里说明
   规则未生效——而不是拿一个猜出来的数字硬算。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from engine.datahub.adjust import restore_real_price
from engine.datahub.rules import limit_prices
from engine.errors import ValidationError
from engine.num import floor_to_lot, to_decimal
from engine.ports import IntradayRiskRule, PortfolioSnapshot, PreTradeRiskRule, RiskContext
from engine.risk.kill_switch import KillSwitch
from engine.types import Bar, Position, RiskDecision, RiskLevel, Side

_ZERO: Final[Decimal] = Decimal(0)
#: 集中度/金额类规则缩量时的整手粒度（A 股买入必须是 100 股整数倍）
LOT_SIZE: Final[int] = 100
#: 单笔委托金额上限的默认值（20 万元）
DEFAULT_MAX_ORDER_VALUE: Final[Decimal] = Decimal("200000")


# ============================================================ 通用小工具
def _real_price(bar: Bar | None) -> Decimal | None:
    """由行情推导**真实价**（下单与风控必须用真实价，不能用后复权价）。"""
    if bar is None or bar.close <= 0:
        return None
    try:
        return restore_real_price(bar.close, bar.adj_factor)
    except ValidationError:
        return None


def _market_value(qty: int, price: Decimal | None) -> Decimal:
    if price is None or qty <= 0:
        return _ZERO
    return price * qty


@dataclass(frozen=True, slots=True)
class _Base:
    name: str = "base"
    priority: int = 100


# ============================================================ 事前规则
@dataclass(frozen=True, slots=True)
class KillSwitchRule(_Base):
    """急停激活时拒买、放行卖出（降低暴露的操作不该被拦）。"""

    switch: KillSwitch | None = None
    name: str = "kill_switch"
    priority: int = 0

    def check(self, ctx: RiskContext) -> RiskDecision:
        if self.switch is None or not self.switch.is_active():
            return RiskDecision.allow(ctx.plan.target_qty)
        state = self.switch.state()
        reason = str(state.get("reason") or "急停开关已激活")
        if ctx.plan.side is Side.SELL:
            return RiskDecision.allow(ctx.plan.target_qty)
        return RiskDecision.reject(self.name, f"急停开关生效：{reason}", requested_qty=ctx.plan.target_qty)


@dataclass(frozen=True, slots=True)
class RiskLevelRule(_Base):
    """按风险等级放行：HALT 只平不开，FLATTEN 仅允许卖出。"""

    name: str = "risk_level"
    priority: int = 1

    def check(self, ctx: RiskContext) -> RiskDecision:
        qty = ctx.plan.target_qty
        if ctx.risk_level is RiskLevel.NORMAL or ctx.risk_level is RiskLevel.WARN:
            return RiskDecision.allow(qty)
        if ctx.plan.side is Side.SELL:
            return RiskDecision.allow(qty)
        if ctx.risk_level is RiskLevel.HALT:
            return RiskDecision.reject(self.name, "风控等级 HALT：只平不开", requested_qty=qty)
        return RiskDecision.reject(self.name, "风控等级 FLATTEN：仅允许卖出", requested_qty=qty)


@dataclass(frozen=True, slots=True)
class PositionConcentrationRule(_Base):
    """单票集中度上限：买入后该票市值 / 总资产不得超过 ``max_weight``。

    超限时**缩量**到上限而不是拒单：策略的信号可能没问题，只是仓位太大。
    """

    max_weight: float = 0.2
    name: str = "position_concentration"
    priority: int = 10

    def check(self, ctx: RiskContext) -> RiskDecision:
        if ctx.plan.side is not Side.BUY:
            return RiskDecision.allow(ctx.plan.target_qty)
        price = _real_price(ctx.bar_of(ctx.plan.code))
        total_asset = ctx.account.total_asset
        if price is None or total_asset <= 0:
            return RiskDecision.allow(ctx.plan.target_qty)  # 无价可算 → 本规则不生效

        position: Position | None = ctx.positions.get(ctx.plan.code)
        held_qty = position.qty if position is not None else 0
        cap_value = to_decimal(self.max_weight) * total_asset
        allowed_qty = int((cap_value - _market_value(held_qty, price)) / price)
        allowed_qty = floor_to_lot(allowed_qty, LOT_SIZE)

        if allowed_qty >= ctx.plan.target_qty:
            return RiskDecision.allow(ctx.plan.target_qty)
        if allowed_qty <= 0:
            return RiskDecision.reject(
                self.name,
                f"单票集中度已达上限 {self.max_weight:.0%}",
                requested_qty=ctx.plan.target_qty,
            )
        return RiskDecision.shrink(
            self.name,
            f"单票集中度上限 {self.max_weight:.0%}：{ctx.plan.target_qty} → {allowed_qty}",
            requested_qty=ctx.plan.target_qty,
            adjusted_qty=allowed_qty,
        )


@dataclass(frozen=True, slots=True)
class OrderValueRule(_Base):
    """单笔委托金额上限（按真实价估算），超限缩量。"""

    # 用模块级常量而不是 to_decimal("200000")：dataclass 默认值里不做函数调用
    max_value: Decimal = DEFAULT_MAX_ORDER_VALUE
    name: str = "order_value"
    priority: int = 11

    def check(self, ctx: RiskContext) -> RiskDecision:
        price = _real_price(ctx.bar_of(ctx.plan.code))
        if price is None or price <= 0:
            return RiskDecision.allow(ctx.plan.target_qty)
        allowed_qty = int(self.max_value / price)
        if ctx.plan.side is Side.BUY:
            allowed_qty = floor_to_lot(allowed_qty, LOT_SIZE)
        if allowed_qty >= ctx.plan.target_qty:
            return RiskDecision.allow(ctx.plan.target_qty)
        if allowed_qty <= 0:
            return RiskDecision.reject(
                self.name, f"单笔委托金额超过上限 {self.max_value}", requested_qty=ctx.plan.target_qty
            )
        return RiskDecision.shrink(
            self.name,
            f"单笔委托金额上限 {self.max_value}：{ctx.plan.target_qty} → {allowed_qty}",
            requested_qty=ctx.plan.target_qty,
            adjusted_qty=allowed_qty,
        )


@dataclass(frozen=True, slots=True)
class DailyOrderCountRule(_Base):
    """当日委托笔数上限（``ctx.extra["orders_today"]`` 由调用方维护）。"""

    max_orders: int = 100
    name: str = "daily_order_count"
    priority: int = 12

    def check(self, ctx: RiskContext) -> RiskDecision:
        used = int(ctx.extra.get("orders_today", 0))
        if used < self.max_orders:
            return RiskDecision.allow(ctx.plan.target_qty)
        return RiskDecision.reject(
            self.name,
            f"当日委托笔数已达上限 {self.max_orders}（已用 {used}）",
            requested_qty=ctx.plan.target_qty,
        )


@dataclass(frozen=True, slots=True)
class PriceLimitRule(_Base):
    """限价单必须落在当日涨跌停区间内（超出即非法委托，交易所会直接废单）。

    涨跌停优先取行情自带字段；缺失时按前收 × 板块阈值推算（T-09 的规则），
    两者都拿不到则本规则不生效并如实说明。
    """

    name: str = "price_limit"
    priority: int = 13

    def check(self, ctx: RiskContext) -> RiskDecision:
        plan = ctx.plan
        if plan.limit_price is None:
            return RiskDecision.allow(plan.target_qty)
        bar = ctx.bar_of(plan.code)
        if bar is None:
            return RiskDecision.allow(plan.target_qty)
        upper, lower = _limit_prices(bar, is_st=bool(ctx.extra.get("is_st", False)))
        if upper is None or lower is None:
            return RiskDecision.allow(plan.target_qty)
        if lower <= plan.limit_price <= upper:
            return RiskDecision.allow(plan.target_qty)
        return RiskDecision.reject(
            self.name,
            f"限价 {plan.limit_price} 超出涨跌停区间 [{lower}, {upper}]",
            requested_qty=plan.target_qty,
        )


@dataclass(frozen=True, slots=True)
class CashRule(_Base):
    """可用资金校验：不足则缩量到含费可买的最大整手数。

    回测里撮合器也会做同样的校验，**重复校验是刻意的**：实盘没有撮合器兜底，
    风控是唯一拦住"超资金下单"的地方。
    """

    name: str = "cash"
    priority: int = 14

    def check(self, ctx: RiskContext) -> RiskDecision:
        if ctx.plan.side is not Side.BUY:
            return RiskDecision.allow(ctx.plan.target_qty)
        from engine.backtest.cost import affordable_qty

        price = _real_price(ctx.bar_of(ctx.plan.code))
        if price is None or price <= 0:
            return RiskDecision.allow(ctx.plan.target_qty)
        allowed = affordable_qty(ctx.account.available_cash, price, lot=LOT_SIZE)
        if allowed >= ctx.plan.target_qty:
            return RiskDecision.allow(ctx.plan.target_qty)
        if allowed <= 0:
            return RiskDecision.reject(
                self.name,
                f"可用资金不足（需 {price * ctx.plan.target_qty}，可用 {ctx.account.available_cash}）",
                requested_qty=ctx.plan.target_qty,
            )
        return RiskDecision.shrink(
            self.name,
            f"可用资金不足，缩量至 {allowed} 股",
            requested_qty=ctx.plan.target_qty,
            adjusted_qty=allowed,
        )


@dataclass(frozen=True, slots=True)
class AvailablePositionRule(_Base):
    """可卖数量校验（T+1 与冻结都在 ``avail_qty`` 里体现）。"""

    name: str = "available_position"
    priority: int = 15

    def check(self, ctx: RiskContext) -> RiskDecision:
        if ctx.plan.side is not Side.SELL:
            return RiskDecision.allow(ctx.plan.target_qty)
        position = ctx.positions.get(ctx.plan.code)
        available = position.avail_qty if position is not None else 0
        if available >= ctx.plan.target_qty:
            return RiskDecision.allow(ctx.plan.target_qty)
        if available <= 0:
            return RiskDecision.reject(
                self.name,
                "无可卖数量（T+1 未解冻或已冻结）",
                requested_qty=ctx.plan.target_qty,
            )
        return RiskDecision.shrink(
            self.name,
            f"可卖数量不足：{ctx.plan.target_qty} → {available}",
            requested_qty=ctx.plan.target_qty,
            adjusted_qty=available,
        )


def _limit_prices(bar: Bar, *, is_st: bool) -> tuple[Decimal | None, Decimal | None]:
    if bar.limit_up is not None and bar.limit_down is not None:
        return restore_real_price(bar.limit_up, bar.adj_factor), restore_real_price(bar.limit_down, bar.adj_factor)
    if bar.prev_close is None or bar.prev_close <= 0:
        return None, None
    previous = restore_real_price(bar.prev_close, bar.adj_factor)
    try:
        return limit_prices(previous, code=bar.code, is_st=is_st)
    except ValidationError:
        return None, None


# ============================================================ 事中规则
@dataclass(frozen=True, slots=True)
class DrawdownRule(_Base):
    """组合回撤触线 → HALT（只平不开）。``drawdown`` 以负数表示。"""

    max_drawdown: float = 0.1
    name: str = "drawdown"
    priority: int = 0

    def check_portfolio(self, snapshot: PortfolioSnapshot) -> RiskDecision:
        if abs(snapshot.drawdown) < self.max_drawdown:
            return RiskDecision.allow(0)
        return RiskDecision.halt(
            self.name,
            f"组合回撤 {snapshot.drawdown:.2%} 触及上限 {self.max_drawdown:.2%}：只平不开",
        )


@dataclass(frozen=True, slots=True)
class DailyLossRule(_Base):
    """当日亏损触线 → HALT。``day_pnl_ratio`` 以负数表示亏损。"""

    max_daily_loss: float = 0.03
    name: str = "daily_loss"
    priority: int = 1

    def check_portfolio(self, snapshot: PortfolioSnapshot) -> RiskDecision:
        if snapshot.day_pnl_ratio > -self.max_daily_loss:
            return RiskDecision.allow(0)
        return RiskDecision.halt(
            self.name,
            f"当日亏损 {snapshot.day_pnl_ratio:.2%} 触及上限 {self.max_daily_loss:.2%}：只平不开",
        )


@dataclass(frozen=True, slots=True)
class SevereDrawdownRule(_Base):
    """深度回撤 → FLATTEN（清仓），比 HALT 更强的一档。"""

    liquidation_drawdown: float = 0.2
    name: str = "severe_drawdown"
    priority: int = -1

    def check_portfolio(self, snapshot: PortfolioSnapshot) -> RiskDecision:
        if abs(snapshot.drawdown) < self.liquidation_drawdown:
            return RiskDecision.allow(0)
        return RiskDecision.flatten(
            self.name,
            f"组合回撤 {snapshot.drawdown:.2%} 触及清仓线 {self.liquidation_drawdown:.2%}",
        )


def default_pre_trade_rules(
    *,
    switch: KillSwitch | None = None,
    max_weight: float = 0.2,
    max_order_value: Decimal | str = "200000",
    max_orders_per_day: int = 100,
) -> tuple[PreTradeRiskRule, ...]:
    """默认事前规则集（按 priority 生效顺序排列）。"""
    return (
        KillSwitchRule(switch=switch),
        RiskLevelRule(),
        PositionConcentrationRule(max_weight=max_weight),
        OrderValueRule(max_value=to_decimal(max_order_value)),
        DailyOrderCountRule(max_orders=max_orders_per_day),
        PriceLimitRule(),
        CashRule(),
        AvailablePositionRule(),
    )


def default_intraday_rules(
    *,
    max_drawdown: float = 0.1,
    max_daily_loss: float = 0.03,
    liquidation_drawdown: float = 0.2,
) -> tuple[IntradayRiskRule, ...]:
    return (
        SevereDrawdownRule(liquidation_drawdown=liquidation_drawdown),
        DrawdownRule(max_drawdown=max_drawdown),
        DailyLossRule(max_daily_loss=max_daily_loss),
    )


__all__ = [
    "LOT_SIZE",
    "AvailablePositionRule",
    "CashRule",
    "DailyLossRule",
    "DailyOrderCountRule",
    "DrawdownRule",
    "KillSwitchRule",
    "OrderValueRule",
    "PositionConcentrationRule",
    "PriceLimitRule",
    "RiskLevelRule",
    "SevereDrawdownRule",
    "default_intraday_rules",
    "default_pre_trade_rules",
]
