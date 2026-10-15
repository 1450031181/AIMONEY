"""AIMoney 领域模型（纯数据 + 账本行为，零 IO、零外部依赖）。

设计约束：
1. 价格/金额统一 ``Decimal``，数量统一 ``int``（见 ``engine.num``）；
2. 事实型数据不可变（``Bar`` / ``Signal`` / ``Trade`` / ``OrderPlan``）：回测中一旦记录就不可篡改，
   便于复现与审计；
3. 账本型对象可变（``Order`` / ``Position`` / ``Account``）：状态迁移必须走 ``transition_to()``，
   非法迁移直接抛 ``StateTransitionError``——**这是"不产生重复委托、不错记持仓"的第一道防线**；
4. A 股 T+1 规则直接编码进 ``Position.apply_buy()``：买入不增加 ``avail_qty``，
   只有日切 ``roll_to_next_day()`` 后才可卖。规则写在模型里，就不会被策略绕过。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

from engine.errors import InsufficientFunds, InsufficientPosition, StateTransitionError, ValidationError
from engine.ids import new_id
from engine.num import round_cost, round_price, safe_div, to_decimal

_ZERO = Decimal(0)


# ============================================================ 枚举
class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """买入 +1 / 卖出 -1，用于权重与持仓变动计算。"""
        return 1 if self is Side.BUY else -1


class Direction(StrEnum):
    """信号方向。HOLD 表示维持现状（可用于显式表达"持仓不动"）。"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(StrEnum):
    PENDING = "PENDING"  # 本地已创建，未报出
    SUBMITTED = "SUBMITTED"  # 已报，未成交
    PARTIAL = "PARTIAL"  # 部分成交
    FILLED = "FILLED"  # 全部成交
    CANCELED = "CANCELED"  # 已撤（含部分成交后撤余量）
    REJECTED = "REJECTED"  # 被拒（风控/资金/交易所）
    UNKNOWN = "UNKNOWN"  # 报单后网络异常，状态待查（绝不可当作未报出处理）

    @property
    def is_final(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)

    @property
    def is_open(self) -> bool:
        """是否仍可能产生成交（用于盘中轮询撤单/校准）。"""
        return self in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL, OrderStatus.UNKNOWN)


class PlanStatus(StrEnum):
    NEW = "NEW"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class RiskAction(StrEnum):
    """风控处置动作，严重度递增。"""

    PASS = "PASS"
    SHRINK = "SHRINK"  # 缩量放行
    REJECT = "REJECT"  # 拒单
    HALT = "HALT"  # 只平不开（停止开新仓）
    FLATTEN = "FLATTEN"  # 强制清仓


class RiskLevel(StrEnum):
    """风控状态机（单向降级，人工确认后才可恢复 NORMAL）。"""

    NORMAL = "NORMAL"
    WARN = "WARN"
    HALT = "HALT"
    FLATTEN = "FLATTEN"


# 订单状态机：唯一合法的迁移路径
_ORDER_TRANSITIONS: Final[dict[OrderStatus, frozenset[OrderStatus]]] = {
    OrderStatus.PENDING: frozenset({OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.CANCELED}),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.UNKNOWN,
        }
    ),
    OrderStatus.PARTIAL: frozenset(
        {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.UNKNOWN}
    ),
    # UNKNOWN 必须能回到任何终态：查单校准后才知道真实状态
    OrderStatus.UNKNOWN: frozenset(
        {
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}

_DECIMAL_FIELDS_CACHE: dict[type, tuple[str, ...]] = {}


def _decimal_fields(cls: type) -> tuple[str, ...]:
    cached = _DECIMAL_FIELDS_CACHE.get(cls)
    if cached is None:
        # 注意 1：开启 `from __future__ import annotations` 后 f.type 是字符串，
        #         故用子串匹配以同时覆盖 "Decimal" 与 "Decimal | None"。
        # 注意 2：这里用 getattr 而非 cls.__dataclass_fields__，因为 cls 是动态类型（mypy attr-defined）。
        cached = tuple(
            name
            for name, f in getattr(cls, "__dataclass_fields__").items()  # noqa: B009
            if "Decimal" in str(f.type)
        )
        _DECIMAL_FIELDS_CACHE[cls] = cached
    return cached


def _coerce_decimals(obj: Any) -> None:
    """把构造时传入的 int/float/str 自动转成 Decimal（frozen 数据类用 object.__setattr__）。"""
    for name in _decimal_fields(type(obj)):
        raw = getattr(obj, name)
        if raw is not None and not isinstance(raw, Decimal):
            object.__setattr__(obj, name, to_decimal(raw))


# ============================================================ 行情
@dataclass(frozen=True, slots=True)
class Bar:
    """单标的单周期行情快照。

    ``adj_factor`` / ``limit_up`` / ``limit_down`` 允许缺失（None 表示"未知"而非"无限制"），
    下游遇 None 必须显式处理，禁止默默当作可成交。
    """

    code: str
    dt: datetime
    open: Decimal = _ZERO
    high: Decimal = _ZERO
    low: Decimal = _ZERO
    close: Decimal = _ZERO
    volume: int = 0
    amount: Decimal = _ZERO
    prev_close: Decimal | None = None
    limit_up: Decimal | None = None
    limit_down: Decimal | None = None
    is_suspended: bool = False
    adj_factor: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        _coerce_decimals(self)
        if not self.code:
            raise ValidationError("Bar.code 不能为空")
        if self.low < 0 or self.high < 0 or self.open < 0 or self.close < 0:
            raise ValidationError("Bar 价格不能为负", detail={"code": self.code, "dt": str(self.dt)})
        if self.high < self.low:
            raise ValidationError(
                "Bar 数据非法：high 小于 low", detail={"code": self.code, "dt": str(self.dt), "high": str(self.high), "low": str(self.low)}
            )
        if self.volume < 0:
            raise ValidationError("Bar.volume 不能为负", detail={"code": self.code, "dt": str(self.dt)})

    @property
    def is_tradable(self) -> bool:
        """是否具备基本可交易条件（非停牌且有成交）。"""
        return not self.is_suspended and self.volume > 0

    @property
    def sealed_limit_up(self) -> bool:
        """是否一字涨停封板（最低价已到涨停价）：此状态下买单无法成交。"""
        return self.limit_up is not None and self.low >= self.limit_up

    @property
    def sealed_limit_down(self) -> bool:
        """是否一字跌停封板：此状态下卖单无法成交。"""
        return self.limit_down is not None and self.high <= self.limit_down


# ============================================================ 信号
@dataclass(frozen=True, slots=True)
class Signal:
    """策略输出信号。策略**只能**产出 Signal，不能直接下单（见 engine.ports.Context）。"""

    code: str
    strategy_id: str
    trade_date: date
    direction: Direction
    target_weight: float = 0.0
    score: float = 0.0
    reason: str = ""
    signal_id: str = field(default_factory=lambda: new_id("sig"))
    factor_snapshot: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValidationError("Signal.code 不能为空")
        if not -1.0 <= self.target_weight <= 1.0:
            raise ValidationError(
                "Signal.target_weight 必须落在 [-1, 1]",
                detail={"code": self.code, "target_weight": self.target_weight},
            )
        if self.direction is Direction.BUY and self.target_weight < 0:
            raise ValidationError("BUY 信号的 target_weight 不能为负", detail={"code": self.code})
        if self.direction is Direction.SELL and self.target_weight > 0:
            raise ValidationError("SELL 信号的 target_weight 不能为正", detail={"code": self.code})


# ============================================================ 订单计划
@dataclass(frozen=True, slots=True)
class OrderPlan:
    """从信号到订单的中间层：先算目标数量，再经风控裁剪，最后才落到 Order。

    保留这一层是为了：风控只处理 OrderPlan（不含券商语义），且执行失败可重算。
    """

    code: str
    strategy_id: str
    trade_date: date
    side: Side
    target_qty: int
    limit_price: Decimal | None = None
    urgency: int = 5
    reason: str = ""
    status: PlanStatus = PlanStatus.NEW
    plan_id: str = field(default_factory=lambda: new_id("plan"))

    def __post_init__(self) -> None:
        _coerce_decimals(self)
        if self.target_qty <= 0:
            raise ValidationError("OrderPlan.target_qty 必须为正", detail={"code": self.code, "qty": self.target_qty})
        if not 0 <= self.urgency <= 10:
            raise ValidationError("OrderPlan.urgency 必须落在 [0, 10]", detail={"urgency": self.urgency})


# ============================================================ 委托
@dataclass(slots=True)
class Order:
    """本地委托账本。状态迁移与成交回填一律通过方法进行，禁止外部直接改 ``status``。"""

    code: str
    side: Side
    qty: int
    plan_id: str = ""
    strategy_id: str = ""
    price: Decimal | None = None
    order_type: OrderType = OrderType.LIMIT
    order_id: str = field(default_factory=lambda: new_id("ord"))
    broker_order_id: str | None = None
    filled_qty: int = 0
    avg_price: Decimal = _ZERO
    status: OrderStatus = OrderStatus.PENDING
    reject_reason: str | None = None
    idem_key: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        _coerce_decimals(self)
        if self.qty <= 0:
            raise ValidationError("Order.qty 必须为正", detail={"code": self.code, "qty": self.qty})
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValidationError("限价单必须指定 price", detail={"code": self.code})
        if self.price is not None and self.price <= 0:
            raise ValidationError("Order.price 必须为正", detail={"code": self.code, "price": str(self.price)})
        if not 0 <= self.filled_qty <= self.qty:
            raise ValidationError("Order.filled_qty 越界", detail={"filled_qty": self.filled_qty, "qty": self.qty})

    # -------------------------------------------------- 派生属性
    @property
    def remaining_qty(self) -> int:
        return self.qty - self.filled_qty

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def filled_amount(self) -> Decimal:
        return self.avg_price * self.filled_qty

    # -------------------------------------------------- 幂等键
    @staticmethod
    def build_idem_key(plan_id: str, seq: int = 0) -> str:
        """构造幂等键：同一计划同一序号只允许存在一笔委托。

        重试前先按 ``idem_key`` 查库，命中则直接返回既有委托，绝不重复报单。
        """
        digest = hashlib.sha256(f"{plan_id}|{seq}".encode()).hexdigest()
        return f"idem-{digest[:32]}"

    # -------------------------------------------------- 状态机
    def can_transition_to(self, status: OrderStatus) -> bool:
        return status is self.status or status in _ORDER_TRANSITIONS[self.status]

    def transition_to(
        self,
        status: OrderStatus,
        *,
        reason: str | None = None,
        broker_order_id: str | None = None,
    ) -> None:
        """唯一合法的状态迁移入口。同状态重复调用视为幂等（直接返回）。"""
        if status is self.status:
            return
        allowed = _ORDER_TRANSITIONS[self.status]
        if status not in allowed:
            raise StateTransitionError(
                f"非法订单状态迁移：{self.status.value} -> {status.value}",
                detail={
                    "order_id": self.order_id,
                    "current": self.status.value,
                    "target": status.value,
                    "allowed": sorted(s.value for s in allowed),
                },
            )
        if broker_order_id:
            self.broker_order_id = broker_order_id
        if reason:
            self.reject_reason = reason
        if status is OrderStatus.REJECTED and not self.reject_reason:
            raise ValidationError("拒单必须提供原因", detail={"order_id": self.order_id})
        self.status = status
        self.updated_at = datetime.now()

    # -------------------------------------------------- 便捷操作
    def mark_submitted(self, broker_order_id: str | None = None) -> None:
        self.transition_to(OrderStatus.SUBMITTED, broker_order_id=broker_order_id)

    def mark_unknown(self, reason: str) -> None:
        """报单后网络异常：标记 UNKNOWN，必须查单校准，禁止直接重报。"""
        self.transition_to(OrderStatus.UNKNOWN, reason=reason)

    def reject(self, reason: str) -> None:
        self.transition_to(OrderStatus.REJECTED, reason=reason)

    def cancel(self, reason: str | None = None) -> None:
        self.transition_to(OrderStatus.CANCELED, reason=reason)

    def apply_fill(self, fill_qty: int, fill_price: Decimal | float | str) -> None:
        """回填成交：自动计算加权均价并把状态推进到 PARTIAL / FILLED。"""
        price = to_decimal(fill_price)
        if fill_qty <= 0:
            raise ValidationError("成交数量必须为正", detail={"order_id": self.order_id, "fill_qty": fill_qty})
        if fill_qty > self.remaining_qty:
            raise ValidationError(
                "成交量超过剩余委托量",
                detail={"order_id": self.order_id, "fill_qty": fill_qty, "remaining": self.remaining_qty},
            )
        if price <= 0:
            raise ValidationError("成交价必须为正", detail={"order_id": self.order_id})
        total = self.avg_price * self.filled_qty + price * fill_qty
        self.filled_qty += fill_qty
        self.avg_price = round_cost(total / self.filled_qty)
        self.updated_at = datetime.now()
        target = OrderStatus.FILLED if self.remaining_qty == 0 else OrderStatus.PARTIAL
        self.transition_to(target)


# ============================================================ 成交
@dataclass(frozen=True, slots=True)
class Trade:
    """成交回报（不可变）。费用拆项存储，便于后续归因与对账。"""

    code: str
    side: Side
    qty: int
    price: Decimal
    order_id: str = ""
    strategy_id: str = ""
    commission: Decimal = _ZERO
    stamp_tax: Decimal = _ZERO
    transfer_fee: Decimal = _ZERO
    slippage_cost: Decimal = _ZERO
    trade_id: str = field(default_factory=lambda: new_id("trd"))
    trade_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        _coerce_decimals(self)
        if self.qty <= 0:
            raise ValidationError("Trade.qty 必须为正", detail={"code": self.code, "qty": self.qty})
        if self.price <= 0:
            raise ValidationError("Trade.price 必须为正", detail={"code": self.code})

    @property
    def amount(self) -> Decimal:
        """成交金额（不含费用）。"""
        return self.price * self.qty

    @property
    def total_fee(self) -> Decimal:
        return self.commission + self.stamp_tax + self.transfer_fee

    @property
    def cash_flow(self) -> Decimal:
        """现金净变动：买入为负（含费用），卖出为正（已扣费用）。"""
        if self.side is Side.BUY:
            return -(self.amount + self.total_fee)
        return self.amount - self.total_fee


# ============================================================ 持仓
@dataclass(slots=True)
class Position:
    """持仓账本。T+1 规则内建于 ``apply_buy``：当日买入不计入可卖数量。"""

    code: str
    account_id: str = "default"
    qty: int = 0  # 总持仓
    avail_qty: int = 0  # 可卖数量（T+1 约束体现在这里）
    frozen_qty: int = 0  # 挂单冻结（卖出报单后从 avail 扣减）
    avg_cost: Decimal = _ZERO
    last_price: Decimal = _ZERO

    def __post_init__(self) -> None:
        _coerce_decimals(self)
        if self.qty < 0 or self.avail_qty < 0 or self.frozen_qty < 0:
            raise ValidationError("持仓数量不能为负", detail={"code": self.code})
        if self.avail_qty + self.frozen_qty > self.qty:
            raise ValidationError(
                "持仓账本不平：avail + frozen > qty",
                detail={"code": self.code, "qty": self.qty, "avail": self.avail_qty, "frozen": self.frozen_qty},
            )

    @property
    def market_value(self) -> Decimal:
        return round_price(self.last_price * self.qty)

    @property
    def cost_value(self) -> Decimal:
        return self.avg_cost * self.qty

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.qty == 0:
            return _ZERO
        return round_price((self.last_price - self.avg_cost) * self.qty)

    @property
    def unrealized_pnl_ratio(self) -> float:
        return safe_div(float(self.unrealized_pnl), float(self.cost_value))

    @property
    def is_empty(self) -> bool:
        return self.qty == 0

    # -------------------------------------------------- 账本操作
    def apply_buy(self, qty: int, price: Decimal | float | str) -> None:
        """买入成交：加权更新成本价；**不增加 avail_qty**（T+1 当日不可卖）。"""
        if qty <= 0:
            raise ValidationError("买入数量必须为正", detail={"code": self.code, "qty": qty})
        cost = to_decimal(price)
        self.avg_cost = round_cost((self.avg_cost * self.qty + cost * qty) / (self.qty + qty))
        self.qty += qty
        self.last_price = cost

    def apply_sell(self, qty: int, price: Decimal | float | str) -> None:
        """卖出成交：校验可卖数量；清仓时成本价归零。"""
        if qty <= 0:
            raise ValidationError("卖出数量必须为正", detail={"code": self.code, "qty": qty})
        if qty > self.avail_qty:
            raise InsufficientPosition(
                "可卖数量不足（注意 T+1：当日买入不可卖）",
                detail={"code": self.code, "requested": qty, "avail": self.avail_qty, "qty": self.qty},
            )
        self.qty -= qty
        self.avail_qty -= qty
        self.last_price = to_decimal(price)
        if self.qty == 0:
            self.avg_cost = _ZERO
            self.avail_qty = 0
            self.frozen_qty = 0

    def freeze(self, qty: int) -> None:
        """卖出报单成功：可卖数量转入冻结。"""
        if qty > self.avail_qty:
            raise InsufficientPosition(
                "冻结数量超过可卖数量", detail={"code": self.code, "requested": qty, "avail": self.avail_qty}
            )
        self.avail_qty -= qty
        self.frozen_qty += qty

    def unfreeze(self, qty: int) -> None:
        """撤单/废单：冻结数量退回可卖。"""
        if qty > self.frozen_qty:
            raise ValidationError(
                "解冻数量超过冻结数量", detail={"code": self.code, "requested": qty, "frozen": self.frozen_qty}
            )
        self.frozen_qty -= qty
        self.avail_qty += qty

    def roll_to_next_day(self) -> None:
        """日切：昨日买入的持仓转为可卖（T+1 解除）。"""
        self.avail_qty = self.qty - self.frozen_qty


# ============================================================ 账户
@dataclass(slots=True)
class Account:
    """资金账户。``total_asset`` 为派生值，避免与 cash/market_value 出现三处不一致。"""

    account_id: str = "default"
    initial_cash: Decimal = _ZERO
    cash: Decimal = _ZERO  # 总现金（含冻结）
    market_value: Decimal = _ZERO
    frozen_cash: Decimal = _ZERO
    nav: float = 1.0
    trade_date: date | None = None

    def __post_init__(self) -> None:
        _coerce_decimals(self)
        if self.cash < 0:
            raise ValidationError("现金不能为负", detail={"cash": str(self.cash)})
        if self.frozen_cash < 0 or self.frozen_cash > self.cash:
            raise ValidationError(
                "冻结资金非法", detail={"cash": str(self.cash), "frozen_cash": str(self.frozen_cash)}
            )

    @property
    def total_asset(self) -> Decimal:
        return self.cash + self.market_value

    @property
    def available_cash(self) -> Decimal:
        return self.cash - self.frozen_cash

    @property
    def total_pnl(self) -> Decimal:
        return self.total_asset - self.initial_cash

    @property
    def total_pnl_ratio(self) -> float:
        return safe_div(float(self.total_pnl), float(self.initial_cash))

    def freeze_cash(self, amount: Decimal | float | str) -> None:
        """买入报单：冻结资金（含预估费用）。"""
        value = to_decimal(amount)
        if value < 0:
            raise ValidationError("冻结金额不能为负")
        if value > self.available_cash:
            raise InsufficientFunds(
                "可用资金不足",
                detail={
                    "account_id": self.account_id,
                    "required": str(value),
                    "available": str(self.available_cash),
                },
            )
        self.frozen_cash += value

    def unfreeze_cash(self, amount: Decimal | float | str) -> None:
        value = to_decimal(amount)
        if value > self.frozen_cash:
            raise ValidationError(
                "解冻金额超过冻结金额", detail={"requested": str(value), "frozen": str(self.frozen_cash)}
            )
        self.frozen_cash -= value

    def settle_cash(self, delta: Decimal) -> None:
        """结算现金净变动（delta 为负数即扣款）。"""
        new_cash = self.cash + delta
        if new_cash < 0:
            raise InsufficientFunds("结算后现金为负", detail={"delta": str(delta), "cash": str(self.cash)})
        self.cash = new_cash

    def mark_to_market(self, positions: Iterable[Position]) -> None:
        self.market_value = sum((p.market_value for p in positions), start=_ZERO)

    def refresh_nav(self) -> float:
        if self.initial_cash > 0:
            self.nav = float(self.total_asset / self.initial_cash)
        return self.nav


# ============================================================ 风控决策
@dataclass(frozen=True, slots=True)
class RiskDecision:
    """风控裁定结果。``adjusted_qty`` 才是最终允许下发的数量。"""

    action: RiskAction = RiskAction.PASS
    rule: str | None = None
    message: str = ""
    requested_qty: int = 0
    adjusted_qty: int = 0

    @property
    def passed(self) -> bool:
        return self.action in (RiskAction.PASS, RiskAction.SHRINK) and self.adjusted_qty > 0

    @property
    def qty_changed(self) -> bool:
        return self.adjusted_qty != self.requested_qty

    @classmethod
    def allow(cls, qty: int, *, rule: str | None = None) -> RiskDecision:
        return cls(action=RiskAction.PASS, rule=rule, requested_qty=qty, adjusted_qty=qty)

    @classmethod
    def shrink(cls, rule: str, message: str, *, requested_qty: int, adjusted_qty: int) -> RiskDecision:
        if adjusted_qty > requested_qty:
            raise ValidationError("缩量后的数量不能大于原始数量")
        return cls(
            action=RiskAction.SHRINK,
            rule=rule,
            message=message,
            requested_qty=requested_qty,
            adjusted_qty=max(adjusted_qty, 0),
        )

    @classmethod
    def reject(cls, rule: str, message: str, *, requested_qty: int = 0) -> RiskDecision:
        return cls(
            action=RiskAction.REJECT,
            rule=rule,
            message=message,
            requested_qty=requested_qty,
            adjusted_qty=0,
        )

    @classmethod
    def halt(cls, rule: str, message: str) -> RiskDecision:
        return cls(action=RiskAction.HALT, rule=rule, message=message)

    @classmethod
    def flatten(cls, rule: str, message: str) -> RiskDecision:
        return cls(action=RiskAction.FLATTEN, rule=rule, message=message)


__all__ = [
    "Account",
    "Bar",
    "Direction",
    "Order",
    "OrderPlan",
    "OrderStatus",
    "OrderType",
    "PlanStatus",
    "Position",
    "RiskAction",
    "RiskDecision",
    "RiskLevel",
    "Side",
    "Signal",
    "Trade",
]
