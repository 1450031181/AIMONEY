"""T-03 领域模型测试。

重点覆盖两类"钱相关"的不变量：
1. 订单状态机不允许非法迁移（防重复委托、防错记状态）；
2. A 股 T+1 规则写在 Position 里，策略无法绕过。
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from decimal import Decimal

import pytest

from engine.errors import InsufficientFunds, InsufficientPosition, StateTransitionError, ValidationError
from engine.types import (
    Account,
    Bar,
    Direction,
    Order,
    OrderPlan,
    OrderStatus,
    OrderType,
    Position,
    RiskAction,
    RiskDecision,
    Side,
    Signal,
    Trade,
)

ZERO = Decimal("0")
D10 = Decimal("10.00")


# ============================================================ 订单状态机
def test_订单状态机_合法迁移_部分成交到全部成交() -> None:
    order = Order(code="600000", side=Side.BUY, qty=1000, price=D10)
    assert order.status is OrderStatus.PENDING

    order.mark_submitted("B-001")
    assert order.status is OrderStatus.SUBMITTED
    assert order.broker_order_id == "B-001"

    order.apply_fill(400, "10.00")
    assert order.status is OrderStatus.PARTIAL
    assert order.filled_qty == 400
    assert order.remaining_qty == 600

    order.apply_fill(600, "10.20")
    assert order.status is OrderStatus.FILLED
    assert order.filled_qty == 1000
    assert order.remaining_qty == 0
    # 加权均价 = (400*10.00 + 600*10.20) / 1000 = 10.12
    assert order.avg_price == Decimal("10.1200")


def test_订单状态机_终态不可再迁移() -> None:
    order = Order(code="600000", side=Side.BUY, qty=100, price=D10)
    order.mark_submitted()
    order.apply_fill(100, "10.00")
    assert order.status is OrderStatus.FILLED
    assert order.status.is_final

    with pytest.raises(StateTransitionError):
        order.cancel("撤单")


def test_订单状态机_未报出不能直接成交() -> None:
    order = Order(code="600000", side=Side.BUY, qty=100, price=D10)
    with pytest.raises(StateTransitionError):
        order.apply_fill(100, "10.00")


def test_订单状态机_unknown必须能回到终态() -> None:
    order = Order(code="600000", side=Side.BUY, qty=100, price=D10)
    order.mark_submitted("B-002")
    order.mark_unknown("报单后网络超时")
    assert order.status is OrderStatus.UNKNOWN
    # UNKNOWN 查单校准后确认已撤
    order.cancel("查单确认已撤")
    assert order.status is OrderStatus.CANCELED


def test_订单状态机_拒单必须提供原因() -> None:
    order = Order(code="600000", side=Side.BUY, qty=100, price=D10)
    with pytest.raises(ValidationError):
        order.transition_to(OrderStatus.REJECTED)
    order.reject("可用资金不足")
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason == "可用资金不足"


def test_订单_成交回填超量被拒() -> None:
    order = Order(code="600000", side=Side.BUY, qty=100, price=D10)
    order.mark_submitted()
    with pytest.raises(ValidationError):
        order.apply_fill(200, "10.00")


def test_订单_限价单必须带价格() -> None:
    with pytest.raises(ValidationError):
        Order(code="600000", side=Side.BUY, qty=100, order_type=OrderType.LIMIT)


def test_订单_幂等键稳定且区分序号() -> None:
    key1 = Order.build_idem_key("plan-001", 0)
    key2 = Order.build_idem_key("plan-001", 0)
    key3 = Order.build_idem_key("plan-001", 1)
    assert key1 == key2
    assert key1 != key3
    assert key1.startswith("idem-")


# ============================================================ 行情
def test_行情_价格自动转Decimal且精度无损() -> None:
    bar = Bar(code="600000", dt=datetime(2026, 10, 14, 15, 0), open=10.1, high=10.5, low=10.0, close=10.3, volume=1000)
    assert bar.open == Decimal("10.1")
    assert isinstance(bar.close, Decimal)


def test_行情_非法数据被拒() -> None:
    with pytest.raises(ValidationError):
        Bar(code="600000", dt=datetime(2026, 10, 14), open=10, high=9, low=11, close=10)


def test_行情_涨停封板判定() -> None:
    sealed = Bar(
        code="600000",
        dt=datetime(2026, 10, 14),
        open=11,
        high=11,
        low=11,
        close=11,
        volume=100,
        limit_up=Decimal("11.00"),
        limit_down=Decimal("9.00"),
    )
    assert sealed.sealed_limit_up
    assert sealed.is_tradable

    opened = dataclasses.replace(sealed, low=Decimal("10.50"))
    assert not opened.sealed_limit_up


def test_行情_停牌且零成交不可交易() -> None:
    bar = Bar(code="600000", dt=datetime(2026, 10, 14), is_suspended=True, volume=0)
    assert not bar.is_tradable


def test_行情_不可变() -> None:
    bar = Bar(code="600000", dt=datetime(2026, 10, 14), open=10, high=10, low=10, close=10, volume=100)
    with pytest.raises(dataclasses.FrozenInstanceError):
        bar.close = Decimal("11")  # type: ignore[misc]


# ============================================================ 信号
def test_信号_方向与权重必须一致() -> None:
    with pytest.raises(ValidationError):
        Signal(code="600000", strategy_id="s1", trade_date=date(2026, 10, 14), direction=Direction.BUY, target_weight=-0.1)
    ok = Signal(code="600000", strategy_id="s1", trade_date=date(2026, 10, 14), direction=Direction.SELL, target_weight=-0.2)
    assert ok.target_weight == -0.2


def test_信号_权重越界被拒() -> None:
    with pytest.raises(ValidationError):
        Signal(code="600000", strategy_id="s1", trade_date=date(2026, 10, 14), direction=Direction.BUY, target_weight=1.5)


# ============================================================ 订单计划
def test_订单计划_数量必须为正() -> None:
    with pytest.raises(ValidationError):
        OrderPlan(code="600000", strategy_id="s1", trade_date=date(2026, 10, 14), side=Side.BUY, target_qty=0)


# ============================================================ 持仓（T+1）
def test_持仓_当日买入不可卖() -> None:
    pos = Position(code="600000")
    pos.apply_buy(1000, "10.00")
    assert pos.qty == 1000
    assert pos.avail_qty == 0, "T+1：买入当日可卖数量必须为 0"
    assert pos.avg_cost == Decimal("10.0000")

    with pytest.raises(InsufficientPosition):
        pos.apply_sell(1000, "10.50")


def test_持仓_日切后可卖() -> None:
    pos = Position(code="600000")
    pos.apply_buy(1000, "10.00")
    pos.roll_to_next_day()
    assert pos.avail_qty == 1000
    pos.apply_sell(600, "10.50")
    assert pos.qty == 400
    assert pos.avail_qty == 400


def test_持仓_加权成本与清仓归零() -> None:
    pos = Position(code="600000")
    pos.apply_buy(1000, "10.00")
    pos.apply_buy(1000, "12.00")
    assert pos.avg_cost == Decimal("11.0000")
    pos.roll_to_next_day()
    pos.apply_sell(2000, "13.00")
    assert pos.qty == 0
    assert pos.avg_cost == ZERO
    assert pos.unrealized_pnl == ZERO


def test_持仓_冻结与解冻() -> None:
    pos = Position(code="600000")
    pos.apply_buy(1000, "10.00")
    pos.roll_to_next_day()
    pos.freeze(300)
    assert (pos.avail_qty, pos.frozen_qty) == (700, 300)
    pos.unfreeze(300)
    assert (pos.avail_qty, pos.frozen_qty) == (1000, 0)


def test_持仓_账本不平被拒() -> None:
    with pytest.raises(ValidationError):
        Position(code="600000", qty=100, avail_qty=100, frozen_qty=100)


def test_持仓_市值与浮盈() -> None:
    pos = Position(code="600000", qty=1000, avail_qty=1000, avg_cost=Decimal("10.00"), last_price=Decimal("11.00"))
    assert pos.market_value == Decimal("11000.00")
    assert pos.unrealized_pnl == Decimal("1000.00")
    assert pos.unrealized_pnl_ratio == pytest.approx(0.1)


# ============================================================ 账户
def test_账户_可用资金与冻结() -> None:
    account = Account(initial_cash=Decimal("100000"), cash=Decimal("100000"))
    account.freeze_cash("30000")
    assert account.frozen_cash == Decimal("30000")
    assert account.available_cash == Decimal("70000")
    with pytest.raises(InsufficientFunds):
        account.freeze_cash("80000")
    account.unfreeze_cash("30000")
    assert account.available_cash == Decimal("100000")


def test_账户_总资产与净值() -> None:
    account = Account(initial_cash=Decimal("100000"), cash=Decimal("60000"))
    positions = [
        Position(code="600000", qty=1000, avail_qty=1000, avg_cost=Decimal("10"), last_price=Decimal("11")),
        Position(code="000001", qty=2000, avail_qty=2000, avg_cost=Decimal("5"), last_price=Decimal("6")),
    ]
    account.mark_to_market(positions)
    assert account.market_value == Decimal("23000.00")
    assert account.total_asset == Decimal("83000.00")
    assert account.refresh_nav() == pytest.approx(0.83)
    assert account.total_pnl == Decimal("-17000.00")


def test_账户_结算后现金不得为负() -> None:
    account = Account(initial_cash=Decimal("1000"), cash=Decimal("1000"))
    with pytest.raises(InsufficientFunds):
        account.settle_cash(Decimal("-2000"))


# ============================================================ 成交
def test_成交_现金流方向与费用() -> None:
    buy = Trade(
        code="600000",
        side=Side.BUY,
        qty=100,
        price=Decimal("10.00"),
        commission=Decimal("5.00"),
        transfer_fee=Decimal("0.01"),
    )
    assert buy.amount == Decimal("1000.00")
    assert buy.total_fee == Decimal("5.01")
    assert buy.cash_flow == Decimal("-1005.01")

    sell = Trade(
        code="600000",
        side=Side.SELL,
        qty=100,
        price=Decimal("10.00"),
        commission=Decimal("5.00"),
        stamp_tax=Decimal("1.00"),
    )
    assert sell.cash_flow == Decimal("994.00")


# ============================================================ 风控决策
def test_风控决策_缩量放行() -> None:
    decision = RiskDecision.shrink("max_weight_per_stock", "单票集中度超限", requested_qty=1000, adjusted_qty=600)
    assert decision.passed
    assert decision.qty_changed
    assert decision.action is RiskAction.SHRINK


def test_风控决策_拒单不放行() -> None:
    decision = RiskDecision.reject("blacklist", "ST 股禁止买入", requested_qty=100)
    assert not decision.passed
    assert decision.adjusted_qty == 0


def test_风控决策_缩量不得超过原量() -> None:
    with pytest.raises(ValidationError):
        RiskDecision.shrink("rule", "msg", requested_qty=100, adjusted_qty=200)


def test_风控决策_熔断动作() -> None:
    assert RiskDecision.halt("daily_loss_halt", "单日亏损触线").action is RiskAction.HALT
    assert RiskDecision.flatten("max_drawdown", "回撤熔断").action is RiskAction.FLATTEN
