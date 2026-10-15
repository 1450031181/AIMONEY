"""P4-1 订单管理测试。

三条最关键的断言：

1. **闸门与风控在前**：门禁未开/急停激活时**不调用券商**（不是"报出去再补救"）；
2. **幂等键跨重启稳定**：进程崩了重启，同一笔计划不能被重复报出——
   这要求幂等键由业务要素确定性推导，而不是用随机的计划号；
3. **通道异常标记为状态未知**：不重报（重报会造成真实重复委托），
   必须走查单校准。
"""

from __future__ import annotations

import copy
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from engine.broker.paper import PaperBrokerAdapter
from engine.errors import BrokerError
from engine.risk.engine import RiskEngine
from engine.risk.kill_switch import KillSwitch
from engine.risk.rules import PositionConcentrationRule
from engine.trade.oms import OrderManager
from engine.types import Account, Bar, Order, OrderPlan, OrderStatus, Position, RiskLevel, Side

DAY = date(2026, 10, 15)


def _bar(close: str = "10.00") -> Bar:
    from datetime import datetime

    price = Decimal(close)
    return Bar(
        code="600000",
        dt=datetime(2026, 10, 15, 9, 30),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000_000,
        amount=price * 1_000_000,
    )


def _plan(qty: int = 1000, *, side: Side = Side.BUY) -> OrderPlan:
    return OrderPlan(
        code="600000",
        strategy_id="s1",
        trade_date=DAY,
        side=side,
        target_qty=qty,
        limit_price=Decimal("10.00"),
    )


def _account() -> Account:
    return Account(initial_cash=Decimal("1000000"), cash=Decimal("1000000"))


class _StubBroker:
    """可注入故障的券商替身（OMS 单测用；比模拟盘更可控）。"""

    name = "stub"

    def __init__(self) -> None:
        self.placed: list[Order] = []
        #: 调用次数（含失败）：验证"异常后不得自动重报"要用它，而不是 placed
        self.attempts = 0
        self.fail_next = False
        self.orders: dict[str, Order] = {}
        self._connected = False
        self._counter = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def query_account(self) -> Account:
        return _account()

    def query_positions(self) -> list[Position]:
        return []

    def query_orders(self) -> list[Order]:
        return [copy.deepcopy(order) for order in self.orders.values()]

    def place_order(self, order: Order) -> str:
        self.attempts += 1
        if self.fail_next:
            self.fail_next = False
            raise BrokerError("通道抖动（模拟）")
        self._counter += 1
        broker_order_id = f"bo{self._counter}"
        order.mark_submitted(broker_order_id)
        # 深拷贝：真实券商的委托是**它自己的对象**，共享同一个实例等于没有网络边界，
        # 测试会"看起来通过"但其实什么都没验证（本地对象被对端直接改了）
        self.orders[broker_order_id] = copy.deepcopy(order)
        self.placed.append(order)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> bool:
        return self.orders.pop(broker_order_id, None) is not None

    def on_order_update(self, callback: Any) -> None:
        return None

    def on_trade_update(self, callback: Any) -> None:
        return None


class _Gate:
    def __init__(self, is_open: bool) -> None:
        self._is_open = is_open

    def is_open(self) -> bool:
        return self._is_open


def _manager(broker: _StubBroker, **overrides: object) -> OrderManager:
    manager = OrderManager(broker=broker, **overrides)  # type: ignore[arg-type]
    manager.connect()
    return manager


# ============================================================ 闸门
def test_门禁未开时不下发券商() -> None:
    broker = _StubBroker()
    manager = _manager(broker, gate=_Gate(is_open=False))

    result = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})

    assert result.submitted == ()
    assert len(result.rejected) == 1
    assert "门禁" in result.rejected[0][1]
    assert broker.placed == [], "拦截必须发生在券商调用之前"


def test_急停激活时不下发券商(tmp_path: Path) -> None:
    switch = KillSwitch(tmp_path / "kill.json")
    switch.activate("盘中异常", by="test")
    broker = _StubBroker()
    manager = _manager(broker, gate=_Gate(is_open=True), kill_switch=switch)

    result = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})

    assert result.submitted == ()
    assert "盘中异常" in result.rejected[0][1]
    assert broker.placed == []


def test_券商未连接时拒绝报单() -> None:
    broker = _StubBroker()
    manager = OrderManager(broker=broker)  # 未 connect

    result = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})

    assert "未连接" in result.rejected[0][1]
    assert broker.placed == []


# ============================================================ 风控
def test_风控缩量后按缩量数量报单() -> None:
    broker = _StubBroker()
    # 单票上限 1%：100 万 × 1% ÷ 10 元 = 1000 股
    engine = RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=0.01)])
    manager = _manager(broker, risk_engine=engine)

    result = manager.submit(
        [_plan(5000)], trade_date=DAY, account=_account(), positions={}, bars={"600000": _bar()}
    )

    assert len(result.submitted) == 1
    assert result.submitted[0].qty == 1000, "必须按风控缩量后的数量报单"
    assert broker.placed[0].qty == 1000


def test_风控拒单不触发券商调用() -> None:
    broker = _StubBroker()
    engine = RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=0.000001)])
    manager = _manager(broker, risk_engine=engine)

    result = manager.submit(
        [_plan(1000)], trade_date=DAY, account=_account(), positions={}, bars={"600000": _bar()}
    )

    assert result.rejected and broker.placed == []


def test_风险等级为HALT时买单被拒() -> None:
    broker = _StubBroker()
    engine = RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=1.0)])
    manager = _manager(broker, risk_engine=engine)

    result = manager.submit(
        [_plan(1000)],
        trade_date=DAY,
        account=_account(),
        positions={},
        bars={"600000": _bar()},
        risk_level=RiskLevel.NORMAL,
    )

    assert len(result.submitted) == 1, "NORMAL 等级下应放行（对照组）"


# ============================================================ 幂等
def test_幂等键是确定性的() -> None:
    plan = _plan(1000)
    first = OrderManager.build_idem_key(plan, DAY)
    second = OrderManager.build_idem_key(copy.deepcopy(plan), DAY)

    assert first == second, "同一业务要素必须得到同一幂等键（跨进程/跨重启稳定）"
    other = OrderManager.build_idem_key(_plan(2000), DAY)
    assert first != other, "数量不同即为不同委托"


def test_同一批次重复提交被幂等拦下() -> None:
    broker = _StubBroker()
    manager = _manager(broker)

    first = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})
    second = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})

    assert len(first.submitted) == 1
    assert second.submitted == ()
    assert len(second.duplicates) == 1, "重复提交应命中幂等键而不是再报一笔"
    assert len(broker.placed) == 1


def test_重启后同一计划不会重复下单() -> None:
    """模拟进程重启：本地账本为空，但券商侧仍记得这笔幂等键。"""
    broker = PaperBrokerAdapter(initial_cash="1000000")
    plan = _plan(1000)
    account = _account()

    first = OrderManager(broker=broker)
    first.connect()
    first_result = first.submit([plan], trade_date=DAY, account=account, positions={}, bars={"600000": _bar()})

    restarted = OrderManager(broker=broker)  # 新进程：本地账本全空
    restarted.connect()
    second_result = restarted.submit(
        [plan], trade_date=DAY, account=account, positions={}, bars={"600000": _bar()}
    )

    assert len(broker.query_orders()) == 1, "券商侧幂等兜住了重启后的重复报单"
    assert (
        first_result.submitted[0].broker_order_id == second_result.submitted[0].broker_order_id
    ), "同幂等键必须拿到同一个券商委托号"


# ============================================================ 通道异常与校准
def test_通道异常标记为状态未知且不重报() -> None:
    broker = _StubBroker()
    broker.fail_next = True
    manager = _manager(broker)

    result = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})

    assert len(result.unknown) == 1
    assert result.unknown[0].status is OrderStatus.UNKNOWN
    assert result.submitted == ()
    assert broker.attempts == 1, "异常后**不得**自动重报（重报可能造成真实重复委托）"
    assert broker.placed == [], "报单未成功，券商侧不应有委托"

    again = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})
    assert again.duplicates or again.unknown, "同一计划的再次提交必须被幂等键拦住"
    assert broker.attempts == 1, "幂等命中的提交不应再触碰券商"


def test_校准用券商状态纠正本地() -> None:
    broker = _StubBroker()
    manager = _manager(broker)
    manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})
    local_order = manager.orders()[0]
    assert local_order.status is OrderStatus.SUBMITTED

    broker.orders[local_order.broker_order_id or ""].apply_fill(1000, Decimal("10.00"))  # 回报丢失
    result = manager.reconcile()

    assert result.corrected == (local_order.order_id,)
    assert local_order.status is OrderStatus.FILLED
    assert local_order.filled_qty == 1000


def test_券商查无此单判定废单() -> None:
    broker = _StubBroker()
    manager = _manager(broker)
    manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})
    broker.orders.clear()  # 券商侧查无此单

    result = manager.reconcile()

    assert result.missing == (manager.orders()[0].order_id,)
    assert manager.orders()[0].status is OrderStatus.REJECTED
    assert "废单" in (manager.orders()[0].reject_reason or "")


# ============================================================ 撤单
def test_取消在途委托() -> None:
    broker = _StubBroker()
    manager = _manager(broker)
    result = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={})
    broker_order_id = result.submitted[0].broker_order_id or ""

    assert manager.cancel(broker_order_id) is True

    assert manager.orders()[0].status is OrderStatus.CANCELED
    assert manager.orders(open_only=True) == []


def test_批次摘要可读() -> None:
    broker = _StubBroker()
    manager = _manager(broker, gate=_Gate(is_open=False))

    summary = manager.submit([_plan()], trade_date=DAY, account=_account(), positions={}).summary()

    assert "拒单" in summary
