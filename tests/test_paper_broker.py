"""P4-1 模拟盘券商测试。

重点验证三件事（都是"错了不会报异常、只会慢慢错"的类型）：

1. **幂等报单**：断线重连后的重试不能变成两笔真实委托；
2. **冻结守恒**：撤单/废单/顺延超限都必须把冻结的资金与持仓还回去，
   否则资金与可卖数量会缓慢泄漏；
3. **查询返回副本**：内部账本不能被调用方的无心修改污染。
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from engine.backtest.cost import FeeModel
from engine.broker.paper import PaperBrokerAdapter
from engine.errors import BrokerUnavailable, OrderRejected, ValidationError
from engine.types import Bar, Order, OrderStatus, OrderType, Side

DAY = date(2026, 10, 15)


def _expected_fees(amount: str, side: Side = Side.BUY) -> Decimal:
    """按费用模型算出的费用（不硬编码，避免"费率一改测试就红"）。"""
    return FeeModel().estimate(Decimal(amount), side).total


def _bar(close: str = "10.00", *, volume: int = 1_000_000, **overrides: object) -> Bar:
    price = Decimal(close)
    values: dict[str, object] = {
        "code": "600000",
        "dt": datetime(2026, 10, 15, 9, 30),
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": volume,
        "amount": price * volume,
    }
    values.update(overrides)
    return Bar(**values)  # type: ignore[arg-type]


def _broker(**overrides: object) -> PaperBrokerAdapter:
    options: dict[str, object] = {"initial_cash": "1000000", "slippage": None, "max_defer_days": 2}
    options.update(overrides)
    broker = PaperBrokerAdapter(**options)  # type: ignore[arg-type]
    broker.connect()
    return broker


def _order(qty: int = 1000, *, side: Side = Side.BUY, price: str = "10.00", idem: str = "k1") -> Order:
    return Order(
        code="600000",
        side=side,
        qty=qty,
        price=Decimal(price),
        idem_key=idem,
        strategy_id="s1",
    )


# ============================================================ 连接
def test_未连接禁止报单() -> None:
    broker = PaperBrokerAdapter(initial_cash="1000000")

    with pytest.raises(BrokerUnavailable) as excinfo:
        broker.place_order(_order())

    assert "未连接" in str(excinfo.value)


# ============================================================ 幂等
def test_报单幂等_同一幂等键只产生一笔委托() -> None:
    broker = _broker()

    first = broker.place_order(_order(1000))
    second = broker.place_order(_order(1000))

    assert first == second, "重复报单必须返回同一个券商委托号"
    assert len(broker.query_orders()) == 1, "不得产生第二笔委托"
    account = broker.query_account()
    assert account.frozen_cash == Decimal("10000") + _expected_fees("10000"), "冻结也只能发生一次"


def test_不同幂等键是两笔委托() -> None:
    broker = _broker()

    first = broker.place_order(_order(1000, idem="k1"))
    second = broker.place_order(_order(1000, idem="k2"))

    assert first != second
    assert len(broker.query_orders()) == 2


# ============================================================ 冻结与成交
def test_买入冻结含费资金并在成交后按实际金额扣款() -> None:
    broker = _broker()
    broker.place_order(_order(1000))

    frozen = broker.query_account().frozen_cash
    assert frozen == Decimal("10000") + _expected_fees("10000"), "冻结 = 成交额 + 预估费用"

    trades = broker.push_bar(_bar())

    assert len(trades) == 1
    account = broker.query_account()
    assert account.frozen_cash == Decimal("0.00"), "成交后冻结必须清零"
    assert account.cash == Decimal("1000000") - trades[0].amount - trades[0].total_fee
    position = broker.query_positions()[0]
    assert position.qty == 1000
    assert position.avail_qty == 0, "T+1：当日买入不可卖"


def test_日切后可卖() -> None:
    broker = _broker()
    broker.place_order(_order(1000))
    broker.push_bar(_bar())

    broker.roll_to_next_day()

    assert broker.query_positions()[0].avail_qty == 1000


def test_市价单无参考价被拒且说明原因() -> None:
    broker = _broker()
    market_order = Order(
        code="600000",
        side=Side.BUY,
        qty=1000,
        strategy_id="s1",
        idem_key="m1",
        order_type=OrderType.MARKET,
    )

    with pytest.raises(OrderRejected) as excinfo:
        broker.place_order(market_order)

    assert "参考价" in str(excinfo.value)
    assert market_order.status is OrderStatus.REJECTED, "被拒的订单要落到账本上（可审计）"
    assert broker.query_account().frozen_cash == Decimal("0")


def test_推送行情后市价单可报() -> None:
    """交易日循环的时序要求：先推送当日行情（更新参考价），再报单。"""
    broker = _broker()
    broker.push_bars({"600000": _bar()})
    market_order = Order(
        code="600000",
        side=Side.BUY,
        qty=1000,
        strategy_id="s1",
        idem_key="m1",
        order_type=OrderType.MARKET,
    )

    broker_order_id = broker.place_order(market_order)

    assert broker_order_id
    assert market_order.status is OrderStatus.SUBMITTED


def test_报单失败不留下冻结() -> None:
    broker = _broker(initial_cash="5000")  # 资金不足以买 1000 股

    with pytest.raises(OrderRejected):
        broker.place_order(_order(1000))

    assert broker.query_account().frozen_cash == Decimal("0"), "失败必须不留冻结"


def test_卖出冻结可卖数量并在成交后减少持仓() -> None:
    broker = _broker()
    broker.place_order(_order(1000))
    broker.push_bar(_bar())
    broker.roll_to_next_day()
    broker.push_bars({})  # 行情推进（无委托，仅刷新参考价）

    sell = broker.place_order(_order(600, side=Side.SELL, idem="s1"))
    assert sell
    position = broker.query_positions()[0]
    assert position.avail_qty == 400 and position.frozen_qty == 600

    broker.push_bar(_bar("11.00"))

    position = broker.query_positions()[0]
    assert position.qty == 400, "卖出 600 股后剩 400 股"
    assert position.frozen_qty == 0 and position.avail_qty == 400


def test_可卖不足时卖单被拒且不泄漏冻结() -> None:
    broker = _broker()
    broker.place_order(_order(1000))
    broker.push_bar(_bar())  # 当日买入 → avail = 0

    with pytest.raises(OrderRejected):
        broker.place_order(_order(1000, side=Side.SELL, idem="s1"))

    position = broker.query_positions()[0]
    assert position.frozen_qty == 0 and position.avail_qty == 0
    assert "T+1" in (broker.query_orders()[-1].reject_reason or "")


# ============================================================ 撤单与顺延
def test_撤单归还冻结资金() -> None:
    broker = _broker()
    broker_order_id = broker.place_order(_order(1000))

    assert broker.cancel_order(broker_order_id) is True

    account = broker.query_account()
    assert account.frozen_cash == Decimal("0"), "撤单必须归还冻结资金"
    assert account.available_cash == Decimal("1000000")
    assert broker.query_orders()[0].status is OrderStatus.CANCELED


def test_撤单未知委托号返回False() -> None:
    broker = _broker()
    assert broker.cancel_order("not-exists") is False


def test_停牌顺延超限自动撤单并归还冻结() -> None:
    broker = _broker(max_defer_days=1)
    broker.place_order(_order(1000))
    suspended_bar = _bar("10.00", volume=0, is_suspended=True)

    broker.push_bar(suspended_bar)
    broker.push_bar(suspended_bar)  # 超过 1 日 → 放弃

    order = broker.query_orders()[0]
    assert order.status is OrderStatus.CANCELED
    assert "顺延" in (order.reject_reason or "")
    assert broker.query_account().frozen_cash == Decimal("0"), "顺延超限也必须归还冻结"


def test_一字涨停买单顺延不成交() -> None:
    broker = _broker()
    broker.place_order(_order(1000))
    # 一字涨停：开=高=低=收=11.00，且等于涨停价
    limit_bar = _bar("11.00", limit_up="11.00", limit_down="9.00")

    trades = broker.push_bar(limit_bar)

    assert trades == []
    assert broker.query_orders()[0].status is OrderStatus.SUBMITTED, "顺延期间仍在途"


def test_部分成交时归还对应比例冻结() -> None:
    broker = _broker()
    broker.place_order(_order(20_000))
    frozen_before = broker.query_account().frozen_cash

    trades = broker.push_bar(_bar("10.00", volume=100_000))  # 10% = 10000 股

    assert trades and trades[0].qty == 10_000
    account = broker.query_account()
    assert Decimal("0") < account.frozen_cash < frozen_before, "部分成交只能释放对应比例的冻结"
    assert broker.query_orders()[0].status is OrderStatus.PARTIAL


# ============================================================ 查询与回调
def test_查询返回副本不污染内部账本() -> None:
    broker = _broker()
    broker.place_order(_order(1000))
    broker.push_bar(_bar())
    broker.roll_to_next_day()

    position = broker.query_positions()[0]
    position.qty = 999_999
    position.avail_qty = 999_999
    account = broker.query_account()
    account.cash = Decimal("1")

    assert broker.query_positions()[0].qty == 1000, "返回的必须是快照"
    assert broker.query_account().cash != Decimal("1")


def test_成交通知与订单状态回调() -> None:
    broker = _broker()
    orders: list[OrderStatus] = []
    trades: list[int] = []
    broker.on_order_update(lambda order: orders.append(order.status))
    broker.on_trade_update(lambda trade: trades.append(trade.qty))

    broker.place_order(_order(1000))
    broker.push_bar(_bar())

    assert orders[0] is OrderStatus.SUBMITTED and orders[-1] is OrderStatus.FILLED
    assert trades == [1000]


def test_恢复后委托簿可查且不被误判为废单() -> None:
    """回归：restore() 曾漏掉委托簿，导致重启后 query_orders() 为空，
    查单校准会把所有在途委托误判成废单并永久标记 REJECTED。"""
    broker = _broker()
    broker.place_order(_order(1000))
    account = broker.query_account()
    positions = {position.code: position for position in broker.query_positions()}
    orders = list(broker.query_orders())

    restarted = PaperBrokerAdapter(initial_cash="1000000")
    restarted.connect()
    restarted.restore(
        account=account,
        positions=positions,
        last_prices=broker.last_prices,
        orders={order.order_id: order for order in orders},
        open_orders=broker.open_order_records(),
    )

    assert len(restarted.query_orders()) == 1, "恢复后必须能看到既有委托"
    assert restarted.open_order_records(), "在途委托也要恢复"


def test_初始资金必须为正() -> None:
    with pytest.raises(ValidationError):
        PaperBrokerAdapter(initial_cash="0")
