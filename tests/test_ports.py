"""T-04 端口契约测试。

最重要的一条：**Context 不得含任何下单方法**。它是"风控不可绕过"的防线，
一旦有人图方便往 Context 上加了 place_order，策略就能绕开风控直接发单。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from engine.ports import (
    BrokerAdapter,
    Context,
    DataSource,
    PortfolioSnapshot,
    RiskContext,
    Strategy,
    TradingCalendar,
)
from engine.types import (
    Account,
    Bar,
    OrderPlan,
    Position,
    RiskAction,
    RiskDecision,
    RiskLevel,
    Side,
    Signal,
    Trade,
)

#: 这些名字一旦出现在 Context 上就说明策略可以自己发单 —— 必须为 0
FORBIDDEN_ON_CONTEXT = {
    "place",
    "place_order",
    "place_orders",
    "buy",
    "sell",
    "order",
    "submit",
    "cancel",
    "cancel_order",
    "send_order",
    "trade",
}


def _protocol_members(protocol: type) -> set[str]:
    return set(vars(protocol)) | set(getattr(protocol, "__annotations__", {}))


def test_策略上下文不得暴露任何下单方法() -> None:
    leaked = _protocol_members(Context) & FORBIDDEN_ON_CONTEXT
    assert leaked == set(), f"Context 泄漏了交易能力：{sorted(leaked)}"


def test_策略上下文只提供只读数据访问() -> None:
    members = {name for name in vars(Context) if not name.startswith("_")}
    assert {
        "strategy_id",
        "trade_date",
        "now",
        "params",
        "universe",
        "bars",
        "history",
        "factor",
        "positions",
        "position",
        "account",
    } <= members


def test_券商端口必须保留下单与撤单能力() -> None:
    """反向断言：下单能力只在 BrokerAdapter 上，别被误删。"""
    members = _protocol_members(BrokerAdapter)
    assert {"place_order", "cancel_order", "query_orders", "query_positions", "query_account"} <= members


def test_数据源端口契约完整() -> None:
    members = _protocol_members(DataSource)
    assert {"daily", "adjust_factors", "universe", "stock_info", "trading_calendar"} <= members
    assert {"is_trade_day", "prev_trade_day", "next_trade_day", "between", "shift"} <= _protocol_members(
        TradingCalendar
    )


def test_风控上下文可按代码取行情() -> None:
    bar = Bar(code="600000", dt=datetime(2026, 10, 14, 15), open=10, high=11, low=9, close=10, volume=100)
    plan = OrderPlan(code="600000", strategy_id="s1", trade_date=date(2026, 10, 14), side=Side.BUY, target_qty=100)
    ctx = RiskContext(
        trade_date=date(2026, 10, 14),
        plan=plan,
        account=Account(initial_cash=Decimal("100000"), cash=Decimal("100000")),
        positions={},
        bars={"600000": bar},
    )
    assert ctx.bar_of("600000") is bar
    assert ctx.bar_of("000001") is None
    assert ctx.risk_level is RiskLevel.NORMAL


def test_组合快照统计非空持仓数() -> None:
    snapshot = PortfolioSnapshot(
        trade_date=date(2026, 10, 14),
        account=Account(initial_cash=Decimal("100000"), cash=Decimal("100000")),
        positions={
            "600000": Position(code="600000", qty=100, avail_qty=100),
            "000001": Position(code="000001"),  # 空仓不计入
        },
    )
    assert snapshot.position_count == 1


class _HoldStrategy:
    """不继承任何基类，靠结构化子类型满足 Strategy 端口。"""

    strategy_id = "hold"

    def init(self, ctx: Context) -> None:
        return None

    def on_bar(self, ctx: Context, bars: Mapping[str, Bar]) -> Sequence[Signal]:
        return ()

    def on_order_filled(self, ctx: Context, trade: Trade) -> None:
        return None


def test_策略端口为结构化子类型() -> None:
    strategy: Strategy = _HoldStrategy()
    assert strategy.strategy_id == "hold"
    assert list(strategy.on_bar(_FakeContext(), {})) == []


class _FakeContext:
    """最小 Context 实现：证明端口不需要 engine 提供任何基类。"""

    strategy_id = "fake"
    trade_date = date(2026, 10, 14)
    now = datetime(2026, 10, 14, 15, 0)
    params: Mapping[str, Any] = {}

    def universe(self) -> Sequence[str]:
        return ()

    def bars(self, codes: Sequence[str] | None = None) -> Mapping[str, Bar]:
        return {}

    def history(
        self, code: str, *, field: str = "close", window: int = 1, adj: str = "post"
    ) -> list[Decimal]:
        return []

    def factor(self, name: str, code: str) -> float | None:
        return None

    def positions(self) -> Mapping[str, Position]:
        return {}

    def position(self, code: str) -> Position | None:
        return None

    def account(self) -> Account:
        return Account(initial_cash=Decimal("100000"), cash=Decimal("100000"))


def test_风控决策可用于策略无关的规则链() -> None:
    decision = RiskDecision.reject("blacklist", "ST 股禁止买入", requested_qty=100)
    assert decision.action is RiskAction.REJECT
    assert not decision.passed
