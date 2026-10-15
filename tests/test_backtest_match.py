"""P1-B1 撮合器测试：逐条 A 股规则独立验证。

这些用例是"回测可信度"的根基——每条规则错误都不会报异常，
只会让回测收益系统性失真（比如涨停也能买入，会凭空多出打板收益）。
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from decimal import Decimal

import pytest

from engine.backtest.match import FillStatus, MatchEngine, MatcherConfig
from engine.backtest.slippage import FixedBpsSlippage
from engine.errors import ValidationError
from engine.types import Account, Bar, OrderPlan, Position, Side

DAY = datetime(2026, 10, 15)


def _sell(plan: OrderPlan, **overrides: object) -> OrderPlan:
    """把买入计划翻转为卖出（OrderPlan 是 frozen+slots，用 replace 派生）。"""
    values: dict[str, object] = {"side": Side.SELL, **overrides}
    return dataclasses.replace(plan, **values)  # type: ignore[arg-type]


def _bar(**overrides: object) -> Bar:
    values: dict[str, object] = {
        "code": "600000",
        "dt": DAY,
        "open": Decimal("10.00"),
        "high": Decimal("10.20"),
        "low": Decimal("9.90"),
        "close": Decimal("10.10"),
        "volume": 1_000_000,
        "amount": Decimal("10_100_000"),
        "limit_up": Decimal("11.00"),
        "limit_down": Decimal("9.00"),
    }
    values.update(overrides)
    return Bar(**values)  # type: ignore[arg-type]


def _buy_plan(qty: int = 1000, **overrides: object) -> OrderPlan:
    values: dict[str, object] = {
        "code": "600000",
        "strategy_id": "s1",
        "trade_date": DAY.date(),
        "side": Side.BUY,
        "target_qty": qty,
    }
    values.update(overrides)
    return OrderPlan(**values)  # type: ignore[arg-type]


def _account(cash: str = "100000") -> Account:
    return Account(initial_cash=Decimal(cash), cash=Decimal(cash))


def _position(code: str = "600000", *, qty: int = 1000, avail: int | None = None) -> Position:
    return Position(code=code, qty=qty, avail_qty=qty if avail is None else avail)


# ============================================================ 正常成交
def test_正常买入按开盘价全量成交() -> None:
    engine = MatchEngine()
    attempt = engine.attempt(_buy_plan(1000), _bar(), account=_account(), positions={})

    assert attempt.status is FillStatus.FILLED
    assert attempt.qty == 1000
    assert attempt.price == Decimal("10.00")


def test_正常卖出全量成交() -> None:
    engine = MatchEngine()
    sell_plan = _sell(_buy_plan(1000))

    attempt = engine.attempt(sell_plan, _bar(), account=_account(), positions={"600000": _position(qty=1000)})

    assert attempt.status is FillStatus.FILLED
    assert attempt.qty == 1000


def test_可用收盘价与均价模式() -> None:
    close_engine = MatchEngine(MatcherConfig(price_mode="close"))
    assert close_engine.attempt(_buy_plan(100), _bar(), account=_account(), positions={}).price == Decimal("10.10")

    vwap_engine = MatchEngine(MatcherConfig(price_mode="vwap"))
    # vwap = 10_100_000 / 1_000_000 = 10.10
    assert vwap_engine.attempt(_buy_plan(100), _bar(), account=_account(), positions={}).price == Decimal("10.10")


def test_滑点影响成交价方向() -> None:
    engine = MatchEngine(slippage=FixedBpsSlippage(bps=10))
    buy = engine.attempt(_buy_plan(100), _bar(), account=_account(), positions={})
    assert buy.price == Decimal("10.01"), "买入成交价应上浮 10bp"


# ============================================================ 顺延类约束
def test_停牌顺延() -> None:
    engine = MatchEngine()
    attempt = engine.attempt(_buy_plan(100), _bar(is_suspended=True, volume=0), account=_account(), positions={})

    assert attempt.status is FillStatus.DEFERRED
    assert "停牌" in attempt.reason


def test_无行情顺延() -> None:
    engine = MatchEngine()
    attempt = engine.attempt(_buy_plan(100), None, account=_account(), positions={})

    assert attempt.status is FillStatus.DEFERRED


def test_一字涨停买入顺延() -> None:
    engine = MatchEngine()
    bar = _bar(open=Decimal("11.00"), high=Decimal("11.00"), low=Decimal("11.00"), close=Decimal("11.00"))
    attempt = engine.attempt(_buy_plan(100), bar, account=_account(), positions={})

    assert attempt.status is FillStatus.DEFERRED
    assert "涨停" in attempt.reason


def test_一字跌停卖出顺延() -> None:
    engine = MatchEngine()
    bar = _bar(open=Decimal("9.00"), high=Decimal("9.00"), low=Decimal("9.00"), close=Decimal("9.00"))
    sell_plan = _sell(_buy_plan(100))

    attempt = engine.attempt(sell_plan, bar, account=_account(), positions={"600000": _position(qty=1000)})

    assert attempt.status is FillStatus.DEFERRED
    assert "跌停" in attempt.reason


def test_未涨停日可以买入() -> None:
    engine = MatchEngine()
    bar = _bar(open=Decimal("10.90"), high=Decimal("11.00"), low=Decimal("10.50"), close=Decimal("10.95"))
    attempt = engine.attempt(_buy_plan(100), bar, account=_account(), positions={})

    assert attempt.status is FillStatus.FILLED, "10.90 < limit_up 11.00，不是一字板，应可成交"


# ============================================================ 成交量约束
def test_买入受当日成交量约束部分成交() -> None:
    engine = MatchEngine(MatcherConfig(max_vol_ratio=0.10))
    bar = _bar(volume=100_000)  # 10% = 10000 股 < 计划 50000 股
    attempt = engine.attempt(_buy_plan(50_000), bar, account=_account(cash="1000000"), positions={})

    assert attempt.status is FillStatus.PARTIAL
    assert attempt.qty == 10_000


def test_卖出受成交量与可卖双重约束() -> None:
    engine = MatchEngine(MatcherConfig(max_vol_ratio=0.10))
    bar = _bar(volume=100_000)  # 10% = 10000 股
    sell_plan = _sell(_buy_plan(20_000))

    attempt = engine.attempt(
        sell_plan, bar, account=_account(), positions={"600000": _position(qty=15_000, avail=15_000)}
    )

    assert attempt.status is FillStatus.PARTIAL
    assert attempt.qty == 10_000, "受成交量约束（10000）而非可卖（15000）"


# ============================================================ 资金与持仓约束
def test_资金不足时按整手缩量且不超资金() -> None:
    engine = MatchEngine()
    attempt = engine.attempt(_buy_plan(10_000), _bar(), account=_account(cash="20000"), positions={})

    assert attempt.status is FillStatus.PARTIAL
    assert attempt.qty % 100 == 0
    amount = attempt.price * attempt.qty  # type: ignore[operator]
    fees = amount * Decimal("0.0003")
    assert amount + fees <= Decimal("20000"), "缩量后含费成本不得超过可用资金"


def test_资金完全不足时顺延而不是部分成交() -> None:
    engine = MatchEngine()
    attempt = engine.attempt(_buy_plan(1000), _bar(), account=_account(cash="500"), positions={})

    assert attempt.status is FillStatus.DEFERRED
    assert "资金" in attempt.reason


def test_可卖不足时部分成交() -> None:
    engine = MatchEngine()
    sell_plan = _sell(_buy_plan(1000))

    attempt = engine.attempt(
        sell_plan, _bar(), account=_account(), positions={"600000": _position(qty=1000, avail=600)}
    )

    assert attempt.status is FillStatus.PARTIAL
    assert attempt.qty == 600, "可卖 600 股只能成交 600 股"


def test_可卖为零时顺延而不是拒绝() -> None:
    """T+1 当日买入后 avail=0：卖出请求应顺延到次日，而不是丢弃信号。"""
    engine = MatchEngine()
    sell_plan = _sell(_buy_plan(1000))

    attempt = engine.attempt(
        sell_plan, _bar(), account=_account(), positions={"600000": _position(qty=1000, avail=0)}
    )

    assert attempt.status is FillStatus.DEFERRED
    assert "可卖" in attempt.reason


def test_无持仓卖出顺延() -> None:
    engine = MatchEngine()
    sell_plan = _sell(_buy_plan(100))

    attempt = engine.attempt(sell_plan, _bar(), account=_account(), positions={})

    assert attempt.status is FillStatus.DEFERRED


# ============================================================ 限价语义
def test_限价买入未触及则不成交也不追价() -> None:
    engine = MatchEngine()
    plan = _buy_plan(100, limit_price=Decimal("9.95"))  # 开盘 10.00 > 9.95
    attempt = engine.attempt(plan, _bar(), account=_account(), positions={})

    assert attempt.status is FillStatus.DEFERRED
    assert "限价" in attempt.reason


def test_限价触及则按市场价成交且不劣于限价() -> None:
    """可成交的限价单按市场价成交（10.00 优于限价 10.05），与实盘一致。"""
    engine = MatchEngine()
    plan = _buy_plan(100, limit_price=Decimal("10.05"))
    attempt = engine.attempt(plan, _bar(), account=_account(), positions={})

    assert attempt.status is FillStatus.FILLED
    assert attempt.price == Decimal("10.00")
    assert attempt.price <= plan.limit_price


def test_限价卖出未触及则顺延() -> None:
    engine = MatchEngine()
    sell_plan = _sell(_buy_plan(100), limit_price=Decimal("10.50"))

    attempt = engine.attempt(sell_plan, _bar(), account=_account(), positions={"600000": _position()})

    assert attempt.status is FillStatus.DEFERRED


# ============================================================ 非法输入
def test_非法计划数量被上游模型拒绝() -> None:
    with pytest.raises(ValidationError):
        _buy_plan(0)


def test_成交量约束为零时顺延() -> None:
    engine = MatchEngine(MatcherConfig(max_vol_ratio=0.0))
    attempt = engine.attempt(_buy_plan(100), _bar(), account=_account(), positions={})

    assert attempt.status is FillStatus.DEFERRED


def test_日期一致性由调用方保证() -> None:
    """撮合器不做日期校验（bar 的日期由引擎保证与当前时点一致），这里只锁约定。"""
    bar = _bar()
    assert bar.dt.date() == date(2026, 10, 15)
