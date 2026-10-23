"""P4-4 收益归因测试（每笔成本都手算）。

为什么必须手算：归因是"钱从哪来"的唯一答案，错了不会报错，只会让人对策略产生
错误认知（比如把成本当成收益、把已清仓标的的亏损凭空丢掉）。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from engine.errors import ValidationError
from engine.report.attribution import PortfolioPnl, compute_attribution, per_symbol_returns
from engine.types import Position, Side, Trade

DAY = datetime(2026, 10, 15, 9, 30)


class _Approx:
    """Decimal / float 通用的近似比较。

    直接用 ``pytest.approx`` 比 ``Decimal`` 会抛 ``TypeError``（Decimal − float 不合法），
    而这里所有金额都是 Decimal —— 转成 float 再比是唯一不污染被测代码的做法。
    """

    def __init__(self, expected: float, tol: float = 1e-9) -> None:
        self.expected = float(expected)
        self.tol = tol

    def __eq__(self, actual: object) -> bool:
        return abs(float(actual) - self.expected) <= self.tol  # type: ignore[arg-type]


def approx(value: float, tol: float = 1e-9) -> _Approx:
    return _Approx(value, tol)


def _trade(
    code: str,
    side: Side,
    qty: int,
    price: str,
    *,
    commission: str = "0",
    stamp_tax: str = "0",
    transfer_fee: str = "0",
    minute: int = 0,
) -> Trade:
    return Trade(
        code=code,
        side=side,
        qty=qty,
        price=Decimal(price),
        commission=Decimal(commission),
        stamp_tax=Decimal(stamp_tax),
        transfer_fee=Decimal(transfer_fee),
        trade_time=DAY.replace(minute=minute),
    )


def _position(code: str, qty: int, avg_cost: str, last_price: str) -> Position:
    return Position(code=code, qty=qty, avail_qty=qty, avg_cost=Decimal(avg_cost), last_price=Decimal(last_price))


# ============================================================ 手工验证
def test_单笔买入卖出_已实现与浮动盈亏手工验证() -> None:
    """买 1000@10（费 5）→ 卖 500@12（费 11）→ 现价 13。

    - 买入成本基础 = 10000 + 5 = 10005 → 均价 10.005
    - 卖出净收入   = 6000 − 11 = 5989
    - 已实现       = 5989 − 10.005 × 500 = 986.50
    - 剩余 500 股成本 = 10.005 × 500 = 5002.50
    - 浮动         = (13 − 10.005) × 500 = 1497.50
    """
    trades = [
        _trade("600000", Side.BUY, 1000, "10", commission="5", minute=1),
        _trade("600000", Side.SELL, 500, "12", commission="5", stamp_tax="6", minute=2),
    ]

    # 账本里的 avg_cost 是**不含费**成交均价（券商口径），归因的 avg_cost 含费
    result = compute_attribution(trades, [_position("600000", 500, "10.0000", "13")])

    book = result.symbols[0]
    assert book.realized_pnl == approx(986.50)
    assert book.unrealized_pnl == approx(1497.50)
    assert book.total_pnl == approx(2484.00)
    assert book.fees == approx(16.0)
    assert book.qty == 500
    assert book.avg_cost == approx(10.005), "归因口径：买入费计入成本"
    assert book.gross_avg_cost == approx(10.000), "对账口径：不含费成交均价"


def test_费用计入成本而不是凭空消失() -> None:
    """买入费进成本基础、卖出费减收入：费用必须体现为收益的减项。"""
    with_fee = compute_attribution(
        [
            _trade("600000", Side.BUY, 1000, "10", commission="5", minute=1),
            _trade("600000", Side.SELL, 1000, "10", commission="5", stamp_tax="5", minute=2),
        ]
    )
    no_fee = compute_attribution(
        [
            _trade("600000", Side.BUY, 1000, "10", minute=1),
            _trade("600000", Side.SELL, 1000, "10", minute=2),
        ]
    )

    assert no_fee.realized_pnl == approx(0.0), "同价买卖不赚不亏"
    assert with_fee.realized_pnl == approx(-15.0), "费用必须全部体现为亏损"
    assert with_fee.total_fees == approx(15.0)


def test_清仓后均价归零且已实现保留() -> None:
    trades = [
        _trade("600000", Side.BUY, 1000, "10", commission="5", minute=1),
        _trade("600000", Side.SELL, 1000, "11", commission="5", stamp_tax="5.5", minute=2),
    ]

    book = compute_attribution(trades).symbols[0]

    assert book.qty == 0
    assert book.avg_cost == 0
    assert book.unrealized_pnl == 0, "清仓后不应再有浮动盈亏"
    # 卖出 11000 − 费 10.5 − 成本 10005 = 984.5
    assert book.realized_pnl == approx(984.5)
    assert book.closed


def test_多次加仓后均价按加权成本() -> None:
    trades = [
        _trade("600000", Side.BUY, 1000, "10", minute=1),
        _trade("600000", Side.BUY, 1000, "12", minute=2),
    ]

    book = compute_attribution(trades, [_position("600000", 2000, "11.0000", "11")]).symbols[0]

    assert book.avg_cost == approx(11.0), "无费时均价就是两次成交的加权平均"
    assert book.unrealized_pnl == approx(0.0)


def test_费用率与换手统计() -> None:
    trades = [
        _trade("600000", Side.BUY, 1000, "10", commission="5", minute=1),
        _trade("600000", Side.SELL, 500, "12", commission="5", stamp_tax="6", minute=2),
    ]

    result = compute_attribution(trades)

    assert result.buy_turnover == approx(10000.0)
    assert result.sell_turnover == approx(6000.0)
    assert result.turnover == approx(16000.0)
    assert result.fee_ratio == approx(16.0 / 16000.0)
    assert result.trade_count == 2


def test_多标的分别统计且合计等于各项之和() -> None:
    trades = [
        _trade("600000", Side.BUY, 1000, "10", commission="5", minute=1),
        _trade("000001", Side.BUY, 500, "20", commission="5", minute=2),
        _trade("600000", Side.SELL, 1000, "11", commission="5", stamp_tax="5.5", minute=3),
    ]

    result = compute_attribution(
        trades,
        [
            _position("000001", 500, "20.0000", "21"),
        ],
    )

    assert {book.code for book in result.symbols} == {"600000", "000001"}
    assert result.realized_pnl == approx(sum(book.realized_pnl for book in result.symbols))
    assert result.unrealized_pnl == approx(sum(book.unrealized_pnl for book in result.symbols))
    # 000001：(21 − 20.01) × 500 = 495
    assert result.unrealized_pnl == approx(495.0)


def test_盈亏排序与费用摘要() -> None:
    trades = [
        _trade("600000", Side.BUY, 1000, "10", minute=1),
        _trade("600000", Side.SELL, 1000, "12", minute=2),  # 赚 2000
        _trade("000001", Side.BUY, 1000, "20", minute=3),
        _trade("000001", Side.SELL, 1000, "19", minute=4),  # 亏 1000
    ]

    result = compute_attribution(trades)

    assert [book.code for book in result.winners(1)] == ["600000"]
    assert [book.code for book in result.losers(1)] == ["000001"]
    summary = result.summary()
    assert "已实现" in summary and "费用" in summary


# ============================================================ 对账校验
def test_重放数量与持仓不一致时报错() -> None:
    """账本与流水分叉时必须报错：分叉的账本算出来的归因只会误导人。"""
    trades = [_trade("600000", Side.BUY, 1000, "10", minute=1)]

    with pytest.raises(ValidationError) as excinfo:
        compute_attribution(trades, [_position("600000", 800, "10.0000", "10")])

    assert "数量" in str(excinfo.value)


def test_重放成本与持仓不一致时报错() -> None:
    trades = [_trade("600000", Side.BUY, 1000, "10", minute=1)]

    with pytest.raises(ValidationError) as excinfo:
        compute_attribution(trades, [_position("600000", 1000, "9.0000", "10")])

    assert "成交均价" in str(excinfo.value)


def test_可关闭对账校验() -> None:
    """未平仓标的没有持仓记录时（例如只看历史流水），允许跳过对账。"""
    trades = [_trade("600000", Side.BUY, 1000, "10", minute=1)]

    result = compute_attribution(trades, check_reconcile=False)

    assert result.symbols[0].qty == 1000


def test_卖出超过累计买入时报错() -> None:
    trades = [_trade("600000", Side.SELL, 100, "10", minute=1)]

    with pytest.raises(ValidationError) as excinfo:
        compute_attribution(trades, check_reconcile=False)

    assert "卖出数量超过" in str(excinfo.value)


# ============================================================ 贡献率
def test_对初始资金的贡献率() -> None:
    trades = [
        _trade("600000", Side.BUY, 1000, "10", minute=1),
        _trade("600000", Side.SELL, 1000, "11", minute=2),
    ]

    returns = per_symbol_returns(trades, Decimal("100000"))

    assert returns["600000"] == approx(1000.0 / 100000.0)


def test_空流水的摘要可读() -> None:
    result = PortfolioPnl()

    assert result.total_pnl == 0
    assert result.fee_ratio == 0.0
    assert "无成交" in result.summary()
