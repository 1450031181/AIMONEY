"""P1-B1 成本与滑点测试。

费用是最容易被低估的回测项：最低佣金 5 元意味着小额调仓的实际费率远高于名义费率；
印花税只在卖出端，意味着高换手策略的成本不对称。这两个特性必须被模型保真。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from engine.backtest.cost import FeeModel, affordable_qty
from engine.backtest.slippage import FixedBpsSlippage, NoSlippage
from engine.types import Side


# ============================================================ 费用
def test_小额买单触发最低佣金() -> None:
    breakdown = FeeModel().estimate("10000", Side.BUY)  # 1 万元，名义佣金 3 元
    assert breakdown.commission == Decimal("5.00"), "最低收费 5 元必须生效"
    assert breakdown.stamp_tax == Decimal("0.00"), "印花税只对卖出收取"
    assert breakdown.transfer_fee == Decimal("0.10")
    assert breakdown.total == Decimal("5.10")


def test_大额卖单含印花税() -> None:
    breakdown = FeeModel().estimate("100000", Side.SELL)
    assert breakdown.commission == Decimal("30.00")
    assert breakdown.stamp_tax == Decimal("100.00")
    assert breakdown.transfer_fee == Decimal("1.00")
    assert breakdown.total == Decimal("131.00")


def test_同金额买卖费用不对称() -> None:
    model = FeeModel()
    assert model.estimate("100000", Side.BUY).total < model.estimate("100000", Side.SELL).total


def test_零金额返回全零() -> None:
    assert FeeModel().estimate("0", Side.BUY).total == Decimal("0.00")


# ============================================================ 可买数量
def test_可买数量按整手向下取整() -> None:
    # 10000 元 / 10 元 = 1000 股，扣费后应略少，但必须是 100 的倍数
    qty = affordable_qty("10000", "10.00")
    assert qty % 100 == 0
    assert qty <= 1000


def test_可买数量绝不超资金() -> None:
    qty = affordable_qty("10000", "10.00")
    amount = Decimal("10.00") * qty
    total = amount + FeeModel().estimate(amount, Side.BUY).total
    assert total <= Decimal("10000"), "含费成本不得超过可用资金"


def test_资金不够一手时返回零() -> None:
    assert affordable_qty("500", "10.00") == 0, "500 元买不起 1000 元一手，必须返回 0"


def test_零资金或非法价格返回零() -> None:
    assert affordable_qty("0", "10.00") == 0
    assert affordable_qty("10000", "0") == 0


# ============================================================ 滑点
def test_零滑点原价返回() -> None:
    assert NoSlippage().apply("10.00", Side.BUY) == Decimal("10.00")


def test_固定滑点方向永远不利() -> None:
    model = FixedBpsSlippage(bps=10)
    buy = model.apply("10.00", Side.BUY)
    sell = model.apply("10.00", Side.SELL)
    assert buy == Decimal("10.01")
    assert sell == Decimal("9.99")
    assert sell < NoSlippage().apply("10.00", Side.SELL) < buy


def test_负滑点被拒() -> None:
    with pytest.raises(ValueError):
        FixedBpsSlippage(bps=-1)
