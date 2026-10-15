"""滑点模型（P1-B1）。

为什么必须有滑点：回测里的成交价是"事后已知的价格"，实盘是"你下单之后市场对你的回应"。
不含滑点的回测会系统性高估收益——尤其是成交量小、调仓量大的策略。

模型刻意保持简单（可插拔）：一期用固定 bps；按成交量冲击的非线性模型留到有真实
成交数据校准之后再上，否则参数就是拍脑袋。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from engine.num import round_price, to_decimal
from engine.types import Side

_ZERO = Decimal(0)


class SlippageModel(Protocol):
    """成交价调整模型：返回**实际成交价**（已含滑点方向）。"""

    def apply(self, price: Decimal | float | str, side: Side) -> Decimal: ...


class NoSlippage:
    """零滑点（仅用于基准对照，实盘模拟不建议）。

    参数名 ``_side`` 与协议保持位置一致：零滑点本就不区分方向。
    """

    def apply(self, price: Decimal | float | str, _side: Side) -> Decimal:
        return round_price(price)


class FixedBpsSlippage:
    """固定基点滑点：买入价上浮、卖出价下浮（方向永远对你不利）。"""

    def __init__(self, bps: float = 5.0) -> None:
        if bps < 0:
            raise ValueError("bps 不能为负")
        self._ratio = Decimal(str(bps)) / Decimal(10000)

    def apply(self, price: Decimal | float | str, side: Side) -> Decimal:
        value = to_decimal(price)
        if value <= 0:
            raise ValueError("价格必须为正")
        slipped = value * (Decimal(1) + self._ratio) if side is Side.BUY else value * (Decimal(1) - self._ratio)
        return round_price(slipped)


__all__ = ["FixedBpsSlippage", "NoSlippage", "SlippageModel"]
