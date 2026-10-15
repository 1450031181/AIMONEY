"""交易成本模型（P1-B1）。

口径（A 股，2026 年，可配置覆盖）：
- 佣金：成交额 × 万 3，单笔最低 5 元（买卖都收）；
- 印花税：成交额 × 千 1，**仅卖出**；
- 过户费：成交额 × 万 0.1（沪深两市统一按此近似，精确规则差异不影响回测结论量级）。

为什么单独建模并逐项拆分：回测里费用不是"扣点钱"那么简单——
佣金有最低收费，意味着小额调仓的实际费率远高于名义费率；
印花税只在卖出端，意味着高换手策略的买卖价差成本不对称。
逐项拆分后才能在归因里回答"收益被交易成本吃掉多少"。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from engine.num import to_decimal
from engine.types import Side

_ZERO = Decimal(0)
_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class FeeModel:
    """费用参数（与券商实际费率核对后可覆盖）。"""

    commission_rate: Decimal = Decimal("0.0003")
    commission_min: Decimal = Decimal("5")
    stamp_tax_rate: Decimal = Decimal("0.001")
    transfer_rate: Decimal = Decimal("0.00001")

    def estimate(self, amount: Decimal | float | str, side: Side) -> FeeBreakdown:
        """按成交金额估算费用（金额不含费用）。金额 ≤ 0 返回全零。"""
        value = to_decimal(amount)
        if value <= 0:
            return FeeBreakdown()
        commission = max(value * self.commission_rate, self.commission_min).quantize(_CENT, ROUND_HALF_UP)
        stamp_tax = (value * self.stamp_tax_rate).quantize(_CENT, ROUND_HALF_UP) if side is Side.SELL else _ZERO
        transfer = (value * self.transfer_rate).quantize(_CENT, ROUND_HALF_UP)
        return FeeBreakdown(commission=commission, stamp_tax=stamp_tax, transfer_fee=transfer)


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    commission: Decimal = _ZERO
    stamp_tax: Decimal = _ZERO
    transfer_fee: Decimal = _ZERO

    @property
    def total(self) -> Decimal:
        return self.commission + self.stamp_tax + self.transfer_fee


#: 资金约束估算时的保守费率上界（佣金按最低 5 元折算到小额单会很高，这里用名义费率 + 固定缓冲）
CASH_BUFFER_RATE: Final[Decimal] = Decimal("0.002")


def affordable_qty(cash: Decimal | float | str, price: Decimal | float | str, lot: int = 100) -> int:
    """在可用资金内最多能买多少股（含费用，向下取整到手）。

    费用随数量递增但非连续（最低佣金 5 元是阶梯），因此先按名义费率估一个上界，
    再逐手递减直到资金够用——次数极少（最多几次），但保证**不会超资金**。
    """
    from engine.num import floor_to_lot

    cash_value = to_decimal(cash)
    price_value = to_decimal(price)
    if cash_value <= 0 or price_value <= 0 or lot <= 0:
        return 0
    fee_model = FeeModel()
    upper = int(cash_value / (price_value * (Decimal(1) + CASH_BUFFER_RATE)))
    qty = floor_to_lot(upper, lot)
    while qty > 0:
        amount = price_value * qty
        if amount + fee_model.estimate(amount, Side.BUY).total <= cash_value:
            return qty
        qty -= lot
    return 0


__all__ = ["CASH_BUFFER_RATE", "FeeBreakdown", "FeeModel", "affordable_qty"]
