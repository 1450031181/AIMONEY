"""数值工具：A 股价格/数量口径与安全的 Decimal 转换。

设计约束：
- 价格、金额一律用 ``Decimal``，禁止用 float 参与资金累加（0.1+0.2 != 0.3 会污染净值）；
- ``to_decimal`` 对 float 走 repr 转换，避免二进制浮点噪声（0.1 -> 0.1 而非 0.1000000000000000055）。
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

# A 股最小报价单位（元）
PRICE_TICK = Decimal("0.01")
# 均价/成本保留精度
COST_PRECISION = Decimal("0.0001")
# A 股一手 = 100 股（买入必须为整数手；卖出允许零股，故仅在买入侧取整）
LOT_SIZE = 100

_NumberLike = int | float | str | Decimal


def to_decimal(value: _NumberLike) -> Decimal:
    """把 int/float/str/Decimal 安全转换为 Decimal。

    float 走 ``repr`` 再转换：``to_decimal(0.1) -> Decimal('0.1')``。
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):  # bool 是 int 子类，显式拒绝避免误用
        raise TypeError("bool 不能作为数值使用")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    raise TypeError(f"无法转换为 Decimal：{type(value).__name__}")


def round_price(price: _NumberLike, tick: Decimal = PRICE_TICK) -> Decimal:
    """按最小报价单位四舍五入。"""
    return to_decimal(price).quantize(tick, rounding=ROUND_HALF_UP)


def round_cost(value: _NumberLike) -> Decimal:
    """成本/均价保留 4 位小数。"""
    return to_decimal(value).quantize(COST_PRECISION, rounding=ROUND_HALF_UP)


def floor_to_lot(qty: int, lot: int = LOT_SIZE) -> int:
    """向下取整到整数手（买入用）。"""
    if lot <= 0:
        raise ValueError("lot 必须为正整数")
    return qty // lot * lot


def floor_money(value: _NumberLike, tick: Decimal = PRICE_TICK) -> Decimal:
    """金额向下取整（用于可买数量上限计算，避免超买）。"""
    return to_decimal(value).quantize(tick, rounding=ROUND_DOWN)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """除零安全的浮点除法。"""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    """把 value 夹在 [low, high] 区间内。"""
    if low > high:
        raise ValueError(f"区间非法：low({low}) > high({high})")
    return max(low, min(high, value))


__all__ = [
    "COST_PRECISION",
    "LOT_SIZE",
    "PRICE_TICK",
    "clamp",
    "floor_money",
    "floor_to_lot",
    "round_cost",
    "round_price",
    "safe_div",
    "to_decimal",
]
