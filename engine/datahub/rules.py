"""A 股交易规则（T-09）：板块、涨跌停幅度、涨跌停价。

为什么单独成模块：涨跌停幅度决定三件事的正确性——回测撮合（一字板能否成交）、
涨跌停价还原、以及复权连续性校验的容差。这三处若各写各的阈值，迟早不一致。

已知规则（2026 年）：
- 沪深主板：±10%，ST 为 ±5%
- 创业板、科创板：±20%（ST 同为 ±20%）
- 北交所：±30%
- 涨跌停价按最小报价单位 0.01 **四舍五入**（交易所规则，非四舍六入五成双）
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from engine.errors import ValidationError
from engine.num import PRICE_TICK, to_decimal

BOARD_SH_MAIN: Final[str] = "沪市主板"
BOARD_SZ_MAIN: Final[str] = "深市主板"
BOARD_CHINEXT: Final[str] = "创业板"
BOARD_STAR: Final[str] = "科创板"
BOARD_BSE: Final[str] = "北交所"
BOARD_UNKNOWN: Final[str] = "未知"

MAIN_BOARDS: Final[frozenset[str]] = frozenset({BOARD_SH_MAIN, BOARD_SZ_MAIN})
#: 无涨跌幅限制的场景（新股上市首日等）不在本期支持范围，故不提供 0 幅度分支

_LIMIT_RATIOS: Final[dict[str, Decimal]] = {
    BOARD_SH_MAIN: Decimal("0.10"),
    BOARD_SZ_MAIN: Decimal("0.10"),
    BOARD_CHINEXT: Decimal("0.20"),
    BOARD_STAR: Decimal("0.20"),
    BOARD_BSE: Decimal("0.30"),
}
_ST_LIMIT_RATIO: Final[Decimal] = Decimal("0.05")


def infer_board(code: str) -> str:
    """由代码前缀推断板块。"""
    if code.startswith("68"):
        return BOARD_STAR
    if code.startswith("60"):
        return BOARD_SH_MAIN
    if code.startswith("30"):
        return BOARD_CHINEXT
    if code.startswith("00"):
        return BOARD_SZ_MAIN
    if code.startswith(("8", "4")):
        return BOARD_BSE
    return BOARD_UNKNOWN


def price_limit_ratio(board: str, *, is_st: bool = False) -> Decimal:
    """板块 + ST 状态对应的涨跌幅比例。

    未知板块**直接报错**而不是默认 10%：静默使用错误阈值会让回测撮合与复权校验
    同时失真，而这两处失真都不会抛异常。
    """
    if board == BOARD_UNKNOWN:
        raise ValidationError("无法确定板块，拒绝猜测涨跌停幅度", detail={"board": board})
    if is_st and board in MAIN_BOARDS:
        return _ST_LIMIT_RATIO
    return _LIMIT_RATIOS[board]


def limit_prices(
    prev_close: Decimal | float | str,
    *,
    code: str,
    is_st: bool = False,
    board: str | None = None,
) -> tuple[Decimal, Decimal]:
    """由前收盘价计算涨停价与跌停价（四舍五入到 0.01）。"""
    resolved_board = board or infer_board(code)
    ratio = price_limit_ratio(resolved_board, is_st=is_st)
    base = to_decimal(prev_close)
    if base <= 0:
        raise ValidationError("前收盘价必须为正", detail={"code": code, "prev_close": str(base)})
    limit_up = (base * (Decimal(1) + ratio)).quantize(PRICE_TICK, rounding=ROUND_HALF_UP)
    limit_down = (base * (Decimal(1) - ratio)).quantize(PRICE_TICK, rounding=ROUND_HALF_UP)
    return limit_up, limit_down


__all__ = [
    "BOARD_BSE",
    "BOARD_CHINEXT",
    "BOARD_SH_MAIN",
    "BOARD_STAR",
    "BOARD_SZ_MAIN",
    "BOARD_UNKNOWN",
    "MAIN_BOARDS",
    "infer_board",
    "limit_prices",
    "price_limit_ratio",
]
