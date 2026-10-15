"""T-09 A 股交易规则测试：板块涨跌停幅度、涨跌停价取整、未知板块必须报错。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from engine.datahub.rules import (
    BOARD_BSE,
    BOARD_CHINEXT,
    BOARD_SH_MAIN,
    BOARD_STAR,
    BOARD_SZ_MAIN,
    BOARD_UNKNOWN,
    infer_board,
    limit_prices,
    price_limit_ratio,
)
from engine.errors import ValidationError


@pytest.mark.parametrize(
    ("board", "expected"),
    [
        (BOARD_SH_MAIN, "0.10"),
        (BOARD_SZ_MAIN, "0.10"),
        (BOARD_CHINEXT, "0.20"),
        (BOARD_STAR, "0.20"),
        (BOARD_BSE, "0.30"),
    ],
)
def test_各板块涨跌停幅度(board: str, expected: str) -> None:
    assert price_limit_ratio(board) == Decimal(expected)


def test_主板ST为5个百分点而创业板ST仍为20() -> None:
    assert price_limit_ratio(BOARD_SH_MAIN, is_st=True) == Decimal("0.05")
    assert price_limit_ratio(BOARD_SZ_MAIN, is_st=True) == Decimal("0.05")
    assert price_limit_ratio(BOARD_CHINEXT, is_st=True) == Decimal("0.20")


def test_未知板块拒绝猜测() -> None:
    with pytest.raises(ValidationError):
        price_limit_ratio(BOARD_UNKNOWN)


def test_代码前缀推断板块() -> None:
    assert infer_board("688981") == BOARD_STAR
    assert infer_board("601398") == BOARD_SH_MAIN
    assert infer_board("000001") == BOARD_SZ_MAIN
    assert infer_board("300750") == BOARD_CHINEXT
    assert infer_board("830799") == BOARD_BSE
    assert infer_board("430047") == BOARD_BSE
    assert infer_board("999999") == BOARD_UNKNOWN


def test_涨跌停价四舍五入到分() -> None:
    limit_up, limit_down = limit_prices("9.99", code="600000")
    # 9.99 × 1.1 = 10.989 -> 10.99；9.99 × 0.9 = 8.991 -> 8.99
    assert (limit_up, limit_down) == (Decimal("10.99"), Decimal("8.99"))


def test_ST股涨跌停价() -> None:
    limit_up, limit_down = limit_prices("10.00", code="600000", is_st=True)
    assert (limit_up, limit_down) == (Decimal("10.50"), Decimal("9.50"))


def test_创业板涨跌停价() -> None:
    limit_up, limit_down = limit_prices("100.00", code="300750")
    assert (limit_up, limit_down) == (Decimal("120.00"), Decimal("80.00"))


def test_前收非法时报错() -> None:
    with pytest.raises(ValidationError):
        limit_prices("0", code="600000")
    with pytest.raises(ValidationError):
        limit_prices("-1", code="600000")


def test_可显式指定板块跳过推断() -> None:
    limit_up, _ = limit_prices("10.00", code="999999", board=BOARD_BSE)
    assert limit_up == Decimal("13.00")
