"""T-09 复权测试：三种口径的换算正确性 + 连续性探针。

最关键的两条断言：
1. 前复权的**最新一日**必须等于真实价（口径自洽）；
2. 后复权序列缺因子时会产生远超涨跌停的跳变，必须能被探测出来。
   第二条是"复权因子漏乘"的唯一自动防线——这种错误在净值曲线上只是"多赚一点"，肉眼极难发现。
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from engine.datahub.adjust import (
    FACTOR_COLUMN,
    assert_continuity,
    ensure_factor_column,
    find_continuity_violations,
    restore_real_frame,
    restore_real_price,
    to_adjusted_price,
    to_pre_adjusted,
)
from engine.errors import DataQualityError, ValidationError


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _series(prices: list[float], factors: list[float]) -> pd.DataFrame:
    return _frame(
        [
            {
                "code": "600000",
                "date": f"2026-10-{12 + index:02d}",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                FACTOR_COLUMN: factor,
            }
            for index, (price, factor) in enumerate(zip(prices, factors, strict=True))
        ]
    )


# ============================================================ 单值换算
def test_后复权还原真实价() -> None:
    # hfq = 真实价 × 因子  =>  真实价 = hfq / 因子
    assert restore_real_price("22.00", "2.0") == Decimal("11.00")


def test_还原真实价按分四舍五入() -> None:
    assert restore_real_price("10.005", "1.0") == Decimal("10.01")
    assert restore_real_price("10.004", "1.0") == Decimal("10.00")


def test_真实价转后复权() -> None:
    assert to_adjusted_price("11.00", "2.0") == Decimal("22.00")


def test_非法因子被拒而不是当作1() -> None:
    with pytest.raises(ValidationError):
        restore_real_price("10.00", "0")
    with pytest.raises(ValidationError):
        to_adjusted_price("10.00", "-1")


# ============================================================ 批量换算
def test_缺少因子列时报错() -> None:
    frame = _frame([{"code": "600000", "date": "2026-10-12", "close": 10.0}])
    with pytest.raises(DataQualityError):
        ensure_factor_column(frame)
    with pytest.raises(DataQualityError):
        restore_real_frame(frame)


def test_因子非正时报错并给出样本() -> None:
    frame = _series([20.0, 0.0], [2.0, 0.0])
    with pytest.raises(DataQualityError):
        ensure_factor_column(frame)


def test_整表还原本地保留其他列() -> None:
    frame = _series([20.0, 21.0, 22.0], [2.0, 2.0, 2.0])
    real = restore_real_frame(frame)

    assert list(real["close"]) == [10.0, 10.5, 11.0]
    assert FACTOR_COLUMN in real.columns, "因子列必须保留，下游还要用它还原"
    assert list(real["code"]) == ["600000"] * 3


def test_空表还原不报错() -> None:
    frame = pd.DataFrame(columns=["code", "date", "close", FACTOR_COLUMN])
    assert restore_real_frame(frame).empty


def test_前复权最新一日等于真实价() -> None:
    """口径自洽的关键断言：前复权以最新因子为基准，故最新价 == 真实价。"""
    frame = _series([20.0, 21.0, 22.0], [2.0, 2.0, 2.0])
    qfq = to_pre_adjusted(frame)
    real = restore_real_frame(frame)

    assert list(qfq["close"]) == [10.0, 10.5, 11.0]
    assert qfq["close"].iloc[-1] == real["close"].iloc[-1] == 11.0


def test_前复权依赖区间最新因子() -> None:
    """同一只票、不同区间会得到不同的前复权价——这正是它不能用于回测计算的原因。"""
    full = _series([20.0, 21.0, 22.0], [2.0, 2.0, 2.0])
    partial = _series([21.0, 22.0], [2.0, 2.0])

    assert to_pre_adjusted(full)["close"].iloc[0] == 10.0
    assert to_pre_adjusted(partial)["close"].iloc[0] == 10.5


# ============================================================ 连续性探针
def test_正常序列无告警() -> None:
    frame = _series([10.0, 10.5, 11.0], [1.0, 1.0, 1.0])
    assert find_continuity_violations(frame) == []


def test_创业板20个点波动不算异常() -> None:
    frame = _frame(
        [
            {"code": "300750", "date": "2026-10-12", "close": 100.0},
            {"code": "300750", "date": "2026-10-13", "close": 120.0},
        ]
    )
    assert find_continuity_violations(frame) == [], "20cm 品种的正常涨跌不该被误报"


def test_主板超过涨跌停的跳变被识别() -> None:
    frame = _frame(
        [
            {"code": "600000", "date": "2026-10-12", "close": 10.0},
            {"code": "600000", "date": "2026-10-13", "close": 13.0},  # +30%，主板不可能
        ]
    )
    violations = find_continuity_violations(frame)
    assert len(violations) == 1
    assert violations[0].code == "600000"
    assert violations[0].jump_ratio == pytest.approx(0.3)
    assert violations[0].allowed_ratio == pytest.approx(0.12)


def test_缺因子的除权跳变被探测出来() -> None:
    """10 派 3 若不乘因子，会造出约 -23% 的假跌幅（主板不可能）。"""
    frame = _frame(
        [
            {"code": "600000", "date": "2026-10-12", "close": 10.0, FACTOR_COLUMN: 1.0},
            {"code": "600000", "date": "2026-10-13", "close": 7.7, FACTOR_COLUMN: 1.0},
        ]
    )
    violations = find_continuity_violations(frame)
    assert len(violations) == 1
    assert violations[0].jump_ratio == pytest.approx(-0.23)


def test_乘上因子后同一段行情连续() -> None:
    """存储的是后复权价：7.7 × 1.3 = 10.01，相对 10.0 只有 +0.1%，属正常波动。"""
    frame = _frame(
        [
            {"code": "600000", "date": "2026-10-12", "close": 10.0, FACTOR_COLUMN: 1.0},
            {"code": "600000", "date": "2026-10-13", "close": 10.01, FACTOR_COLUMN: 1.3},
        ]
    )
    assert find_continuity_violations(frame) == []


def test_ST股容差更严格() -> None:
    frame = _frame(
        [
            {"code": "600000", "date": "2026-10-12", "close": 10.0, "is_st": True},
            {"code": "600000", "date": "2026-10-13", "close": 11.0, "is_st": True},  # +10% > 5%+2%
        ]
    )
    assert len(find_continuity_violations(frame)) == 1


def test_连续性校验失败抛数据质量异常() -> None:
    frame = _frame(
        [
            {"code": "600000", "date": "2026-10-12", "close": 10.0},
            {"code": "600000", "date": "2026-10-13", "close": 20.0},
        ]
    )
    with pytest.raises(DataQualityError):
        assert_continuity(frame, dataset="daily")
    assert_continuity(_series([10.0, 10.1], [1.0, 1.0]), dataset="daily")  # 正常序列不抛
