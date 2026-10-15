"""T-05 交易日历测试：区间查询、前后推算、平移、越界必须报错。"""

from __future__ import annotations

from datetime import date

import pytest

from engine.datahub.calendar import TradingCalendarImpl, weekday_calendar
from engine.errors import ValidationError

DAYS = (
    date(2026, 9, 30),
    date(2026, 10, 9),
    date(2026, 10, 12),
    date(2026, 10, 13),
    date(2026, 10, 14),
)


@pytest.fixture
def calendar() -> TradingCalendarImpl:
    return TradingCalendarImpl(days=DAYS, name="test")


def test_从集合构造会自动排序去重() -> None:
    calendar = TradingCalendarImpl.from_dates(
        [date(2026, 10, 14), date(2026, 10, 9), date(2026, 10, 14), date(2026, 10, 12)]
    )
    assert calendar.sessions() == (date(2026, 10, 9), date(2026, 10, 12), date(2026, 10, 14))
    assert calendar.first_day == date(2026, 10, 9)
    assert calendar.last_day == date(2026, 10, 14)


def test_空日历被拒() -> None:
    with pytest.raises(ValidationError):
        TradingCalendarImpl(days=())


def test_未排序日历被拒() -> None:
    with pytest.raises(ValidationError):
        TradingCalendarImpl(days=(date(2026, 10, 14), date(2026, 10, 9)))


def test_重复交易日被拒() -> None:
    with pytest.raises(ValidationError):
        TradingCalendarImpl(days=(date(2026, 10, 9), date(2026, 10, 9)))


def test_是否交易日(calendar: TradingCalendarImpl) -> None:
    assert calendar.is_trade_day(date(2026, 10, 12))
    assert not calendar.is_trade_day(date(2026, 10, 10))  # 周六
    assert not calendar.is_trade_day(date(2026, 10, 1))  # 国庆假期
    assert calendar.count() == 5


def test_前一交易日(calendar: TradingCalendarImpl) -> None:
    # 10-01 是假期，应回溯到 09-30
    assert calendar.prev_trade_day(date(2026, 10, 1)) == date(2026, 9, 30)
    # 10-12 是周一，前一交易日是 10-09（跨周末 + 节后首个交易日）
    assert calendar.prev_trade_day(date(2026, 10, 12)) == date(2026, 10, 9)
    assert calendar.prev_trade_day(date(2026, 10, 12), include_self=True) == date(2026, 10, 12)


def test_后一交易日(calendar: TradingCalendarImpl) -> None:
    assert calendar.next_trade_day(date(2026, 9, 30)) == date(2026, 10, 9)
    assert calendar.next_trade_day(date(2026, 10, 10)) == date(2026, 10, 12)
    assert calendar.next_trade_day(date(2026, 10, 12), include_self=True) == date(2026, 10, 12)


def test_区间查询_闭区间与开区间(calendar: TradingCalendarImpl) -> None:
    assert calendar.between(date(2026, 10, 9), date(2026, 10, 13)) == (
        date(2026, 10, 9),
        date(2026, 10, 12),
        date(2026, 10, 13),
    )
    assert calendar.between(date(2026, 10, 9), date(2026, 10, 13), inclusive=False) == (date(2026, 10, 12),)
    # 区间内无交易日时返回空（不是报错）
    assert calendar.between(date(2026, 10, 1), date(2026, 10, 8)) == ()


def test_区间起点晚于终点报错(calendar: TradingCalendarImpl) -> None:
    with pytest.raises(ValidationError):
        calendar.between(date(2026, 10, 14), date(2026, 10, 9))


def test_交易日平移(calendar: TradingCalendarImpl) -> None:
    assert calendar.shift(date(2026, 10, 9), 1) == date(2026, 10, 12)
    assert calendar.shift(date(2026, 10, 12), -1) == date(2026, 10, 9)
    assert calendar.shift(date(2026, 10, 12), 0) == date(2026, 10, 12)


def test_平移起点必须是交易日(calendar: TradingCalendarImpl) -> None:
    with pytest.raises(ValidationError):
        calendar.shift(date(2026, 10, 10), 1)


def test_越界必须报错而非静默夹取(calendar: TradingCalendarImpl) -> None:
    with pytest.raises(ValidationError):
        calendar.prev_trade_day(date(2026, 9, 30))
    with pytest.raises(ValidationError):
        calendar.next_trade_day(date(2026, 10, 14))
    with pytest.raises(ValidationError):
        calendar.shift(date(2026, 10, 14), 1)


def test_工作日近似日历仅覆盖周一到周五() -> None:
    calendar = weekday_calendar(date(2026, 10, 12), date(2026, 10, 18))
    assert calendar.sessions() == (
        date(2026, 10, 12),
        date(2026, 10, 13),
        date(2026, 10, 14),
        date(2026, 10, 15),
        date(2026, 10, 16),
    )
    assert not calendar.is_trade_day(date(2026, 10, 17))
