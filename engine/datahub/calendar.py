"""交易日历实现（T-05）。

为什么不用自然日推算：A 股每年有 10 天以上的节假日 + 调休，用「工作日近似」会让
回测与实盘在节前后错位（会导致持仓天数、调仓频率、手续费全部算错）。
因此日历必须来自真实数据（``akshare.tool_trade_date_hist_sina`` 等），
本模块只负责「给定交易日集合后的一切推算」。

本模块实现 ``engine.ports.TradingCalendar`` 端口。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

from engine.errors import ValidationError

#: 交易日历默认名称（A 股）
DEFAULT_CALENDAR_NAME: Final[str] = "cn_a_share"


@dataclass(frozen=True, slots=True)
class TradingCalendarImpl:
    """基于交易日集合的日历。

    ``days`` 必须严格升序且无重复；用 ``from_dates()`` 可从任意可迭代对象构造（自动排序去重）。
    构造期强校验而非运行期兜底，避免"日历少了一天"在回测里变成"少交易一次"这种隐性错误。
    """

    days: tuple[date, ...]
    name: str = DEFAULT_CALENDAR_NAME

    def __post_init__(self) -> None:
        if not self.days:
            raise ValidationError("交易日历不能为空", detail={"name": self.name})
        if len(set(self.days)) != len(self.days):
            raise ValidationError("交易日历存在重复日期", detail={"name": self.name, "count": len(self.days)})
        # strict=False 是刻意的：这里比较的是错位一位的相邻元素，长度天然不同
        for previous, current in zip(self.days, self.days[1:], strict=False):
            if current <= previous:
                raise ValidationError(
                    "交易日历必须严格升序",
                    detail={"name": self.name, "previous": previous.isoformat(), "current": current.isoformat()},
                )

    # -------------------------------------------------- 构造
    @classmethod
    def from_dates(cls, dates: Iterable[date], name: str = DEFAULT_CALENDAR_NAME) -> TradingCalendarImpl:
        """从任意日期集合构造（自动去重升序）。"""
        return cls(days=tuple(sorted(set(dates))), name=name)

    # -------------------------------------------------- 基础信息
    def sessions(self) -> tuple[date, ...]:
        return self.days

    @property
    def first_day(self) -> date:
        return self.days[0]

    @property
    def last_day(self) -> date:
        return self.days[-1]

    def count(self) -> int:
        return len(self.days)

    def _index(self, day: date) -> int | None:
        position = bisect_left(self.days, day)
        if position < len(self.days) and self.days[position] == day:
            return position
        return None

    def is_trade_day(self, day: date) -> bool:
        return self._index(day) is not None

    # -------------------------------------------------- 区间查询
    def between(self, start: date, end: date, *, inclusive: bool = True) -> tuple[date, ...]:
        if start > end:
            raise ValidationError(
                "区间起点晚于终点", detail={"start": start.isoformat(), "end": end.isoformat()}
            )
        if inclusive:
            low, high = bisect_left(self.days, start), bisect_right(self.days, end)
        else:
            low, high = bisect_right(self.days, start), bisect_left(self.days, end)
        return self.days[low:high]

    # -------------------------------------------------- 相对推算
    def prev_trade_day(self, day: date, *, include_self: bool = False) -> date:
        """前一个交易日。``include_self=True`` 且 ``day`` 本身是交易日时返回 ``day``。"""
        if include_self and self.is_trade_day(day):
            return day
        position = bisect_left(self.days, day)
        if position == 0:
            raise ValidationError(
                "早于日历范围，无法推算前一交易日",
                detail={"day": day.isoformat(), "first_day": self.first_day.isoformat()},
            )
        return self.days[position - 1]

    def next_trade_day(self, day: date, *, include_self: bool = False) -> date:
        """后一个交易日。``include_self=True`` 且 ``day`` 本身是交易日时返回 ``day``。"""
        if include_self and self.is_trade_day(day):
            return day
        position = bisect_right(self.days, day)
        if position >= len(self.days):
            raise ValidationError(
                "晚于日历范围，无法推算后一交易日",
                detail={"day": day.isoformat(), "last_day": self.last_day.isoformat()},
            )
        return self.days[position]

    def shift(self, day: date, n: int) -> date:
        """在交易日序列上平移 n 个交易日。起点必须是交易日（否则语义不明确）。"""
        position = self._index(day)
        if position is None:
            raise ValidationError(
                "平移的起点不是交易日", detail={"day": day.isoformat(), "calendar": self.name}
            )
        target = position + n
        if not 0 <= target < len(self.days):
            raise ValidationError(
                "平移后超出日历范围",
                detail={"day": day.isoformat(), "n": n, "range": [self.first_day.isoformat(), self.last_day.isoformat()]},
            )
        return self.days[target]


def weekday_calendar(start: date, end: date, *, name: str = "weekday-approx") -> TradingCalendarImpl:
    """按「周一至周五」生成的近似日历。

    **仅供开发与单元测试**：它不认识法定节假日与调休，绝不可用于实盘或正式回测。
    """
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return TradingCalendarImpl(days=tuple(days), name=name)


def warmup_start(calendar: TradingCalendarImpl, start: date, sessions: int) -> date:
    """把起点前推 ``sessions`` 个交易日（因子预热用）。

    日历不足以覆盖时**退化为最早交易日**而不是报错：此时窗口开头的因子值会是 NaN，
    下游会自动跳过这些截面——这是诚实降级；报错则会让"数据刚够一天"的场景完全不可用，
    而按短窗口硬算会产出系统性失真的值（最糟的选择）。
    """
    if sessions <= 0:
        return start
    try:
        anchor = calendar.next_trade_day(start, include_self=True)
        return calendar.shift(anchor, -sessions)
    except ValidationError:
        return calendar.first_day


__all__ = ["DEFAULT_CALENDAR_NAME", "TradingCalendarImpl", "weekday_calendar"]
