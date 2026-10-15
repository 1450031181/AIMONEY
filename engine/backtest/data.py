"""回测数据提供层（P1-B3）。

职责边界：把批量行情 DataFrame 变成"按日取 bar、按窗口取历史"的只读访问器，
**所有历史查询强制以 ``end`` 为界**——防未来函数在这里实现一次，
Context（P1-B4）与引擎都只能经由它拿数据，无法绕过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Final

import pandas as pd

from engine.datahub.adjust import restore_real_price
from engine.datahub.schema import DAILY_REQUIRED_COLUMNS, ensure_columns
from engine.errors import LookaheadError, ValidationError
from engine.types import Bar

_HISTORY_FIELDS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adj_factor",
)


@dataclass
class DataFrameBarProvider:
    """基于行情 DataFrame 的数据访问器（同步任务产出的 daily 数据可直接传入）。"""

    frame: pd.DataFrame
    _by_day: dict[date, dict[str, Bar]] = field(default_factory=dict, init=False, repr=False)
    _days: list[date] = field(default_factory=list, init=False, repr=False)
    _by_code: dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_columns(self.frame, DAILY_REQUIRED_COLUMNS, dataset="backtest bars")
        normalized = self.frame.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        normalized = normalized.sort_values(["date", "code"]).reset_index(drop=True)

        for day, group in normalized.groupby("date", sort=True):
            self._by_day[_as_date(day)] = {
                str(row["code"]): self._to_bar(row, _as_date(day)) for _, row in group.iterrows()
            }
        self._days = sorted(self._by_day)
        if not self._days:
            raise ValidationError("行情数据为空，无法构建回测数据源")
        self._by_code = {
            str(code): group.sort_values("date").reset_index(drop=True)
            for code, group in normalized.groupby("code", sort=True)
        }

    # -------------------------------------------------- 访问接口
    def trading_days(self) -> list[date]:
        return list(self._days)

    def bars_on(self, day: date) -> dict[str, Bar]:
        """当日全部标的行情。查无此日返回空映射（调用方决定是否跳过）。"""
        return dict(self._by_day.get(day, {}))

    def codes_on(self, day: date) -> tuple[str, ...]:
        return tuple(self.bars_on(day))

    def history(
        self,
        code: str,
        *,
        field: str = "close",
        end: date,
        window: int,
    ) -> list[Decimal]:
        """截至 ``end``（含）的最近 ``window`` 个值，升序。

        **越界即报错**：``end`` 晚于当前引擎时点的问题应在 Context 层被拦截；
        这里再兜底一层，保证任何直接访问都拿不到未来数据。
        """
        if field not in _HISTORY_FIELDS:
            raise ValidationError("不支持的历史字段", detail={"field": field, "supported": list(_HISTORY_FIELDS)})
        if window <= 0:
            raise ValidationError("window 必须为正", detail={"window": window})
        frame = self._by_code.get(code)
        if frame is None:
            return []
        cutoff = pd.Timestamp(end)
        subset = frame.loc[frame["date"] <= cutoff].tail(window)
        return [Decimal(str(value)) for value in subset[field].tolist()]

    def last_price(self, code: str, *, end: date) -> Decimal | None:
        values = self.history(code, field="close", end=end, window=1)
        return values[-1] if values else None

    # -------------------------------------------------- 内部
    @staticmethod
    def _to_bar(row: pd.Series, day: date) -> Bar:
        """把存储行转成 ``Bar``。

        **价格一律还原成真实价**（``adj_factor = 后复权价 / 真实价``）。原因是 ``Bar``
        是**执行口径**：撮合按它定价、资金按人民币结算、涨跌停与限价也按它比较。
        若直接放后复权价（存储口径），在 ``adj_factor ≠ 1``（存在分红送转）的标的上会
        **静默**出现两类错误：

        1. 用真实人民币按放大后的价格成交 → 实际仓位被系统性缩小 ``1/F``；
        2. 上游给的涨跌停/前收是真实价，与后复权成交价比大小必然错乱。

        需要**计算口径**（后复权）的地方走 ``history()`` / ``Context.history(field=...)``，
        那条通道默认返回后复权序列（收益、均线、动量必须用它，否则除权日出假跳变）。
        """
        factor = Decimal(str(row.get("adj_factor", 1) or 1))

        def optional(column: str) -> Decimal | None:
            if column not in row.index or pd.isna(row[column]):
                return None
            return Decimal(str(row[column]))

        def real(column: str) -> Decimal:
            return restore_real_price(Decimal(str(row[column])), factor)

        suspended = bool(row["is_suspended"]) if "is_suspended" in row.index and pd.notna(row["is_suspended"]) else False
        return Bar(
            code=str(row["code"]),
            dt=pd.Timestamp(day),
            open=real("open"),
            high=real("high"),
            low=real("low"),
            close=real("close"),
            volume=int(row["volume"]),
            amount=Decimal(str(row.get("amount", 0))),
            limit_up=optional("limit_up"),
            limit_down=optional("limit_down"),
            is_suspended=suspended,
            adj_factor=factor,
        )


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def check_history_window(provider: DataFrameBarProvider, code: str, *, end: date, window: int) -> None:
    """显式越界检查工具（供测试与 Context 使用）。"""
    history = provider.history(code, end=end, window=window)
    if not history:
        raise LookaheadError(f"{code} 在 {end} 之前没有任何历史数据", detail={"code": code, "end": end.isoformat()})


__all__ = ["DataFrameBarProvider", "check_history_window"]
