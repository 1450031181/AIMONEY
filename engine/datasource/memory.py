"""内存数据源（T-04 测试替身 + 本地演练用）。

用途：
- 单元测试与引擎正确性回归（可精确构造"一字涨停""停牌""跨除权日"等边界场景）；
- 离线演练：没有网络/没有行情权限时也能把回测链路跑通。

设计取舍：不复权数据 + ``adj_factor`` 列分开存储，``daily(adj="post")`` 在读取时缩放。
``adj="pre"`` 明确报错而不是"随便给一个"——前复权需要"以哪一天为基准"这个额外语义，
区间查询无法自洽表达，统一由展示层处理（见设计文档 5.4）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from engine.datahub.calendar import TradingCalendarImpl
from engine.datahub.schema import (
    DAILY_COLUMNS,
    DAILY_REQUIRED_COLUMNS,
    PRICE_COLUMNS,
    ensure_columns,
    ensure_primary_key_unique,
)
from engine.datahub.universe import UniverseHistory
from engine.errors import ValidationError
from engine.ports import AdjType

_STOCK_INFO_COLUMNS = ("code", "name", "list_date", "delist_date", "is_st", "board")
_ADJ_FACTOR_COLUMNS = ("code", "date", "adj_factor")


@dataclass
class InMemoryDataSource:
    """可编程的内存数据源，行为与真实 ``DataSource`` 契约保持一致。"""

    calendar: TradingCalendarImpl
    name: str = "memory"
    #: 指数成分历史（与 AkShare 源共用同一套「按生效日解析」逻辑）
    universe_history: UniverseHistory = field(default_factory=UniverseHistory)
    _daily: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _stock_info: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=list(_STOCK_INFO_COLUMNS)), repr=False
    )

    # -------------------------------------------------- 装载数据
    def add_daily(self, frame: pd.DataFrame) -> None:
        """装载日线行情（可多次调用，同一 code+date 以最后一次为准）。"""
        ensure_columns(frame, DAILY_REQUIRED_COLUMNS, dataset="daily")
        normalized = frame.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        if "adj_factor" not in normalized.columns:
            normalized["adj_factor"] = 1.0
        if "is_suspended" not in normalized.columns:
            normalized["is_suspended"] = False
        for code, group in normalized.groupby("code", sort=True):
            stored = (
                group.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            )
            self._daily[str(code)] = stored

    def add_stock_info(self, frame: pd.DataFrame) -> None:
        ensure_columns(frame, ("code",), dataset="stock_info")
        self._stock_info = frame.copy().reset_index(drop=True)

    def add_universe(self, index_code: str, effective_date: date, codes: Sequence[str]) -> None:
        """登记指数成分；同一指数多次登记即构成成分变更历史，按生效日生效。"""
        self.universe_history.add(index_code, effective_date, codes)

    # -------------------------------------------------- DataSource 契约
    def trading_calendar(self) -> TradingCalendarImpl:
        return self.calendar

    def daily(
        self,
        codes: Sequence[str],
        start: date,
        end: date,
        *,
        adj: AdjType = "post",
    ) -> pd.DataFrame:
        if adj == "pre":
            raise ValidationError(
                "前复权属于展示口径，需指定基准日，数据源不提供",
                detail={"hint": "请使用 adj='post' 计算，展示层再转前复权（设计文档 5.4）"},
            )
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for code in codes:
            stored = self._daily.get(code)
            if stored is None:
                continue
            subset = stored.loc[(stored["date"] >= start_ts) & (stored["date"] <= end_ts)]
            if subset.empty:
                continue
            frames.append(self._apply_adjust(subset, adj))
        if not frames:
            return pd.DataFrame(columns=list(DAILY_COLUMNS))
        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values(["date", "code"]).reset_index(drop=True)

    def adjust_factors(self, codes: Sequence[str], start: date, end: date) -> pd.DataFrame:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        rows: list[dict[str, object]] = []
        for code in codes:
            stored = self._daily.get(code)
            if stored is None:
                continue
            subset = stored.loc[(stored["date"] >= start_ts) & (stored["date"] <= end_ts), ["date", "adj_factor"]]
            rows.extend({"code": code, "date": record.date, "adj_factor": record.adj_factor} for record in subset.itertuples(index=False))
        return pd.DataFrame(rows, columns=list(_ADJ_FACTOR_COLUMNS))

    def universe(self, index_code: str, on: date) -> tuple[str, ...]:
        return self.universe_history.resolve(index_code, on)

    def index_members(self, index_code: str) -> tuple[str, ...]:
        """内存源把「最近一次登记的快照」当作当前成分。"""
        return self.universe_history.latest(index_code)

    def stock_info(self) -> pd.DataFrame:
        return self._stock_info.copy()

    # -------------------------------------------------- 内部工具
    @staticmethod
    def _apply_adjust(frame: pd.DataFrame, adj: AdjType) -> pd.DataFrame:
        result = frame.copy()
        if adj == "none":
            return result
        factor = result["adj_factor"].astype(float)
        for column in PRICE_COLUMNS:
            if column in result.columns:
                result[column] = result[column].astype(float) * factor
        return result

    # -------------------------------------------------- 一致性自检（测试与演练用）
    def assert_no_duplicated_keys(self, codes: Sequence[str]) -> None:
        for code in codes:
            stored = self._daily.get(code)
            if stored is not None:
                ensure_primary_key_unique(stored, dataset=f"daily[{code}]")


__all__ = ["InMemoryDataSource"]
