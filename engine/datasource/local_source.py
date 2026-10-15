"""本地数据源（P1-B7）：从 Parquet 列存读取，回测与离线分析的取数入口。

复权口径约定（与设计文档 5.4 一致）：
- 存储层保存的是**后复权价 + 真实复权因子**，因此 ``adj="post"``（回测默认）原样返回；
- ``adj="none"`` 用 ``后复权价 / 因子`` 还原真实价（对账、展示盘口用）；
- ``adj="pre"`` 与内存源一致：明确拒绝——前复权需要指定基准日，区间查询无法自洽表达。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from engine.datahub.adjust import restore_real_frame
from engine.datahub.calendar import TradingCalendarImpl
from engine.datahub.reference import ReferenceStore
from engine.datahub.schema import DAILY_COLUMNS
from engine.datahub.store import ParquetStore
from engine.errors import DataNotReadyError, ValidationError
from engine.ports import AdjType


@dataclass
class LocalDataSource:
    """本地列存数据源（实现 ``engine.ports.DataSource`` 的只读部分）。"""

    store: ParquetStore
    dataset: str = "daily"
    name: str = "local"

    def daily(
        self,
        codes: Sequence[str],
        start: date,
        end: date,
        *,
        adj: AdjType = "post",
    ) -> pd.DataFrame:
        frame = self.store.read(self.dataset, codes=list(codes), start=start, end=end)
        if frame.empty:
            return pd.DataFrame(columns=list(DAILY_COLUMNS))
        if adj == "post":
            return frame
        if adj == "pre":
            raise ValidationError(
                "前复权属于展示口径，需指定基准日，本地数据源不提供",
                detail={"hint": "回测请使用 adj='post'，展示层再转前复权"},
            )
        return restore_real_frame(frame, validate=True)

    def trading_calendar(self) -> TradingCalendarImpl:
        """由本地数据推导交易日（数据集覆盖到哪，日历就到哪）。"""
        frame = self.store.read(self.dataset, columns=["date"])
        if frame.empty:
            raise DataNotReadyError(
                "本地无行情数据，无法构建交易日历",
                detail={"dataset": self.dataset, "hint": "先执行 aimoney-data sync"},
            )
        days = sorted({pd.Timestamp(value).date() for value in frame["date"].tolist()})
        return TradingCalendarImpl.from_dates(days, name=self.name)

    def adjust_factors(self, codes: Sequence[str], start: date, end: date) -> pd.DataFrame:
        return self.store.read(
            self.dataset, codes=list(codes), start=start, end=end, columns=["code", "date", "adj_factor"]
        )

    def stock_info(self) -> pd.DataFrame:
        """股票基础信息来自参考数据目录（由 sync 顺带落盘）。"""
        return ReferenceStore(self.store.root.parent / "reference").load_stock_info()


__all__ = ["LocalDataSource"]
