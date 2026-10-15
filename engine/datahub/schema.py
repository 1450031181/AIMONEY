"""数据契约（T-04）：数据源 / 存储层 / 适配器共用的唯一列规范。

为什么单独成模块：列名是跨模块的隐式契约，一旦各处硬编码字符串，
「上游改了列名、下游静默拿到 NaN」这类故障极难排查。全部列名收敛到此处，
并在数据入口做一次强制校验。

类型选择说明：批量行情用 **pandas**（akshare/tushare/券商导出原生均为 pandas，
边界层零转换），因子计算层再转 Polars 做向量化加速。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd

from engine.errors import DataQualityError

# ---------------------------------------------------------------- 日线行情
DAILY_PRIMARY_KEY: Final[tuple[str, ...]] = ("code", "date")

#: 必需列：缺任何一列都无法继续，直接判为数据不可用
DAILY_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("code", "date", "open", "high", "low", "close", "volume")

#: 可选列：缺失时按语义默认值处理（adj_factor=1、未停牌、涨跌停未知）
DAILY_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "amount",
    "prev_close",
    "adj_factor",
    "limit_up",
    "limit_down",
    "is_suspended",
)

DAILY_COLUMNS: Final[tuple[str, ...]] = DAILY_REQUIRED_COLUMNS + DAILY_OPTIONAL_COLUMNS

# ---------------------------------------------------------------- 股票基础信息
STOCK_INFO_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("code", "name")
STOCK_INFO_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("is_st", "list_date", "delist_date", "board")
STOCK_INFO_COLUMNS: Final[tuple[str, ...]] = STOCK_INFO_REQUIRED_COLUMNS + STOCK_INFO_OPTIONAL_COLUMNS

#: 需要按复权因子缩放的价格列
PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "prev_close", "limit_up", "limit_down")


def ensure_columns(frame: pd.DataFrame, required: Sequence[str], *, dataset: str) -> None:
    """校验必要列齐全，缺失即抛错（宁可失败，不可让 NaN 静默流向下游）。"""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataQualityError(
            f"{dataset} 缺少必要列",
            detail={"dataset": dataset, "missing": missing, "actual": list(frame.columns)},
        )


def ensure_primary_key_unique(frame: pd.DataFrame, *, dataset: str, keys: Sequence[str] = DAILY_PRIMARY_KEY) -> None:
    """校验主键唯一，重复主键意味着数据源返回了脏数据。"""
    if frame.empty:
        return
    ensure_columns(frame, keys, dataset=dataset)
    duplicated = frame.duplicated(subset=list(keys))
    if duplicated.any():
        sample = frame.loc[duplicated, list(keys)].head(5).to_dict(orient="records")
        raise DataQualityError(
            f"{dataset} 主键重复",
            detail={"dataset": dataset, "keys": list(keys), "duplicated_rows": len(sample), "sample": sample},
        )


__all__ = [
    "DAILY_COLUMNS",
    "DAILY_OPTIONAL_COLUMNS",
    "DAILY_PRIMARY_KEY",
    "DAILY_REQUIRED_COLUMNS",
    "PRICE_COLUMNS",
    "STOCK_INFO_COLUMNS",
    "STOCK_INFO_OPTIONAL_COLUMNS",
    "STOCK_INFO_REQUIRED_COLUMNS",
    "ensure_columns",
    "ensure_primary_key_unique",
]
