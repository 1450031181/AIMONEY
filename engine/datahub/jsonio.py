"""JSON 安全化工具（T-11 提取）。

为什么需要它：本项目的记录类对象（任务日志、质量报告、成分快照）大量承载
pandas / numpy 的标量——从 Parquet 读出的 ``date`` 是 ``Timestamp``、统计值是
``np.float64``、金额是 ``Decimal``。直接 ``json.dumps`` 会在**写入报告的那一刻**
抛 ``TypeError``，而报告恰恰是排查故障的唯一依据：出错即失去线索。

因此统一在写盘前经过 ``to_jsonable`` 归一化，而不是各处零散地 ``default=str``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from datetime import date, datetime
from decimal import Decimal
from typing import Any

__all__ = ["to_jsonable"]


def to_jsonable(value: Any) -> Any:
    """把任意值递归转换成 ``json.dumps`` 可序列化的形式。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        # float('nan')/inf 不是合法 JSON，统一转 None
        return None if value != value or value in (float("inf"), float("-inf")) else value
    if isinstance(value, Decimal):
        return float(value)
    # pandas.Timestamp 是 datetime 的子类，故这一分支同时覆盖两者
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (Sequence, Set)):
        return [to_jsonable(item) for item in value]
    # numpy / pandas 标量（np.int64 / np.float64 / np.bool_ ...）
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return to_jsonable(item_method())
        except (ValueError, TypeError):  # pragma: no cover —— 极端 dtype 兜底
            pass
    return str(value)
