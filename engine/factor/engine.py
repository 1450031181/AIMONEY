"""因子计算引擎（P2-F1）。

设计取舍：**逐标的独立计算**，而不是全市场矩阵一次性算。
- 正确性：因子只允许看单标的自身历史，groupby 天然隔离，跨标的泄漏结构上不可能；
- 性能：向量化发生在"单标的时间序列"维度上（rolling/shift 都是 C 实现），
  全市场 5000 标的 × 20 因子的瓶颈在 groupby 调度而非数值计算；
  真正需要全矩阵加速的是 P2 后期的截面回归与中性化，届时再上 Polars。
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from engine.datahub.schema import DAILY_REQUIRED_COLUMNS
from engine.errors import ValidationError
from engine.factor.base import get_factor


def compute_factors(frame: pd.DataFrame, factor_names: Sequence[str]) -> pd.DataFrame:
    """计算因子。输入长表（``code``/``date``/OHLCV），输出长表（``date``/``code``/各因子列）。

    历史不足的位置为 NaN（由各因子定义的 ``window`` 决定），
    下游评价与预处理必须显式处理 NaN，而不是默默丢行。
    """
    if not factor_names:
        raise ValidationError("未指定要计算的因子")
    if len(set(factor_names)) != len(factor_names):
        raise ValidationError("因子名重复（同名列会互相覆盖）", detail={"factors": list(factor_names)})
    definitions = [get_factor(name) for name in factor_names]

    required = ("code", *DAILY_REQUIRED_COLUMNS[1:])  # code/date/OHLCV
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValidationError("因子计算输入缺少必要列", detail={"missing": missing})

    parts: list[pd.DataFrame] = []
    for code, group in frame.groupby("code", sort=True):
        series_frame = group.sort_values("date").set_index("date")
        data: dict[str, object] = {"code": code, "date": series_frame.index.to_numpy()}
        for definition in definitions:
            values = definition.func(series_frame)
            if not isinstance(values, pd.Series):
                raise ValidationError(
                    "因子计算函数必须返回 Series",
                    detail={"factor": definition.name, "actual": type(values).__name__},
                )
            if len(values) != len(series_frame):
                raise ValidationError(
                    "因子返回长度与输入不一致（禁止丢行或重排）",
                    detail={"factor": definition.name, "expected": len(series_frame), "actual": len(values)},
                )
            data[definition.name] = values.to_numpy()
        parts.append(pd.DataFrame(data))

    result = pd.concat(parts, ignore_index=True)
    columns = ["date", "code", *factor_names]
    return result.loc[:, columns].sort_values(["date", "code"]).reset_index(drop=True)


__all__ = ["compute_factors"]
