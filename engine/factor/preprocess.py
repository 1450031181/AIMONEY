"""因子截面预处理（P2-F1）。

默认流水线：``winsorize → rank → zscore``，按**交易日截面**逐列处理。

为什么默认先 rank 再 zscore：量价因子的截面分布普遍厚尾，
直接 zscore 会让极端值主导组合权重；先转成截面分位再标准化，
对厚尾稳健，且让不同量纲的因子可以横向合成（P2 的多因子合成依赖这一点）。

类目约定（与 P0 一致）：步骤可选 ``winsorize`` / ``rank`` / ``demean`` / ``zscore``；
``demean`` 是市场中性化的最简形式（行业/市值中性化待行业数据接入后扩展）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from engine.errors import ValidationError

_ALLOWED_STEPS = ("winsorize", "rank", "demean", "zscore")


def winsorize_mad(values: pd.Series, *, n_mad: float = 5.0) -> pd.Series:
    """MAD 去极值：以中位数为中心、按 ``1.4826 × MAD`` 度量，超出 n 倍截断。"""
    median = values.median()
    mad = (values - median).abs().median() * 1.4826
    if pd.isna(mad) or mad == 0:
        return values
    return values.clip(median - n_mad * mad, median + n_mad * mad)


#: 浮点误差会让常数序列的标准差是 1e-17 而不是 0，必须按容差判定
_ZERO_TOLERANCE = 1e-12


def zscore(values: pd.Series) -> pd.Series:
    """截面标准化（均值 0、总体标准差 1）。波动为零时返回全 0 而非 inf。"""
    std = values.std(ddof=0)
    if pd.isna(std) or std <= _ZERO_TOLERANCE:
        return values * 0.0
    return (values - values.mean()) / std


def rank_pct(values: pd.Series) -> pd.Series:
    """截面分位（0~1），NaN 保持 NaN。"""
    return values.rank(pct=True, na_option="keep")


def demean(values: pd.Series) -> pd.Series:
    """截面去均值（市场中性化的最简形式）。"""
    return values - values.mean()


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    winsorize_n_mad: float = 5.0
    steps: tuple[str, ...] = ("winsorize", "rank", "zscore")

    def __post_init__(self) -> None:
        unknown = [step for step in self.steps if step not in _ALLOWED_STEPS]
        if unknown:
            raise ValidationError(
                "未知预处理步骤", detail={"unknown": unknown, "allowed": list(_ALLOWED_STEPS)}
            )


def preprocess_cross_section(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """对长表（``date``/``code``/因子列）按日截面执行预处理流水线。

    按 ``date`` 分组处理是硬约束：预处理是**截面**操作，
    跨日期混算会把时间序列信息泄漏进截面分布。
    """
    resolved = config or PreprocessConfig()
    columns = list(factor_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError("待预处理的因子列不存在", detail={"missing": missing})

    result = frame.copy()
    for _, group_index in result.groupby("date", sort=True).groups.items():
        index = pd.Index(group_index)
        for column in columns:
            values = result.loc[index, column]
            for step in resolved.steps:
                values = _apply_step(values, step, resolved)
            result.loc[index, column] = values.to_numpy() if isinstance(values, pd.Series) else values
    return result


def _apply_step(values: pd.Series, step: str, config: PreprocessConfig) -> pd.Series:
    if step == "winsorize":
        return winsorize_mad(values, n_mad=config.winsorize_n_mad)
    if step == "rank":
        return rank_pct(values)
    if step == "demean":
        return demean(values)
    if step == "zscore":
        return zscore(values)
    raise ValidationError("未知预处理步骤", detail={"step": step})  # pragma: no cover —— 构造期已校验


__all__ = [
    "PreprocessConfig",
    "demean",
    "preprocess_cross_section",
    "rank_pct",
    "winsorize_mad",
    "zscore",
]
