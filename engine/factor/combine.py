"""多因子合成（P2-F4）。

合成本身很简单：把若干**已标准化**的因子按权重线性叠加。全部难点在"权重怎么定"：

| 方法 | 权重 | 特点 |
| --- | --- | --- |
| ``equal`` | 各因子等权 | 无估计、无过拟合风险，但会低估真正有效的因子、高估噪声因子 |
| ``ic`` | ``w_i ∝ mean(IC_i)`` | 方向自适应（IC 为负 → 反向使用该因子） |
| ``icir`` | ``w_i ∝ mean(IC_i)/std(IC_i)`` | 兼顾方向与稳定性，默认（IC 高但抖动的因子不该给高权重） |

**头号约束：权重只能由"当时可见"的 IC 估计。**

用全样本 IC 定权重、再回看整段区间，是量化里最典型的未来函数——它等价于
"提前知道了哪个因子在未来更有效"，合成因子的 IC 会被系统性高估，
而回测曲线会漂亮得毫无意义。因此本模块只提供**滚动权重**：
``t`` 日权重仅由 ``[t-window, t-1]`` 的 IC 估计，前 ``window`` 个交易日没有权重
（记为不可用），而不是拿未来的数字补上。

另外，因子之间高度相关时（例如 ``mom_20`` 与 ``rev_5`` 往往是镜像关系），
等权或 IC 加权都会**重复押注同一个信号**并放大该风险敞口，
故提供 ``deduplicate_factors()`` 在合成前按相关性去重。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from engine.errors import ValidationError
from engine.factor.evaluate import information_coefficient

#: 支持的合成方法
SUPPORTED_METHODS = ("equal", "ic", "icir")

#: 相关性去重默认阈值：|ρ| 超过它视为"同一信号的两种写法"
DEFAULT_CORRELATION_THRESHOLD = 0.7

_ZERO_TOLERANCE = 1e-12


# ============================================================ 权重
def normalize_weights(raw: Mapping[str, float]) -> dict[str, float]:
    """归一化到 ``Σ|w| = 1``。

    用绝对值之和而非代数和：因子可以做反向使用（负权重），
    若按代数和归一化，一个 +0.5 与一个 −0.5 会互相抵消到"总暴露为零"，
    权重就失去意义了。
    """
    total = float(sum(abs(value) for value in raw.values()))
    if total <= _ZERO_TOLERANCE:
        return dict.fromkeys(raw, 0.0)
    return {name: float(value) / total for name, value in raw.items()}


def weights_from_ic(
    ic_mean: Mapping[str, float],
    ic_std: Mapping[str, float] | None = None,
    *,
    method: str = "icir",
    min_abs_ic: float = 0.0,
) -> dict[str, float]:
    """由 IC 统计量计算权重（纯函数，不含任何时间信息）。

    ``min_abs_ic`` 用于过滤噪声因子：|IC| 低于该阈值的因子直接给 0 权重，
    而不是给一个"很小但不为零"的权重——后者会让因子池越加越大、噪声累积。
    """
    if method not in SUPPORTED_METHODS:
        raise ValidationError("不支持的合成方法", detail={"method": method, "supported": list(SUPPORTED_METHODS)})
    if method == "equal":
        return normalize_weights(dict.fromkeys(ic_mean, 1.0))

    deviations = ic_std or {}
    raw: dict[str, float] = {}
    for name, mean in ic_mean.items():
        if not np.isfinite(mean) or abs(mean) < min_abs_ic:
            raw[name] = 0.0
            continue
        if method == "ic":
            raw[name] = float(mean)
            continue
        std = float(deviations.get(name, 0.0))
        # ICIR：IC 波动为零说明"稳定"到无从度量（或样本太少），退回 IC 本身避免除零
        raw[name] = float(mean) if std <= _ZERO_TOLERANCE else float(mean) / std
    return normalize_weights(raw)


def rolling_ic_weights(
    merged: pd.DataFrame,
    factors: Sequence[str],
    *,
    window: int,
    method: str = "icir",
    min_cross_section: int = 5,
    min_abs_ic: float = 0.0,
) -> pd.DataFrame:
    """逐日滚动估计权重，返回 ``date × factor`` 矩阵（前 ``window`` 个截面为 NaN）。

    **严格使用 t 之前的信息**：``t`` 日权重来自 ``t`` 之前最近 ``window`` 个有效 IC，
    不含当日、更不含未来。前 ``window`` 天权重为 NaN —— 调用方应把这几天的合成分数
    视为不可用，而不是用全样本权重"补"上去。
    """
    if window <= 0:
        raise ValidationError("权重估计窗口必须为正", detail={"window": window})
    if method not in SUPPORTED_METHODS:
        raise ValidationError("不支持的合成方法", detail={"method": method, "supported": list(SUPPORTED_METHODS)})
    if not factors:
        raise ValidationError("至少指定一个因子")

    ic_frame = pd.DataFrame(
        {
            name: information_coefficient(merged, factor_column=name, min_cross_section=min_cross_section)
            for name in factors
        },
        dtype=float,
    ).sort_index()
    if ic_frame.empty:
        return pd.DataFrame(index=ic_frame.index, columns=list(factors), dtype=float)

    weights: dict[pd.Timestamp, dict[str, float]] = {}
    for position, day in enumerate(ic_frame.index):
        history = ic_frame.iloc[:position].tail(window)  # 严格 t 之前
        if len(history) < window:
            continue  # 预热不足：宁可不给权重，也不用不足的样本凑
        means = {name: float(history[name].mean(skipna=True)) for name in factors}
        stds = {name: float(history[name].std(ddof=1, skipna=True)) for name in factors}
        weights[pd.Timestamp(str(day))] = weights_from_ic(
            means, stds, method=method, min_abs_ic=min_abs_ic
        )

    if not weights:
        return pd.DataFrame(index=ic_frame.index, columns=list(factors), dtype=float)
    result = pd.DataFrame.from_dict(weights, orient="index")
    return result.reindex(columns=list(factors)).reindex(ic_frame.index)


# ============================================================ 去重
def deduplicate_factors(
    correlation: pd.DataFrame,
    strength: Mapping[str, float],
    *,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> tuple[str, ...]:
    """按 ``|strength|`` 降序贪心选因子，与已选因子相关性超阈值者剔除。

    ``strength`` 通常是各因子的 |IC|。被剔除的不代表是坏因子，
    只代表它与更强的因子在表达同一件事——留两个只会把同一风险敞口放大一倍。
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValidationError("相关性阈值应在 [0, 1] 内", detail={"threshold": threshold})
    ordered = sorted(strength, key=lambda name: abs(float(strength[name])), reverse=True)
    kept: list[str] = []
    for name in ordered:
        if name not in correlation.columns:
            continue
        # 转成 dict 再取值：DataFrame.loc[name, other] 的返回类型被存根标注得过宽
        row = {str(column): float(value) for column, value in correlation.loc[name].to_dict().items()}
        if all(abs(row[other]) <= threshold for other in kept if other in row):
            kept.append(name)
    return tuple(kept)


# ============================================================ 合成与选股
def composite_score(
    frame: pd.DataFrame,
    factors: Sequence[str],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """按当日权重把因子线性合成，返回 ``(date, code, score)``。

    权重缺失（预热期）的截面直接跳过——那一天就是没有合成分数，
    不能用"全局权重"或"隔日权重"顶上，否则等于把未来信息提前使用。
    """
    names = list(factors)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValidationError("缺少待合成的因子列", detail={"missing": missing})

    rows: list[pd.DataFrame] = []
    for day, group in frame.groupby("date", sort=True):
        if day not in weights.index:
            continue
        row: dict[str, float] = {str(name): float(value) for name, value in weights.loc[day].to_dict().items()}
        active = {
            name: value
            for name, value in row.items()
            if name in names and pd.notna(value) and abs(value) > 0.0
        }
        if not active:
            continue
        values = group[names].to_numpy(dtype=float)
        weight_vector = np.array([active.get(name, 0.0) for name in names], dtype=float)
        score = values @ weight_vector
        rows.append(
            pd.DataFrame(
                {
                    "date": group["date"].to_numpy(),
                    "code": group["code"].to_numpy(),
                    "score": score,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["date", "code", "score"])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)


def select_top(scores: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """每个截面取分数最高的 ``top_k`` 只，返回 ``(date, code, score, rank, weight)``。

    ``weight`` 是等权配置（``1/top_k``）：这一步只产**名单**，
    真正的组合权重与风控留给组合层，避免因子层越权决定资金分配。
    """
    if top_k <= 0:
        raise ValidationError("top_k 必须为正", detail={"top_k": top_k})
    if scores.empty:
        return pd.DataFrame(columns=["date", "code", "score", "rank", "weight"])

    ranked: list[pd.DataFrame] = []
    for _, group in scores.groupby("date", sort=True):
        sample = group.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_k).copy()
        if sample.empty:
            continue
        sample["rank"] = range(1, len(sample) + 1)
        sample["weight"] = 1.0 / len(sample)
        ranked.append(sample)
    if not ranked:
        return pd.DataFrame(columns=["date", "code", "score", "rank", "weight"])
    return pd.concat(ranked, ignore_index=True).loc[:, ["date", "code", "score", "rank", "weight"]]


__all__ = [
    "DEFAULT_CORRELATION_THRESHOLD",
    "SUPPORTED_METHODS",
    "composite_score",
    "deduplicate_factors",
    "normalize_weights",
    "rolling_ic_weights",
    "select_top",
    "weights_from_ic",
]
