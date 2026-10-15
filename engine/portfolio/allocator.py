"""权重分配方案（P3-1）。

输入是**已选中的标的及其得分**（得分来自因子合成层，已截面标准化），输出是目标权重。

| 方案 | 权重 | 适用与代价 |
| --- | --- | --- |
| ``equal`` | ``1/N`` | 最稳的基线；完全不看得分，弱的候选也被给足仓位 |
| ``score`` | ``∝ 正得分`` | 让更强的候选拿更多；**负分标的直接排除**（A 股不能做空） |
| ``rank`` | ``∝ (N−排名+1)`` | 只用序关系，对异常值最稳健；得分差距大时信息利用不充分 |
| ``inverse_vol`` | ``∝ 1/波动`` | 风险平价雏形，让低波动标的承担更多权重；不看方向，必须配合选股使用 |

三条共同约定：
1. **权重归一化到 Σ=1**，再由调用方决定留多少现金（见 ``constraints``）；
2. ``score`` 方案在全部得分为负时返回**空组合**——这本身就是"当前没有值得持有的标的"的信号，
   强行挑"最不差"的标的买入，是把负 alpha 当正 alpha 用；
3. 任一方案都不做杠杆与做空（权重恒 ≥ 0，Σ ≤ 1）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from engine.errors import ValidationError

SUPPORTED_SCHEMES = ("equal", "score", "rank", "inverse_vol")

_ZERO_TOLERANCE = 1e-12


def allocate(
    scores: Mapping[str, float],
    *,
    scheme: str = "equal",
    volatilities: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """按方案把得分转成目标权重（归一化到 Σ=1；无可用标的时返回空字典）。"""
    if scheme not in SUPPORTED_SCHEMES:
        raise ValidationError("不支持的分配方案", detail={"scheme": scheme, "supported": list(SUPPORTED_SCHEMES)})
    if not scores:
        return {}

    if scheme == "equal":
        raw = dict.fromkeys(scores, 1.0)
    elif scheme == "score":
        raw = {code: float(value) for code, value in scores.items() if float(value) > 0.0}
        if not raw:
            return {}  # 全为负分：空仓，而不是挑"最不差"的
    elif scheme == "rank":
        ordered = sorted(scores, key=lambda code: float(scores[code]), reverse=True)
        raw = {code: float(len(ordered) - position) for position, code in enumerate(ordered)}
    else:  # inverse_vol
        if volatilities is None:
            raise ValidationError("inverse_vol 方案需要提供波动率", detail={"scheme": scheme})
        raw = {}
        for code in scores:
            value = float(volatilities.get(code, 0.0))
            if not math.isfinite(value) or value <= _ZERO_TOLERANCE:
                continue  # 波动缺失或为零：无法度量风险，不参与（宁可少配，不用臆测值凑数）
            raw[code] = 1.0 / value

    return normalize(raw)


def normalize(weights: Mapping[str, float]) -> dict[str, float]:
    """归一化到 Σ=1（全零/空时返回空字典；负值被夹到 0，本项目不做空）。"""
    cleaned = {code: max(float(value), 0.0) for code, value in weights.items()}
    total = sum(cleaned.values())
    if total <= _ZERO_TOLERANCE:
        return {}
    return {code: value / total for code, value in cleaned.items()}


def top_k(scores: Mapping[str, float], k: int) -> dict[str, float]:
    """按得分取前 k 名（得分相同的按代码稳定排序，保证可复现）。"""
    if k <= 0:
        raise ValidationError("top_k 必须为正", detail={"top_k": k})
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
    return dict(ordered[:k])


def realized_volatility(returns: Sequence[float]) -> float:
    """由收益序列算波动率（样本标准差）。少于 2 个观测返回 0（视为不可度量）。"""
    values = [float(value) for value in returns]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


__all__ = ["SUPPORTED_SCHEMES", "allocate", "normalize", "realized_volatility", "top_k"]
