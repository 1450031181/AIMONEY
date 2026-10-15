"""组合约束（P3-1）：把"理想权重"变成"可执行权重"。

现实里没人能按理论权重成交，四条约束按顺序生效：

| 约束 | 作用 | 不做的后果 |
| --- | --- | --- |
| ``cash_buffer`` | 留出现金应对费用/滑点/取整误差 | 满仓时第一笔手续费就可能让买单因资金不足失败 |
| ``max_weight`` | 单标的上限（集中度） | 一只票的坏消息能吃掉整个组合 |
| ``min_weight`` | 剔除过小的仓位 | 碎仓的手续费（最低 5 元）比它的贡献还高 |
| ``no_trade_band`` | 权重变化小于阈值则不动 | 每天为 0.1% 的权重差付一次手续费 |
| ``max_turnover`` | 单次调仓换手上限 | 一次信号抖动就把组合翻个底朝天 |

**两处必须"重新分配"而不是简单截断**（都实现了并被测试守住）：
1. ``cap_weights``：截断上限后多余的权重按比例分给未达上限的标的。
   丢掉它等于偷偷增加现金仓位——风险敞口与预期不符，且不会报错；
   而当所有标的都顶到上限、容量确实不够时（``N × max_weight < budget``），
   我们就**老老实实留现金**，而不是把上限顶破。
2. ``min_weight`` 剔除后同样重新分配，否则剔除碎仓等于再次偷偷加现金。

一个必须接受的取舍：``no_trade_band`` 会把目标权重向当前权重"吸"回去，
因此结果**不再严格等于理想权重**——这是刻意的成本控制，不是 bug。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from engine.errors import ValidationError
from engine.portfolio.allocator import normalize

_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """组合约束。``None`` 表示不启用该约束（默认全部不启用 = 与不加约束等价）。"""

    max_weight: float | None = None
    min_weight: float = 0.0
    cash_buffer: float = 0.0
    max_turnover: float | None = None
    no_trade_band: float = 0.0

    def __post_init__(self) -> None:
        if self.max_weight is not None and not 0.0 < self.max_weight <= 1.0:
            raise ValidationError("max_weight 应在 (0, 1] 内", detail={"max_weight": self.max_weight})
        if self.min_weight < 0.0:
            raise ValidationError("min_weight 不能为负", detail={"min_weight": self.min_weight})
        if self.max_weight is not None and self.min_weight >= self.max_weight:
            raise ValidationError(
                "min_weight 必须小于 max_weight，否则不存在可行权重",
                detail={"min_weight": self.min_weight, "max_weight": self.max_weight},
            )
        if not 0.0 <= self.cash_buffer < 1.0:
            raise ValidationError("cash_buffer 应在 [0, 1) 内", detail={"cash_buffer": self.cash_buffer})
        if self.max_turnover is not None and self.max_turnover <= 0.0:
            raise ValidationError("max_turnover 必须为正", detail={"max_turnover": self.max_turnover})
        if self.no_trade_band < 0.0:
            raise ValidationError("no_trade_band 不能为负", detail={"no_trade_band": self.no_trade_band})

    @property
    def budget(self) -> float:
        """可投资比例（1 − 现金缓冲）。"""
        return 1.0 - self.cash_buffer


def turnover(target: Mapping[str, float], current: Mapping[str, float]) -> float:
    """换手率 = ``0.5 × Σ|w_target − w_current|``（与因子评价里的口径一致）。"""
    codes = set(target) | set(current)
    return 0.5 * sum(abs(float(target.get(code, 0.0)) - float(current.get(code, 0.0))) for code in codes)


def scale_to_budget(weights: Mapping[str, float], budget: float) -> dict[str, float]:
    """把权重整体缩放到目标投资比例（不做归一化，保留相对关系）。"""
    if budget <= 0.0:
        return {}
    return {code: float(value) * budget for code, value in weights.items()}


def cap_weights(weights: Mapping[str, float], max_weight: float | None, *, max_iterations: int = 50) -> dict[str, float]:
    """单标的上限截断 + **把多余权重按比例重新分配**给未达上限的标的。

    迭代到收敛（每轮至少有一个标的被截断，故轮数上界为标的数）；
    容量不足时返回总和 < 1 的权重，剩余部分即现金——不破上限。
    """
    if max_weight is None:
        return {code: float(value) for code, value in weights.items()}
    result = {code: float(value) for code, value in weights.items()}
    for _ in range(max_iterations):
        over = {code: value for code, value in result.items() if value > max_weight + _TOLERANCE}
        if not over:
            break
        excess = sum(value - max_weight for value in over.values())
        for code in over:
            result[code] = max_weight
        recipients = {code: value for code, value in result.items() if value < max_weight - _TOLERANCE}
        if not recipients:
            break  # 全部顶到上限：容量不足，留现金
        room = sum(max_weight - value for value in recipients.values())
        transfer = min(excess, room)
        if transfer <= 0:
            break
        if sum(recipients.values()) <= _TOLERANCE:
            share = transfer / len(recipients)
            for code in recipients:
                result[code] = share
        else:
            total = sum(recipients.values())
            for code, value in recipients.items():
                result[code] = value + transfer * (value / total)
    return result


def fit_to_budget(
    weights: Mapping[str, float],
    constraints: PortfolioConstraints,
    *,
    max_passes: int = 3,
) -> tuple[dict[str, float], dict[str, str]]:
    """按 ``min_weight`` / ``max_weight`` / ``cash_buffer`` 把权重调到可执行状态。

    返回 ``(权重, 被剔除标的及原因)``。迭代若干轮是因为"剔除碎仓 → 重新分配 → 又顶到上限"
    可能相互触发；实践中 2~3 轮即收敛，且每轮都只会让结果更接近可行域。
    """
    dropped: dict[str, str] = {}
    current = scale_to_budget(normalize(weights), constraints.budget)
    for _ in range(max_passes):
        kept = {
            code: value
            for code, value in current.items()
            if value >= constraints.min_weight - _TOLERANCE
        }
        for code in current:
            if code not in kept:
                dropped[code] = f"权重 {current[code]:.4f} 低于下限 {constraints.min_weight:.4f}"
        if not kept:
            return {}, dropped
        current = scale_to_budget(normalize(kept), constraints.budget)
        capped = cap_weights(current, constraints.max_weight)
        if _close(capped, current):
            return capped, dropped
        current = capped
    return current, dropped


def apply_no_trade_band(
    target: Mapping[str, float],
    current: Mapping[str, float],
    band: float,
) -> dict[str, float]:
    """权重变化小于 ``band`` 的标的保持**当前权重**（不产生交易）。"""
    if band <= 0.0:
        return {code: float(value) for code, value in target.items()}
    result: dict[str, float] = {}
    for code, value in target.items():
        if abs(float(value) - float(current.get(code, 0.0))) < band:
            result[code] = float(current.get(code, 0.0))
        else:
            result[code] = float(value)
    return result


def limit_turnover(
    target: Mapping[str, float],
    current: Mapping[str, float],
    max_turnover: float | None,
) -> tuple[dict[str, float], float]:
    """换手超限时把目标权重**向当前权重收缩**，返回 ``(调整后权重, 实际换手)``。

    收缩而不是"挑一部分交易"：挑着交易会让组合偏离目标权重且难以解释，
    等比例收缩则保持相对结构、只是走半步。
    """
    if max_turnover is None:
        return {code: float(value) for code, value in target.items()}, turnover(target, current)
    actual = turnover(target, current)
    if actual <= max_turnover or actual <= _TOLERANCE:
        return {code: float(value) for code, value in target.items()}, actual

    ratio = max_turnover / actual
    codes = set(target) | set(current)
    adjusted = {
        code: float(current.get(code, 0.0)) + ratio * (float(target.get(code, 0.0)) - float(current.get(code, 0.0)))
        for code in codes
    }
    return {code: value for code, value in adjusted.items() if value > 0.0}, max_turnover


def _close(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    if set(left) != set(right):
        return False
    return all(abs(float(left[key]) - float(right[key])) <= _TOLERANCE for key in left)


__all__ = [
    "PortfolioConstraints",
    "apply_no_trade_band",
    "cap_weights",
    "fit_to_budget",
    "limit_turnover",
    "scale_to_budget",
    "turnover",
]
