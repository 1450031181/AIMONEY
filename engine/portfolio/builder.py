"""组合构建（P3-1）：得分 → 目标权重，把分配方案与约束串成一条可测的纯函数流水线。

顺序（每一步都有它必须在前的理由）::

    1. 选股            top_k 按得分取候选 —— 先把"买什么"定下来
    2. 分配方案        equal / score / rank / inverse_vol —— 决定候选之间怎么分
    3. 预算与上下限    现金缓冲、单标的上限、碎仓剔除（含重新分配）
    4. 无交易带        权重变化太小就不动 —— 抑制无效交易
    5. 换手上限        超限则向当前权重收缩 —— 走半步而不是翻仓

第 3 步之后的每一步都可能改变总仓位，因此 ``invested_weight`` 与 ``cash_weight``
是**算出来的结果**、不是假设出来的常量；策略与报告都应读它，而不是假定满仓。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from engine.errors import ValidationError
from engine.portfolio.allocator import SUPPORTED_SCHEMES, allocate, top_k
from engine.portfolio.constraints import (
    PortfolioConstraints,
    apply_no_trade_band,
    fit_to_budget,
    limit_turnover,
    turnover,
)


@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    """一次调仓的目标权重与诊断信息。"""

    weights: dict[str, float]
    dropped: Mapping[str, str]
    current_weights: Mapping[str, float]
    turnover: float
    scheme: str
    candidates: int = 0

    @property
    def invested_weight(self) -> float:
        return sum(self.weights.values())

    @property
    def cash_weight(self) -> float:
        return max(1.0 - self.invested_weight, 0.0)

    def summary(self) -> str:
        head = (
            f"组合：{len(self.weights)} 只，方案 {self.scheme}，"
            f"仓位 {self.invested_weight:.1%}（现金 {self.cash_weight:.1%}），换手 {self.turnover:.1%}"
        )
        if self.dropped:
            head += f"；剔除 {len(self.dropped)} 只"
        return head


@dataclass(frozen=True, slots=True)
class PortfolioBuilder:
    """把因子得分转成带约束的目标权重（无状态、纯函数式）。"""

    scheme: str = "equal"
    top_k: int = 20
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)

    def __post_init__(self) -> None:
        if self.scheme not in SUPPORTED_SCHEMES:
            raise ValidationError(
                "不支持的分配方案", detail={"scheme": self.scheme, "supported": list(SUPPORTED_SCHEMES)}
            )
        if self.top_k <= 0:
            raise ValidationError("top_k 必须为正", detail={"top_k": self.top_k})

    def build(
        self,
        scores: Mapping[str, float],
        *,
        current_weights: Mapping[str, float] | None = None,
        volatilities: Mapping[str, float] | None = None,
    ) -> PortfolioDecision:
        current = {code: float(value) for code, value in (current_weights or {}).items()}
        if not scores:
            return PortfolioDecision(
                weights={},
                dropped=dict.fromkeys(current, "当期无可用得分"),
                current_weights=current,
                turnover=turnover({}, current),
                scheme=self.scheme,
            )

        selected = top_k(scores, self.top_k)
        dropped: dict[str, str] = {
            code: f"未入选前 {self.top_k} 名（得分 {scores[code]:+.4f}）"
            for code in scores
            if code not in selected
        }

        allocated = allocate(selected, scheme=self.scheme, volatilities=volatilities)
        if not allocated:
            for code in selected:
                dropped[code] = "得分为负，long-only 下不持有"
            return PortfolioDecision(
                weights={},
                dropped=dropped,
                current_weights=current,
                turnover=turnover({}, current),
                scheme=self.scheme,
                candidates=len(selected),
            )

        fitted, rejected = fit_to_budget(allocated, self.constraints)
        dropped.update(rejected)

        adjusted = apply_no_trade_band(fitted, current, self.constraints.no_trade_band)
        adjusted, actual_turnover = limit_turnover(adjusted, current, self.constraints.max_turnover)
        weights = {code: value for code, value in adjusted.items() if value > 0.0}

        return PortfolioDecision(
            weights=weights,
            dropped=dropped,
            current_weights=current,
            turnover=actual_turnover,
            scheme=self.scheme,
            candidates=len(selected),
        )


__all__ = ["PortfolioBuilder", "PortfolioDecision"]
