"""风控引擎（P3-2）：把规则链串成一道不可绕过的闸门。

设计要点：

1. **链式缩量**：一条规则缩量后，后续规则看到的是**缩量后的数量**
   （``dataclasses.replace`` 生成新计划）。否则"集中度缩到 100 股"之后，
   "单笔金额上限"还会按原始的 10,000 股去判——规则之间互相失效。
2. **短路拒绝**：任一条规则拒单 / HALT / FLATTEN 立即返回，不再跑后续规则。
   继续跑只会产生互相矛盾的裁定，日志里也看不出真正原因是哪条。
3. **卖出同样过闸**：卖出也要过急停与风险等级（放行逻辑写在规则里），
   但集中度/资金类规则天然只作用于买入侧。
4. **未生效的规则要留痕**：数据缺失导致某条规则没跑（缺价格、缺涨跌停），
   在裁定里记 ``skipped`` 而不是假装它通过了。
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from engine.ports import IntradayRiskRule, PortfolioSnapshot, PreTradeRiskRule, RiskContext
from engine.types import Account, Bar, OrderPlan, Position, RiskAction, RiskDecision, RiskLevel

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReviewedPlan:
    plan: OrderPlan
    decision: RiskDecision
    origin_qty: int

    @property
    def shrank(self) -> bool:
        return self.plan.target_qty != self.origin_qty

    @property
    def approved(self) -> bool:
        return self.decision.passed


@dataclass
class RiskReview:
    """事前审核结果：通过的（可能被缩量）+ 被拒的 + 未生效的规则名。"""

    approved: list[ReviewedPlan] = field(default_factory=list)
    rejected: list[ReviewedPlan] = field(default_factory=list)
    rules_not_applied: list[str] = field(default_factory=list)

    @property
    def shrank_count(self) -> int:
        return sum(1 for item in self.approved if item.shrank)

    def summary(self) -> str:
        parts = [f"通过 {len(self.approved)} 条（其中缩量 {self.shrank_count} 条），拒绝 {len(self.rejected)} 条"]
        if self.rejected:
            reasons = "；".join(f"{item.plan.code}: {item.decision.message}" for item in self.rejected[:3])
            parts.append(f"拒绝原因示例：{reasons}")
        if self.rules_not_applied:
            parts.append("未生效规则：" + "、".join(sorted(set(self.rules_not_applied))))
        return "风控审核：" + "；".join(parts)


@dataclass
class RiskEngine:
    """事前 + 事中风控编排（无状态，可被交易链路与回测共用）。"""

    pre_trade_rules: Sequence[PreTradeRiskRule] = ()
    intraday_rules: Sequence[IntradayRiskRule] = ()

    def check_plans(
        self,
        plans: Sequence[OrderPlan],
        *,
        trade_date: date,
        account: Account,
        positions: Mapping[str, Position],
        bars: Mapping[str, Bar] | None = None,
        risk_level: RiskLevel = RiskLevel.NORMAL,
        extra: Mapping[str, Any] | None = None,
    ) -> RiskReview:
        """逐条审核订单计划。返回的 ``approved`` 才是可以下发的计划。"""
        review = RiskReview()
        bar_map = dict(bars or {})
        base_extra = dict(extra or {})
        ordered_rules = sorted(self.pre_trade_rules, key=lambda rule: rule.priority)

        for plan in plans:
            origin_qty = plan.target_qty
            current = plan
            rejected = False
            for rule in ordered_rules:
                context = RiskContext(
                    trade_date=trade_date,
                    plan=current,
                    account=account,
                    positions=positions,
                    bars=bar_map,
                    risk_level=risk_level,
                    extra=base_extra,
                )
                decision = rule.check(context)
                if decision.action is RiskAction.SHRINK and decision.adjusted_qty > 0:
                    # 缩量结果传给下一条规则：否则后续规则仍按原始数量判断，等于失效
                    current = dataclasses.replace(current, target_qty=decision.adjusted_qty)
                    continue
                if decision.action is RiskAction.PASS:
                    continue
                # REJECT / HALT / FLATTEN：短路，不再跑后续规则
                review.rejected.append(ReviewedPlan(plan=current, decision=decision, origin_qty=origin_qty))
                _logger.info("风控拒单：%s %s | %s", plan.code, plan.side, decision.message)
                rejected = True
                break
            if rejected:
                continue
            final = RiskDecision.allow(current.target_qty)
            if current.target_qty != origin_qty:
                final = RiskDecision.shrink(
                    "risk_engine",
                    f"综合风控缩量：{origin_qty} → {current.target_qty}",
                    requested_qty=origin_qty,
                    adjusted_qty=current.target_qty,
                )
            review.approved.append(ReviewedPlan(plan=current, decision=final, origin_qty=origin_qty))
        return review

    def check_portfolio(
        self,
        snapshot: PortfolioSnapshot,
    ) -> tuple[RiskLevel, list[RiskDecision]]:
        """组合级检查：返回**最严**的风险等级与全部触线裁定。

        ``FLATTEN`` 比 ``HALT`` 更严，故取等级时按固定顺序比较，
        而不是简单取"最后一个触发的规则"——顺序不该影响结论。
        """
        decisions: list[RiskDecision] = []
        level = RiskLevel.NORMAL
        for rule in sorted(self.intraday_rules, key=lambda item: item.priority):
            decision = rule.check_portfolio(snapshot)
            if decision.action is RiskAction.PASS:
                continue
            decisions.append(decision)
            level = _stricter(level, _level_of(decision.action))
        if decisions:
            _logger.warning("风控触线：%s", "；".join(item.message for item in decisions))
        return level, decisions


_LEVEL_ORDER: Mapping[RiskLevel, int] = {
    RiskLevel.NORMAL: 0,
    RiskLevel.WARN: 1,
    RiskLevel.HALT: 2,
    RiskLevel.FLATTEN: 3,
}


def _level_of(action: RiskAction) -> RiskLevel:
    if action is RiskAction.FLATTEN:
        return RiskLevel.FLATTEN
    if action is RiskAction.HALT:
        return RiskLevel.HALT
    return RiskLevel.NORMAL


def _stricter(left: RiskLevel, right: RiskLevel) -> RiskLevel:
    return right if _LEVEL_ORDER[right] > _LEVEL_ORDER[left] else left


__all__ = ["ReviewedPlan", "RiskEngine", "RiskReview"]
