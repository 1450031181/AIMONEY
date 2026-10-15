"""订单管理（P4-1）：本地委托账本 + 券商通道 + 断线校准。

它是"策略想交易"与"券商真报单"之间的唯一通道，负责四件事：

1. **闸门与风控**：交易门禁未开 / 急停激活 / 风控拒单，一律不下发券商。
   顺序是"先风控后报单"，绝不允许先报出去再补救。
2. **幂等报单**：幂等键由 ``策略|日期|代码|方向|数量`` **确定性**推导，
   而不是用随机计划号。原因很直接：进程崩了重启后，随机号会变、幂等键就失效，
   同一笔单会被重复报出——这正是"重启不重复下单"这条要求的落点。
3. **异常宁可疑、不可重报**：报单通道抛异常时标记订单为 ``UNKNOWN``（不是 REJECTED），
   必须走查单校准；直接重报会造成真实世界的重复委托。
4. **断线校准**：以券商全量委托为准，把本地账本纠正过来；
   本地有而券商没有的，判定为废单。

交易日循环的**时序**（顺序错了会出真实问题）：

    1. 推送当日行情 → 券商更新参考价，并撮合昨日挂单
    2. 计算信号 → 组合权重 → 事前风控 → 本模块报单（按当日价冻结资金）
    3. 日切（T+1 解冻）→ 记录净值

第 1 步必须在第 2 步之前：模拟盘券商（以及真实券商）冻结资金需要参考价，
没有参考价的市价单会被明确拒绝——这是刻意的（拿臆测价格冻结资金会导致超额委托）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from engine.errors import BrokerError
from engine.ports import BrokerAdapter
from engine.risk.engine import RiskEngine
from engine.types import (
    Account,
    Bar,
    Order,
    OrderPlan,
    OrderType,
    Position,
    RiskLevel,
)

_logger = logging.getLogger(__name__)


class _GateLike(Protocol):
    """交易门禁（默认关闭，fail-safe）。"""

    def is_open(self) -> bool: ...


class _KillSwitchLike(Protocol):
    def is_active(self) -> bool: ...

    def state(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    submitted: tuple[Order, ...] = ()
    #: 风控/券商拒单：(计划, 原因)
    rejected: tuple[tuple[OrderPlan, str], ...] = ()
    #: 通道异常、状态未知、必须查单校准的订单
    unknown: tuple[Order, ...] = ()
    #: 幂等命中的幂等键（重复提交被正确拦下）
    duplicates: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.rejected and not self.unknown

    def summary(self) -> str:
        parts = [f"报单 {len(self.submitted)} 笔"]
        if self.duplicates:
            parts.append(f"幂等命中 {len(self.duplicates)} 笔")
        if self.rejected:
            parts.append(f"拒单 {len(self.rejected)} 笔")
        if self.unknown:
            parts.append(f"⚠ 状态未知 {len(self.unknown)} 笔（必须查单校准，禁止重报）")
        return "；".join(parts)


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    checked: int = 0
    corrected: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unchanged: int = 0

    def summary(self) -> str:
        return (
            f"校准 {self.checked} 笔：纠正 {len(self.corrected)} 笔，"
            f"废单 {len(self.missing)} 笔，无需变更 {self.unchanged} 笔"
        )


@dataclass
class OrderManager:
    """本地委托账本与券商通道的编排者。"""

    broker: BrokerAdapter
    risk_engine: RiskEngine | None = None
    gate: _GateLike | None = None
    kill_switch: _KillSwitchLike | None = None
    _orders: dict[str, Order] = field(default_factory=dict, init=False, repr=False)
    _by_idem: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)

    # ============================================================ 连接与查询
    def connect(self) -> None:
        self.broker.connect()
        self._connected = True

    def disconnect(self) -> None:
        self.broker.disconnect()
        self._connected = False

    def orders(self, *, open_only: bool = False) -> list[Order]:
        items = self._orders.values()
        return [order for order in items if order.is_open] if open_only else list(items)

    def order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    # ============================================================ 报单
    def submit(
        self,
        plans: Sequence[OrderPlan],
        *,
        trade_date: date,
        account: Account,
        positions: Mapping[str, Position],
        bars: Mapping[str, Bar] | None = None,
        risk_level: RiskLevel = RiskLevel.NORMAL,
    ) -> SubmissionResult:
        if not plans:
            return SubmissionResult()
        blocked = self._blocking_reason()
        if blocked is not None:
            # 门禁/急停：整批不下发，且**不调用券商**（避免"先报出去再补救"）
            _logger.warning("报单被拦截：%s", blocked)
            return SubmissionResult(rejected=tuple((plan, blocked) for plan in plans))

        approved = list(plans)
        rejected: list[tuple[OrderPlan, str]] = []
        if self.risk_engine is not None:
            review = self.risk_engine.check_plans(
                plans,
                trade_date=trade_date,
                account=account,
                positions=positions,
                bars=bars,
                risk_level=risk_level,
                extra={"orders_today": len(self._orders)},
            )
            approved = [item.plan for item in review.approved]
            rejected.extend((item.plan, item.decision.message) for item in review.rejected)

        submitted: list[Order] = []
        unknown: list[Order] = []
        duplicates: list[str] = []
        for plan in approved:
            order = self._to_order(plan, trade_date)
            if order.idem_key in self._by_idem:
                duplicates.append(order.idem_key)
                continue
            try:
                broker_order_id = self.broker.place_order(order)
            except BrokerError as exc:
                # 通道异常：状态未知。**绝不重报**——重报可能造成真实重复委托
                order.mark_submitted(None)
                order.mark_unknown(f"报单通道异常：{exc}")
                self._remember(order)
                unknown.append(order)
                _logger.error("报单状态未知，需查单校准：%s | %s", order.order_id, exc)
                continue
            self._remember(order, broker_order_id=broker_order_id)
            submitted.append(order)

        result = SubmissionResult(
            submitted=tuple(submitted),
            rejected=tuple(rejected),
            unknown=tuple(unknown),
            duplicates=tuple(duplicates),
        )
        _logger.info("订单提交：%s", result.summary())
        return result

    def cancel(self, broker_order_id: str) -> bool:
        cancelled = self.broker.cancel_order(broker_order_id)
        if cancelled:
            for order in self._orders.values():
                if order.broker_order_id == broker_order_id and order.is_open:
                    order.cancel("人工撤单")
        return cancelled

    def cancel_all(self) -> int:
        count = 0
        for order in list(self._orders.values()):
            if order.is_open and order.broker_order_id and self.cancel(order.broker_order_id):
                count += 1
        return count

    # ============================================================ 校准
    def reconcile(self) -> ReconcileResult:
        """以券商全量委托为准纠正本地账本。

        三种情形都要处理：券商已成交/已撤而本地还停留在在途；本地在途而券商查无此单
        （废单）；状态一致则不动。**没有这一步，重启或断线后本地账本会与真实持仓脱节**。
        """
        truth = {order.broker_order_id: order for order in self.broker.query_orders() if order.broker_order_id}
        corrected: list[str] = []
        missing: list[str] = []
        unchanged = 0
        for order in self._orders.values():
            if not order.is_open:
                # 已终结的委托无需校准。计入 unchanged 是为了让
                # "纠正 + 废单 + 未变更" 与"检查总数"对得上——
                # 计数不闭合的报告会让人以为漏查了什么
                unchanged += 1
                continue
            broker_order = truth.get(order.broker_order_id or "")
            if broker_order is None:
                # 本地在途、券商查无此单 → 废单（报单从未真正到达或被对端丢弃）
                order.reject("券商查无此委托（废单）")
                missing.append(order.order_id)
                continue
            if not order.sync_from_broker(broker_order):
                unchanged += 1
                continue
            corrected.append(order.order_id)
        result = ReconcileResult(
            checked=len(self._orders),
            corrected=tuple(corrected),
            missing=tuple(missing),
            unchanged=unchanged,
        )
        _logger.info("%s", result.summary())
        return result

    # ============================================================ 内部
    def _blocking_reason(self) -> str | None:
        if not self._connected:
            return "券商未连接"
        if self.kill_switch is not None and self.kill_switch.is_active():
            state = self.kill_switch.state()
            return f"急停开关生效：{state.get('reason') or '未提供原因'}"
        if self.gate is not None and not self.gate.is_open():
            return "交易门禁未开启（数据质量未通过或尚未校验）"
        return None

    @staticmethod
    def build_idem_key(plan: OrderPlan, trade_date: date, *, seq: int = 0) -> str:
        """确定性幂等键。

        刻意**不用** ``plan.plan_id``（随机生成）：进程重启后随机号会变，
        幂等键随之失效，同一笔单会被重复报出。用业务要素组合才具备跨重启的稳定性。
        """
        signature = f"{plan.strategy_id}|{trade_date.isoformat()}|{plan.code}|{plan.side.value}|{plan.target_qty}"
        return Order.build_idem_key(signature, seq)

    def _to_order(self, plan: OrderPlan, trade_date: date) -> Order:
        return Order(
            code=plan.code,
            side=plan.side,
            qty=plan.target_qty,
            plan_id=plan.plan_id,
            strategy_id=plan.strategy_id,
            price=plan.limit_price,
            order_type=OrderType.LIMIT if plan.limit_price is not None else OrderType.MARKET,
            idem_key=self.build_idem_key(plan, trade_date),
        )

    def _remember(self, order: Order, *, broker_order_id: str | None = None) -> None:
        if broker_order_id:
            order.broker_order_id = broker_order_id
        self._orders[order.order_id] = order
        if order.idem_key:
            self._by_idem[order.idem_key] = order.order_id


__all__ = ["OrderManager", "ReconcileResult", "SubmissionResult"]
