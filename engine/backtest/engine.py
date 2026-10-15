"""回测引擎主循环（P1-B3）。

单日时序（顺序即正确性，不能调换）::

    1. 撮合昨日挂单      —— 用**今日** bar；卖出优先于买入（先回笼资金）
    2. 策略产出今日信号  —— 基于"截至今日收盘"的数据（Context 负责防越界）
    3. 信号 → 订单计划   —— 用今日收盘价测算数量，挂到明日撮合
    4. 盯市              —— 用今日收盘价更新持仓市值，记录净值
    5. 日切              —— 持仓 avail 更新（T+1：当日买入次日才可卖）

两条不可妥协的规则（均有测试）：
- **信号与成交强制错开**：T 日收盘产生的信号，最早 T+1 开盘成交。杜绝
  "同一根 bar 上既算信号又按同一价格成交"的自欺式回测。
- **资金/持仓守恒**：现金变动完全由 ``Trade.cash_flow`` 驱动，任何不一致都是 bug，
  在测试里用守恒断言锁死。
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

import pandas as pd

from engine.backtest.context import BacktestContext
from engine.backtest.cost import FeeModel
from engine.backtest.data import DataFrameBarProvider
from engine.backtest.match import ExecutionPriceMode, FillStatus, MatchEngine, MatcherConfig
from engine.backtest.metrics import PerformanceMetrics, compute_metrics
from engine.backtest.planner import PlannerConfig, TargetWeightPlanner
from engine.backtest.slippage import FixedBpsSlippage, SlippageModel
from engine.errors import ValidationError
from engine.ports import PortfolioSnapshot, Strategy
from engine.risk.engine import RiskEngine, RiskReview
from engine.types import Account, Bar, OrderPlan, Position, RiskLevel, Side, Trade

_LEVEL_ORDER: dict[RiskLevel, int] = {
    RiskLevel.NORMAL: 0,
    RiskLevel.WARN: 1,
    RiskLevel.HALT: 2,
    RiskLevel.FLATTEN: 3,
}


def _stricter_level(left: RiskLevel, right: RiskLevel) -> RiskLevel:
    """取更严的风险等级（等级只升不降）。"""
    return right if _LEVEL_ORDER[right] > _LEVEL_ORDER[left] else left


def _cancel_pending_buys(pending: list[_Pending]) -> tuple[list[_Pending], list[tuple[str, str]]]:
    """撤回未成交的买单，返回 ``(剩余挂单, 撤单记录)``（卖单保留）。"""
    remaining = [item for item in pending if item.plan.side is not Side.BUY]
    cancelled = [
        (item.plan.code, "风控降级撤单：只平不开")
        for item in pending
        if item.plan.side is Side.BUY
    ]
    if cancelled:
        _logger.warning("风控降级：撤回 %s 笔未成交买单", len(cancelled))
    return remaining, cancelled

_logger = logging.getLogger(__name__)


class FactorProvider(Protocol):
    """回测所需的因子视图：按日给出**当日可见**的因子值。

    刻意定义成 Protocol 而不是直接依赖 ``engine.factor.FactorSnapshot``：
    回测内核不需要知道因子是怎么算出来的，只要"能按日取值"。
    这样也避免 engine 内部模块之间出现不必要的耦合。
    """

    def values_on(self, day: date) -> Mapping[str, Mapping[str, float]]: ...


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start: date
    end: date
    initial_cash: Decimal | float | str = "1000000"
    price_mode: ExecutionPriceMode = "open"
    slippage_bps: float = 5.0
    max_vol_ratio: float = 0.10
    #: 挂单顺延超过该交易日数即放弃
    max_delay_days: int = 2
    lot_size: int = 100
    risk_free_rate: float = 0.02

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValidationError(
                "回测区间非法：start 晚于 end",
                detail={"start": self.start.isoformat(), "end": self.end.isoformat()},
            )


@dataclass
class BacktestResult:
    nav: pd.Series
    trades: list[Trade]
    metrics: PerformanceMetrics
    expired: list[OrderPlan]
    start: date
    end: date
    trading_days: int
    total_signals: int
    skipped_signals: int
    final_account: Account
    #: 策略异常停仓（FR-3.5）：非 None 表示策略在该日后不再产生信号，回测继续撮合存量挂单
    halted: str | None = None
    #: 风控拒绝的计划（code + 原因），回测里如实记录，避免"信号被吞掉却无人知道"
    risk_rejections: list[tuple[str, str]] = field(default_factory=list)
    #: 风控缩量的计划数
    risk_shrinks: int = 0
    #: 事中触线记录（日期 + 降级后的等级 + 原因）
    risk_level_events: list[tuple[str, str, str]] = field(default_factory=list)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trade_id": item.trade_id,
                    "code": item.code,
                    "side": str(item.side),
                    "qty": item.qty,
                    "price": float(item.price),
                    "amount": float(item.amount),
                    "total_fee": float(item.total_fee),
                    "trade_time": item.trade_time,
                }
                for item in self.trades
            ]
        )


@dataclass(slots=True)
class _Pending:
    plan: OrderPlan
    deferrals: int = 0


class BacktestEngine:
    """事件驱动回测引擎（研究态与实盘共用同一份策略代码）。"""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        provider: DataFrameBarProvider,
        strategy: Strategy,
        fee_model: FeeModel | None = None,
        slippage: SlippageModel | None = None,
        factors: FactorProvider | None = None,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.strategy = strategy
        #: 事前风控（可选）。回测接入后行为与实盘一致：被风控拒的计划不会进撮合队列
        self.risk_engine = risk_engine
        #: 预计算因子快照（可选）。引擎按日切片后注入 Context，
        #: 使策略只能看到"当日"的因子值——多因子策略因此不可能读到未来因子。
        self.factors = factors
        self.matcher = MatchEngine(
            MatcherConfig(
                max_vol_ratio=config.max_vol_ratio,
                price_mode=config.price_mode,
                lot_size=config.lot_size,
            ),
            slippage=slippage or FixedBpsSlippage(config.slippage_bps),
        )
        self.planner = TargetWeightPlanner(PlannerConfig(lot_size=config.lot_size))
        self.fee_model = fee_model or FeeModel()

    def run(self) -> BacktestResult:
        days = [day for day in self.provider.trading_days() if self.config.start <= day <= self.config.end]
        if not days:
            raise ValidationError(
                "回测区间内没有任何交易日",
                detail={"start": self.config.start.isoformat(), "end": self.config.end.isoformat()},
            )

        initial_cash = Decimal(str(self.config.initial_cash))
        account = Account(initial_cash=initial_cash, cash=initial_cash)
        positions: dict[str, Position] = {}
        pending: list[_Pending] = []
        expired: list[OrderPlan] = []
        trades: list[Trade] = []
        nav_points: list[tuple[date, float]] = []
        total_signals = 0
        skipped_signals = 0
        risk_rejections: list[tuple[str, str]] = []
        risk_shrinks = 0
        risk_level_events: list[tuple[str, str, str]] = []
        risk_level = RiskLevel.NORMAL
        peak_nav = 1.0
        previous_close_asset = initial_cash
        halted: str | None = None

        self._call_init(days[0])

        for day in days:
            bars = self.provider.bars_on(day)

            # ------------------------------------------------ 0) 事中风控（基于昨日收盘状态）
            # 必须排在撮合之前：早上就已知昨天亏了多少，降级后当天就不该再买。
            # 若排在撮合之后，"降级"会晚一天生效——昨天挂出的买单今天照样成交。
            if self.risk_engine is not None:
                risk_level, events = self._assess_portfolio(
                    day, account, positions, risk_level, peak_nav, previous_close_asset
                )
                risk_level_events.extend(events)
                if events and risk_level not in (RiskLevel.NORMAL, RiskLevel.WARN):
                    # 撤回未成交的**买单**（卖单保留——降低暴露的操作不该被拦）
                    pending, cancelled = _cancel_pending_buys(pending)
                    risk_rejections.extend(cancelled)

            # ------------------------------------------------ 1) 撮合昨日挂单（按当日开盘成交）
            pending = self._match_pending(pending, bars, day, account, positions, trades, expired)

            # ------------------------------------------------ 2) 策略信号（收盘后决策）
            context = BacktestContext(
                provider=self.provider,
                strategy_id=self.strategy.strategy_id,
                trade_date_value=day,
                bars_today=bars,
                account_state=account,
                position_map=dict(positions),
                factor_values=self._factors_on(day),
                factors_configured=self.factors is not None,
            )
            if halted is None:
                # FR-3.5 策略沙箱：异常被捕获并停仓（不再产生新信号），存量挂单继续撮合，
                # 净值继续记录——策略崩溃不能把整个回测（或实盘进程）拖崩
                try:
                    signals = list(self.strategy.on_bar(context, context.bars()))
                except Exception as exc:
                    halted = f"{type(exc).__name__}: {exc}"
                    _logger.error("策略异常，已停仓：%s", halted)
                    signals = []
            else:
                signals = []
            total_signals += len(signals)

            # ------------------------------------------------ 3) 信号 → 订单计划（明日撮合）
            plans = self.planner.build(signals, trade_date=day, account=account, positions=positions, bars=bars)
            skipped_signals += len(signals) - len(plans)
            if self.risk_engine is not None and plans:
                approved, review = self._review_plans(plans, day, account, positions, bars, risk_level=risk_level)
                plans = approved
                risk_rejections.extend((item.plan.code, item.decision.message) for item in review.rejected)
                risk_shrinks += review.shrank_count
            pending.extend(_Pending(plan=plan) for plan in plans)

            # ------------------------------------------------ 4) 盯市
            for code, position in positions.items():
                bar = bars.get(code)
                if bar is not None and not bar.is_suspended:
                    position.last_price = bar.close
            account.mark_to_market(positions.values())
            nav_value = account.refresh_nav()
            nav_points.append((day, nav_value))
            peak_nav = max(peak_nav, nav_value)
            previous_close_asset = account.total_asset

            # ------------------------------------------------ 5) 日切（T+1 解冻，供次日撮合）
            for position in positions.values():
                position.roll_to_next_day()

        nav = pd.Series(
            [value for _, value in nav_points],
            index=pd.DatetimeIndex([pd.Timestamp(day) for day, _ in nav_points]),
            dtype=float,
        )
        metrics = compute_metrics(nav, risk_free_rate=self.config.risk_free_rate)
        return BacktestResult(
            nav=nav,
            trades=trades,
            metrics=metrics,
            expired=expired,
            start=days[0],
            end=days[-1],
            trading_days=len(days),
            total_signals=total_signals,
            skipped_signals=skipped_signals,
            final_account=account,
            halted=halted,
            risk_rejections=risk_rejections,
            risk_shrinks=risk_shrinks,
            risk_level_events=risk_level_events,
        )

    def _assess_portfolio(
        self,
        day: date,
        account: Account,
        positions: dict[str, Position],
        current_level: RiskLevel,
        peak_nav: float,
        previous_close_asset: Decimal,
    ) -> tuple[RiskLevel, list[tuple[str, str, str]]]:
        """组合级事中检查：回撤 / 单日亏损触线则降级。

        等级**只升不降**（单向降级）：一旦触发 HALT，当日不会再自动恢复——
        自动恢复会让"只平不开"形同虚设，人工确认后才应复位（见 ``RiskLevel`` 的语义）。
        """
        if self.risk_engine is None or not self.risk_engine.intraday_rules:
            return current_level, []
        drawdown = (account.nav / peak_nav - 1.0) if peak_nav > 0 else 0.0
        # 先在 Decimal 域内算比率再转 float：Decimal 与 float 直接相减会抛 TypeError
        day_pnl = float(account.total_asset / previous_close_asset) - 1.0 if previous_close_asset > 0 else 0.0
        snapshot = PortfolioSnapshot(
            trade_date=day,
            account=account,
            positions=positions,
            risk_level=current_level,
            day_pnl_ratio=day_pnl,
            drawdown=drawdown,
        )
        level, decisions = self.risk_engine.check_portfolio(snapshot)
        level = _stricter_level(current_level, level)
        events: list[tuple[str, str, str]] = []
        if level is not current_level:
            events.append((day.isoformat(), str(level), "；".join(item.message for item in decisions)))
        return level, events

    def _review_plans(
        self,
        plans: list[OrderPlan],
        day: date,
        account: Account,
        positions: dict[str, Position],
        bars: Mapping[str, Bar],
        *,
        risk_level: RiskLevel = RiskLevel.NORMAL,
    ) -> tuple[list[OrderPlan], RiskReview]:
        """过一遍事前风控：通过的计划才进撮合队列。"""
        if self.risk_engine is None:  # pragma: no cover —— 调用方已判空
            return plans, RiskReview()
        review = self.risk_engine.check_plans(
            plans,
            trade_date=day,
            account=account,
            positions=positions,
            bars=bars,
            risk_level=risk_level,
            extra={"orders_today": len(plans)},
        )
        if review.rejected or review.shrank_count:
            _logger.info("风控审核：%s", review.summary())
        return [item.plan for item in review.approved], review

    def _factors_on(self, day: date) -> Mapping[str, Mapping[str, float]]:
        """当日可见因子值。未提供因子快照时返回空（策略会看到 factor() 全为 None）。"""
        return self.factors.values_on(day) if self.factors is not None else {}

    def _call_init(self, first_day: date) -> None:
        """调用策略 init（fail-fast：初始化失败直接终止回测，而不是带着半初始化状态运行）。"""
        context = BacktestContext(
            provider=self.provider,
            strategy_id=self.strategy.strategy_id,
            trade_date_value=first_day,
            bars_today=self.provider.bars_on(first_day),
            factor_values=self._factors_on(first_day),
            factors_configured=self.factors is not None,
        )
        self.strategy.init(context)

    # ============================================================ 内部
    def _match_pending(
        self,
        pending: list[_Pending],
        bars: Mapping[str, Bar],
        day: date,
        account: Account,
        positions: dict[str, Position],
        trades: list[Trade],
        expired: list[OrderPlan],
    ) -> list[_Pending]:
        """撮合挂单：卖出优先（先回笼资金），返回仍未完成的挂单。"""
        ordered = sorted(pending, key=lambda item: 0 if item.plan.side is Side.SELL else 1)
        remaining: list[_Pending] = []
        for item in ordered:
            plan = item.plan
            attempt = self.matcher.attempt(plan, bars.get(plan.code), account=account, positions=positions)

            if attempt.status is FillStatus.DEFERRED:
                if item.deferrals + 1 > self.config.max_delay_days:
                    _logger.warning("挂单顺延超限，放弃：%s %s", plan.code, attempt.reason)
                    expired.append(plan)
                    continue
                remaining.append(_Pending(plan=plan, deferrals=item.deferrals + 1))
                continue

            if attempt.price is None or attempt.qty <= 0:  # pragma: no cover —— 防御性分支
                remaining.append(_Pending(plan=plan, deferrals=item.deferrals + 1))
                continue

            self._settle_fill(plan, attempt.qty, attempt.price, day, account, positions, trades)
            if attempt.status is FillStatus.PARTIAL and plan.target_qty > attempt.qty:
                remainder = dataclasses.replace(plan, target_qty=plan.target_qty - attempt.qty)
                if item.deferrals + 1 > self.config.max_delay_days:
                    expired.append(remainder)
                    continue
                remaining.append(_Pending(plan=remainder, deferrals=item.deferrals + 1))
        return remaining

    def _settle_fill(
        self,
        plan: OrderPlan,
        qty: int,
        price: Decimal,
        day: date,
        account: Account,
        positions: dict[str, Position],
        trades: list[Trade],
    ) -> None:
        fees = self.fee_model.estimate(price * qty, plan.side)
        trade = Trade(
            code=plan.code,
            side=plan.side,
            qty=qty,
            price=price,
            strategy_id=plan.strategy_id,
            commission=fees.commission,
            stamp_tax=fees.stamp_tax,
            transfer_fee=fees.transfer_fee,
            # 成交时间必须是**模拟时间**（撮合日开盘）：回测时间轴独立于墙钟，
            # 用 datetime.now() 会让"按日统计成交"完全失真
            trade_time=datetime(day.year, day.month, day.day, 9, 30),
        )
        account.settle_cash(trade.cash_flow)

        position = positions.get(plan.code)
        if plan.side is Side.BUY:
            if position is None:
                position = Position(code=plan.code, account_id=account.account_id)
                positions[plan.code] = position
            position.apply_buy(qty, price)
        else:
            if position is None:  # pragma: no cover —— 撮合器已保证可卖
                raise ValidationError("卖出成交但无持仓记录", detail={"code": plan.code})
            position.apply_sell(qty, price)
            if position.is_empty:
                del positions[plan.code]

        trades.append(trade)


__all__ = ["BacktestConfig", "BacktestEngine", "BacktestResult", "FactorProvider"]
