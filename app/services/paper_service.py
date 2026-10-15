"""模拟盘运行器（P4-2）：把"一天"串成一条可重跑的命令。

它把已有的全部零件按**真实交易时序**装配起来：

    1. 盘前闸门    数据门禁（质量 FAIL 即关门）+ 急停开关 + 数据覆盖检查
    2. 恢复账本    账户/持仓/委托从磁盘恢复（否则每天从零开始、且会重复下单）
    3. 推送行情    券商更新参考价并撮合**昨日**挂单 → 成交落到账本
    4. 策略决策    用与回测**同一个** Context/策略代码产出信号（因子值按日切片注入）
    5. 组合与风控  信号 → 目标权重 → 计划 → 事前风控（缩量/拒单）
    6. 报单        经 OMS 幂等报出（对下一交易日有效）
    7. 盯市与日切  记录净值、T+1 解冻、状态落盘

第 3 步必须在第 4 步之前：冻结资金需要参考价，没有参考价的市价单会被拒绝。

**这个运行器的价值在于"同一份策略代码"**：回测里跑过的策略无需任何改动，
直接在这里以真实时序、真实费用、真实约束连续运行——这是上真钱之前唯一诚实的验证方式。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine.backtest.context import BacktestContext
from engine.backtest.data import DataFrameBarProvider
from engine.backtest.planner import TargetWeightPlanner
from engine.broker.paper import OpenOrderRecord, PaperBrokerAdapter
from engine.datahub.calendar import warmup_start
from engine.datahub.quality import TradeGate
from engine.datahub.store import ParquetStore
from engine.datasource.local_source import LocalDataSource
from engine.errors import DataNotReadyError
from engine.factor.base import get_factor
from engine.factor.preprocess import DEFAULT_STEPS
from engine.factor.snapshot import FactorSnapshot, build_snapshot_from_prices
from engine.risk.engine import RiskEngine
from engine.risk.kill_switch import KillSwitch
from engine.risk.rules import default_intraday_rules, default_pre_trade_rules
from engine.strategy.loader import load_strategy
from engine.trade.ledger import DEFAULT_LEDGER_FILE, BrokerSnapshot, JsonLedger
from engine.trade.oms import OrderManager, SubmissionResult
from engine.types import Order

_logger = logging.getLogger(__name__)

NAV_FILE = "nav.jsonl"


@dataclass(frozen=True, slots=True)
class PaperRunRequest:
    strategy_dir: Path
    trade_date: date
    codes: tuple[str, ...] = ()
    params_override: Mapping[str, Any] = field(default_factory=dict)
    factors: tuple[str, ...] = ()
    factor_preprocess: tuple[str, ...] = DEFAULT_STEPS
    dataset: str = "daily"
    #: 初始资金（仅首次运行生效；之后以账本为准）
    initial_cash: str = "1000000"
    #: 强制重跑已处理过的交易日。**危险**：上次的挂单会在同一天行情上再撮一次，
    #: 产生本不该有的成交。只用于排查问题，不要放进定时任务。
    force: bool = False


@dataclass(frozen=True, slots=True)
class PaperRunReport:
    trade_date: date
    strategy_id: str
    output_dir: Path
    signals: int
    plans: int
    submission: SubmissionResult
    positions: tuple[tuple[str, int, int], ...]
    total_asset: str
    nav: float
    skipped_reason: str | None = None
    matched_trades: int = 0

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None

    def summary(self) -> str:
        if self.skipped_reason is not None:
            return f"[{self.strategy_id}] {self.trade_date} 跳过：{self.skipped_reason}"
        parts = [
            f"[{self.strategy_id}] {self.trade_date} 模拟盘：成交 {self.matched_trades} 笔，"
            f"信号 {self.signals} 条 → 计划 {self.plans} 条，",
            self.submission.summary(),
            f"；总资产 {self.total_asset}，净值 {self.nav:.4f}，持仓 {len(self.positions)} 只",
        ]
        return "".join(parts)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "strategy_id": self.strategy_id,
            "output_dir": str(self.output_dir),
            "ok": self.ok,
            "skipped_reason": self.skipped_reason,
            "matched_trades": self.matched_trades,
            "signals": self.signals,
            "plans": self.plans,
            "submitted": len(self.submission.submitted),
            "rejected": [{"code": plan.code, "reason": reason} for plan, reason in self.submission.rejected],
            "unknown": [order.order_id for order in self.submission.unknown],
            "duplicates": list(self.submission.duplicates),
            "positions": [{"code": code, "qty": qty, "avail": avail} for code, qty, avail in self.positions],
            "total_asset": self.total_asset,
            "nav": self.nav,
        }


class PaperTradingService:
    """模拟盘编排（只依赖 Settings）。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    # ============================================================ 主流程
    def run(self, request: PaperRunRequest) -> PaperRunReport:
        output_dir = self._output_dir(request)
        gate = TradeGate(self.settings.gate_file)
        switch = KillSwitch(self.settings.kill_switch_file)
        ledger = JsonLedger(output_dir / DEFAULT_LEDGER_FILE)

        loaded = load_strategy(request.strategy_dir, params_override=dict(request.params_override))
        strategy_id = loaded.strategy.strategy_id

        blocked = self._blocking_reason(gate, switch)
        if blocked is not None:
            # 盘前闸门未通过：**不交易**，但仍写一份报告，这样"今天为什么没动"有据可查
            report = self._skipped(request, strategy_id, output_dir, blocked)
            self._write_report(report)
            _logger.warning("%s", report.summary())
            return report

        # 账本只加载**一次**并把同一批 Order 对象交给券商与 OMS：
        # 若各加载一份，券商的成交会改到 A 份、OMS 账本还停在 B 份，
        # 表现为"已成交的委托仍显示在途"，且落盘的账本永远追不上真实状态。
        restored_orders = ledger.load_orders()
        broker, snapshot = self._restore_broker(ledger, request, restored_orders)
        # 日级防重：重复跑同一天**不是幂等的**——上次的挂单会在同一天行情上再撮一次。
        # 定时任务因此天然安全；需要重跑（排查问题）时必须显式 --force。
        processed_through = snapshot.last_trade_date if snapshot is not None else None
        if not request.force and processed_through is not None and request.trade_date <= processed_through:
            report = self._skipped(
                request,
                strategy_id,
                output_dir,
                f"交易日 {request.trade_date} 已处理过（最后处理日 {processed_through}）；"
                "如需重跑请显式加 --force",
            )
            self._write_report(report)
            _logger.warning("%s", report.summary())
            return report
        manager = self._build_manager(broker, switch, restored_orders)

        try:
            store = ParquetStore(self.settings.parquet_dir)
            store.require_data(request.dataset)
            local = LocalDataSource(store, dataset=request.dataset)
            context_provider, factor_snapshot = self._prepare_data(local, request, loaded.params)
        except DataNotReadyError as exc:
            # 数据未就绪（尚未同步 / 无覆盖）属于**预期内的跳过**，不是故障：
            # 定时任务"今天还没同步数据"不应触发告警，否则监控很快会被脱敏
            report = self._skipped(request, strategy_id, output_dir, str(exc))
            self._write_report(report)
            _logger.warning("%s", report.summary())
            return report

        day_bars = context_provider.bars_on(request.trade_date)
        if not day_bars:
            report = self._skipped(
                request,
                strategy_id,
                output_dir,
                f"{request.trade_date} 无行情数据（停牌/未同步？）",
            )
            self._write_report(report)
            return report

        # ---- 0) 日切：新的一天开始，昨日买入转为可卖（T+1 在**日初**解冻）
        # 必须排在撮合之前：否则今天要卖的持仓还带着昨日买入的冻结状态，
        # 卖出计划会被"可卖不足"错误地缩量甚至拒单。
        broker.roll_to_next_day()

        # ---- 1) 推送行情：更新参考价并撮合昨日挂单
        matched = broker.push_bars(day_bars)

        # ---- 2) 策略决策（与回测同一套 Context/策略代码）
        context = BacktestContext(
            provider=context_provider,
            strategy_id=strategy_id,
            trade_date_value=request.trade_date,
            bars_today=day_bars,
            account_state=broker.query_account(),
            position_map={position.code: position for position in broker.query_positions()},
            factor_values=factor_snapshot.values_on(request.trade_date) if factor_snapshot else {},
            factors_configured=factor_snapshot is not None,
        )
        signals = list(loaded.strategy.on_bar(context, context.bars()))

        # ---- 3) 信号 → 计划 → 风控 → 报单
        account = broker.query_account()
        positions = {position.code: position for position in broker.query_positions()}
        plans = TargetWeightPlanner().build(
            signals, trade_date=request.trade_date, account=account, positions=positions, bars=day_bars
        )
        submission = manager.submit(
            plans,
            trade_date=request.trade_date,
            account=account,
            positions=positions,
            bars=day_bars,
        )

        # ---- 4) 盯市、日切、落盘
        # 注意：这里**不能**再推送一次行情。刚报出的委托目标是"下一交易日"，
        # 再推一次当日行情会让它们当场成交，直接破坏"信号与成交错开一日"。
        # 最新价在步骤 1 的 push_bars 里已经更新，query_account 会据此盯市。
        final_account = broker.query_account()
        final_positions = broker.query_positions()
        self._persist(ledger, broker, manager, request.trade_date)
        nav = self._append_nav(output_dir, request.trade_date, final_account)

        report = PaperRunReport(
            trade_date=request.trade_date,
            strategy_id=strategy_id,
            output_dir=output_dir,
            signals=len(signals),
            plans=len(plans),
            submission=submission,
            positions=tuple((p.code, p.qty, p.avail_qty) for p in final_positions),
            total_asset=str(final_account.total_asset),
            nav=nav,
            matched_trades=len(matched),
        )
        self._write_report(report)
        _logger.info("%s", report.summary())
        return report

    # ============================================================ 装配
    def _build_manager(
        self, broker: PaperBrokerAdapter, switch: KillSwitch, restored_orders: Sequence[Order]
    ) -> OrderManager:
        limits = self.settings.risk
        risk_engine: RiskEngine | None = None
        if limits.enabled:
            risk_engine = RiskEngine(
                pre_trade_rules=default_pre_trade_rules(
                    switch=switch,
                    max_weight=limits.max_weight,
                    max_order_value=limits.max_order_value,
                    max_orders_per_day=limits.max_orders_per_day,
                ),
                intraday_rules=default_intraday_rules(
                    max_drawdown=limits.max_drawdown,
                    max_daily_loss=limits.max_daily_loss,
                    liquidation_drawdown=limits.liquidation_drawdown,
                ),
            )
        manager = OrderManager(broker=broker, risk_engine=risk_engine, kill_switch=switch)
        manager.connect()
        restored = manager.restore(restored_orders)
        if restored:
            _logger.info("已恢复委托账本 %s 条（跨进程幂等的前提）", restored)
        return manager

    def _restore_broker(
        self, ledger: JsonLedger, request: PaperRunRequest, restored_orders: Sequence[Order]
    ) -> tuple[PaperBrokerAdapter, BrokerSnapshot | None]:
        broker = PaperBrokerAdapter(initial_cash=request.initial_cash)
        snapshot = ledger.load_broker()
        if snapshot is not None:
            orders = {order.order_id: order for order in restored_orders}
            broker.restore(
                account=snapshot.account,
                positions={position.code: position for position in snapshot.positions},
                last_prices=snapshot.last_prices,
                orders=orders,
                open_orders=tuple(OpenOrderRecord.from_dict(item) for item in snapshot.open_orders),
            )
            _logger.info(
                "已恢复模拟盘账本：现金 %s，持仓 %s 只，在途委托 %s 笔",
                snapshot.account.cash,
                len(snapshot.positions),
                len(snapshot.open_orders),
            )
        broker.connect()
        return broker, snapshot

    def _prepare_data(
        self,
        local: LocalDataSource,
        request: PaperRunRequest,
        params: Mapping[str, Any],
    ) -> tuple[DataFrameBarProvider, FactorSnapshot | None]:
        """加载行情（含因子预热期）并构建快照。"""
        factors = request.factors or _declared_factors(params)
        warmup = max((get_factor(name).window for name in factors), default=0)
        calendar = local.trading_calendar()
        load_start = warmup_start(calendar, request.trade_date, warmup)

        codes = list(request.codes) if request.codes else self._all_codes(local, request.trade_date)
        frame = local.daily(codes, load_start, request.trade_date, adj="post")
        if frame.empty:
            raise DataNotReadyError(
                "交易日在本地无行情数据",
                detail={
                    "trade_date": request.trade_date.isoformat(),
                    "load_start": load_start.isoformat(),
                    "hint": "先执行 aimoney-data sync 同步到该交易日",
                },
            )
        snapshot = (
            build_snapshot_from_prices(frame, factors=factors, preprocess_steps=request.factor_preprocess)
            if factors
            else None
        )
        return DataFrameBarProvider(frame), snapshot

    def _all_codes(self, local: LocalDataSource, end: date) -> list[str]:
        info = local.stock_info()
        codes = info["code"].astype(str)
        if "delist_date" in info.columns:
            alive = info["delist_date"].isna() | (pd.to_datetime(info["delist_date"]) > pd.Timestamp(end))
            codes = codes[alive]
        return sorted(set(codes))

    # ============================================================ 闸门与落盘
    def _blocking_reason(self, gate: TradeGate, switch: KillSwitch) -> str | None:
        if switch.is_active():
            state = switch.state()
            return f"急停开关生效：{state.get('reason') or '未提供原因'}"
        if not gate.is_open():
            state = gate.state() if hasattr(gate, "state") else {}
            reason = state.get("reason") if isinstance(state, dict) else None
            return f"交易门禁未开启（{reason or '数据质量未通过或尚未校验'}）"
        return None

    def _persist(
        self, ledger: JsonLedger, broker: PaperBrokerAdapter, manager: OrderManager, trade_date: date
    ) -> None:
        account = broker.query_account()
        positions = broker.query_positions()
        ledger.save_broker(
            account=account,
            positions=positions,
            last_prices=broker.last_prices,
            open_orders=[record.to_dict() for record in broker.open_order_records()],
            last_trade_date=trade_date,
        )
        ledger.save_orders(manager.orders())

    def _append_nav(self, output_dir: Path, trade_date: date, account: Any) -> float:
        """净值历史追加到 JSONL（每天一行，便于直接画曲线）。"""
        nav = account.nav
        record = {
            "trade_date": trade_date.isoformat(),
            "total_asset": str(account.total_asset),
            "cash": str(account.cash),
            "market_value": str(account.market_value),
            "nav": nav,
        }
        with (output_dir / NAV_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return float(nav)

    def _output_dir(self, request: PaperRunRequest) -> Path:
        # 显式 str() 转换：Settings 在服务层是宽松类型，直接 Path(Any) 过不了严格模式的返回类型检查
        root = Path(str(self.settings.data.root_dir))
        return root / "paper" / request.strategy_dir.name

    def _skipped(
        self, request: PaperRunRequest, strategy_id: str, output_dir: Path, reason: str
    ) -> PaperRunReport:
        return PaperRunReport(
            trade_date=request.trade_date,
            strategy_id=strategy_id,
            output_dir=output_dir,
            signals=0,
            plans=0,
            submission=SubmissionResult(),
            positions=(),
            total_asset="0",
            nav=0.0,
            skipped_reason=reason,
        )

    def _write_report(self, report: PaperRunReport) -> None:
        report.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = report.output_dir / f"run_{report.trade_date:%Y%m%d}_{stamp}.json"
        path.write_text(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8")


def _declared_factors(params: Mapping[str, Any]) -> tuple[str, ...]:
    """策略在 config.yaml 里声明的因子（多因子策略依赖它，运行器据此准备快照）。"""
    raw = params.get("factors")
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, Sequence):
        return tuple(str(item) for item in raw)
    return ()


__all__ = ["NAV_FILE", "PaperRunReport", "PaperRunRequest", "PaperTradingService"]
