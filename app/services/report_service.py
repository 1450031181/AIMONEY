"""日/区间报告（P4-4）：把账本与成交流水变成"人能读的东西"。

数据来源全部是已落盘的事实，不做任何推断：

- ``paper_ledger.json``：账户、持仓、委托、成交流水（最新状态）；
- ``nav.jsonl``：逐日净值（区间绩效的唯一依据）。

**两处口径必须说清**（否则报告会"看起来对、实际错"）：

1. **当日成交按成交日期筛**：跨日重放成本才能算准已实现盈亏，所以归因始终基于**完整**流水，
   只把"展示范围"限制在当日/区间；
2. **区间绩效来自 nav.jsonl**：只有它记录了每日总资产；用"最新持仓 × 某日价格"反推历史净值
   会把期间已平仓的标的漏掉，得到的曲线是假的。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from engine.backtest.metrics import PerformanceMetrics, compute_metrics
from engine.errors import NotFoundError, ValidationError
from engine.report.attribution import PortfolioPnl, compute_attribution
from engine.trade.ledger import DEFAULT_LEDGER_FILE, JsonLedger
from engine.types import Account, Order, Position, Trade

_logger = logging.getLogger(__name__)

NAV_FILE = "nav.jsonl"
REPORTS_DIR = "reports"


@dataclass(frozen=True, slots=True)
class DailyReportRequest:
    strategy_dir: Path
    trade_date: date


@dataclass(frozen=True, slots=True)
class PeriodReportRequest:
    strategy_dir: Path
    start: date
    end: date
    risk_free_rate: float = 0.02


@dataclass
class DailyReport:
    trade_date: date
    strategy: str
    account: Account
    positions: tuple[Position, ...] = ()
    day_trades: tuple[Trade, ...] = ()
    day_orders: tuple[Order, ...] = ()
    attribution: PortfolioPnl = field(default_factory=PortfolioPnl)
    nav: float = 0.0
    #: 账本最后的处理日。与 ``trade_date`` 不同意味着「账户与持仓是最新状态，
    #: 而成交与委托是被请求那一天的」——混读会得出矛盾结论，必须在报告里说清。
    ledger_as_of: date | None = None
    output_dir: Path | None = None

    @property
    def stale(self) -> bool:
        """报告日之后账本又前进过：账户/持仓列示的不是当日状态。"""
        return self.ledger_as_of is not None and self.ledger_as_of != self.trade_date

    def _stale_note(self) -> list[str]:
        if not self.stale:
            return []
        return [
            "",
            f"> ⚠ 账本最新状态为 **{self.ledger_as_of}**。本报告中的「账户」与「持仓」是"
            f"**最新**状态，「当日成交」与「当日委托」才是 {self.trade_date} 的事实。"
            f"要看该日的准确账户，请在当天运行后立即生成报告，或改用区间报告看绩效。",
        ]

    @property
    def open_orders(self) -> tuple[Order, ...]:
        return tuple(order for order in self.day_orders if order.is_open)

    def summary(self) -> str:
        tail = f"（⚠ 账户为 {self.ledger_as_of} 的最新状态）" if self.stale else ""
        return (
            f"[{self.strategy}] {self.trade_date} 日报：净值 {self.nav:.4f}，"
            f"总资产 {self.account.total_asset}，持仓 {len(self.positions)} 只，"
            f"当日成交 {len(self.day_trades)} 笔、委托 {len(self.day_orders)} 笔"
            f"（在途 {len(self.open_orders)}）{tail}"
        )

    def to_markdown(self) -> str:
        lines = [
            f"# 模拟盘日报 {self.trade_date} — {self.strategy}",
            *self._stale_note(),
            "",
            "## 账户",
            "",
            "| 项目 | 数值 |",
            "| --- | --- |",
            f"| 现金 | {self.account.cash} |",
            f"| 持仓市值 | {self.account.market_value} |",
            f"| 总资产 | {self.account.total_asset} |",
            f"| 净值 | {self.nav:.4f} |",
            f"| 冻结资金 | {self.account.frozen_cash} |",
            f"| 持仓只数 | {len(self.positions)} |",
            f"| 在途委托 | {len(self.open_orders)} |",
            "",
            f"## 当日成交（{len(self.day_trades)} 笔）",
            "",
        ]
        if self.day_trades:
            lines += [
                "| 时间 | 标的 | 方向 | 数量 | 价格 | 金额 | 费用 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for trade in self.day_trades:
                lines.append(
                    f"| {trade.trade_time:%H:%M:%S} | {trade.code} | {trade.side.value} | {trade.qty} | "
                    f"{trade.price} | {trade.amount} | {trade.total_fee} |"
                )
        else:
            lines.append("（当日无成交）")

        lines += ["", f"## 当日委托（{len(self.day_orders)} 笔）", ""]
        if self.day_orders:
            lines += [
                "| 标的 | 方向 | 数量 | 已成 | 状态 | 说明 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for order in self.day_orders:
                lines.append(
                    f"| {order.code} | {order.side.value} | {order.qty} | {order.filled_qty} | "
                    f"{order.status.value} | {order.reject_reason or ''} |"
                )
        else:
            lines.append("（当日无委托）")

        lines += ["", f"## 持仓（{len(self.positions)} 只）", ""]
        if self.positions:
            lines += [
                "| 标的 | 数量 | 可卖 | 成本 | 最新价 | 市值 | 浮动盈亏 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for position in self.positions:
                lines.append(
                    f"| {position.code} | {position.qty} | {position.avail_qty} | {position.avg_cost} | "
                    f"{position.last_price} | {position.market_value} | {position.unrealized_pnl} |"
                )
        else:
            lines.append("（空仓）")

        lines += ["", "## 归因（累计）", ""]
        if self.attribution.symbols:
            lines += [
                "| 标的 | 已实现 | 浮动 | 合计 | 费用 | 成交笔数 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for book in sorted(self.attribution.symbols, key=lambda item: item.total_pnl, reverse=True):
                lines.append(
                    f"| {book.code} | {book.realized_pnl:+.2f} | {book.unrealized_pnl:+.2f} | "
                    f"{book.total_pnl:+.2f} | {book.fees:.2f} | {book.buy_count + book.sell_count} |"
                )
            lines += [
                "",
                f"合计：已实现 {self.attribution.realized_pnl:+.2f}，"
                f"浮动 {self.attribution.unrealized_pnl:+.2f}，"
                f"费用 {self.attribution.total_fees:.2f}"
                f"（占成交额 {self.attribution.fee_ratio:.3%}）",
                "",
                "> 口径差异：归因的浮动盈亏按**含费成本**计算（买入费用是真金白银），"
                "而上方「持仓」表的浮动盈亏按券商口径（成交均价不含费）。"
                "两者之差 = 该标的的买入费用，这是刻意的，不是算错。",
            ]
        else:
            lines.append("（尚无成交）")
        return "\n".join(lines) + "\n"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "strategy": self.strategy,
            "nav": self.nav,
            "ledger_as_of": self.ledger_as_of.isoformat() if self.ledger_as_of else None,
            "stale": self.stale,
            "account": {
                "cash": str(self.account.cash),
                "market_value": str(self.account.market_value),
                "total_asset": str(self.account.total_asset),
                "frozen_cash": str(self.account.frozen_cash),
                "initial_cash": str(self.account.initial_cash),
            },
            "positions": [
                {
                    "code": position.code,
                    "qty": position.qty,
                    "avail_qty": position.avail_qty,
                    "avg_cost": str(position.avg_cost),
                    "last_price": str(position.last_price),
                    "market_value": str(position.market_value),
                    "unrealized_pnl": str(position.unrealized_pnl),
                }
                for position in self.positions
            ],
            "trades": [
                {
                    "trade_time": trade.trade_time.isoformat(timespec="seconds"),
                    "code": trade.code,
                    "side": trade.side.value,
                    "qty": trade.qty,
                    "price": str(trade.price),
                    "amount": str(trade.amount),
                    "fees": str(trade.total_fee),
                }
                for trade in self.day_trades
            ],
            "orders": [
                {
                    "code": order.code,
                    "side": order.side.value,
                    "qty": order.qty,
                    "filled_qty": order.filled_qty,
                    "status": order.status.value,
                    "reason": order.reject_reason,
                }
                for order in self.day_orders
            ],
            "attribution": self.attribution.to_dict(),
        }


@dataclass
class PeriodReport:
    start: date
    end: date
    strategy: str
    metrics: PerformanceMetrics
    attribution: PortfolioPnl = field(default_factory=PortfolioPnl)
    #: 区间内有成交的标的（用于报告里筛展示行；归因本身始终按完整流水重放）
    period_codes: frozenset[str] = frozenset()
    trade_count: int = 0
    nav_points: int = 0
    output_dir: Path | None = None

    def shown_symbols(self) -> tuple[Any, ...]:
        """只展示"期间有成交"的标的；汇总行仍是**累计**口径（表头已注明）。"""
        if not self.period_codes:
            return tuple(self.attribution.symbols)
        return tuple(book for book in self.attribution.symbols if book.code in self.period_codes)

    def summary(self) -> str:
        metrics = self.metrics
        return (
            f"[{self.strategy}] {self.start}~{self.end} 区间报告（{self.nav_points} 个净值点）："
            f"总收益 {metrics.total_return:+.2%}，年化 {metrics.annual_return:+.2%}，"
            f"最大回撤 {metrics.max_drawdown:.2%}，夏普 {metrics.sharpe_ratio:.2f}，"
            f"成交 {self.trade_count} 笔，费用 {self.attribution.total_fees:.2f}"
        )

    def to_markdown(self) -> str:
        metrics = self.metrics
        lines = [
            f"# 区间报告 {self.start} ~ {self.end} — {self.strategy}",
            "",
            "## 绩效",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
            f"| 总收益 | {metrics.total_return:+.2%} |",
            f"| 年化收益 | {metrics.annual_return:+.2%} |",
            f"| 年化波动 | {metrics.annual_volatility:.2%} |",
            f"| 最大回撤 | {metrics.max_drawdown:.2%} |",
            f"| 回撤区间 | {metrics.max_drawdown_start or '-'} ~ {metrics.max_drawdown_end or '-'} |",
            f"| 夏普 | {metrics.sharpe_ratio:.2f} |",
            f"| 索提诺 | {metrics.sortino_ratio:.2f} |",
            f"| 卡玛 | {metrics.calmar_ratio:.2f} |",
            f"| 日胜率 | {metrics.daily_win_rate:.2%} |",
            f"| 净值观测点 | {self.nav_points} |",
            "",
            "## 交易与成本",
            "",
            "| 项目 | 数值 |",
            "| --- | --- |",
            f"| 成交笔数 | {self.trade_count} |",
            f"| 买入成交额 | {self.attribution.buy_turnover} |",
            f"| 卖出成交额 | {self.attribution.sell_turnover} |",
            f"| 费用合计 | {self.attribution.total_fees} |",
            f"| 费用 / 成交额 | {self.attribution.fee_ratio:.3%} |",
            "",
            "## 归因（累计口径，展示期间有成交的标的）",
            "",
        ]
        symbols = self.shown_symbols()
        if symbols:
            lines += [
                "| 标的 | 已实现 | 浮动 | 合计 | 费用 |",
                "| --- | --- | --- | --- | --- |",
            ]
            for book in sorted(symbols, key=lambda item: item.total_pnl, reverse=True):
                lines.append(
                    f"| {book.code} | {book.realized_pnl:+.2f} | {book.unrealized_pnl:+.2f} | "
                    f"{book.total_pnl:+.2f} | {book.fees:.2f} |"
                )
        else:
            lines.append("（区间内无成交）")
        if metrics.monthly_returns:
            lines += ["", "## 月度收益", "", "| 月份 | 收益 |", "| --- | --- |"]
            for month, value in sorted(metrics.monthly_returns.items()):
                lines.append(f"| {month} | {value:+.2%} |")
        return "\n".join(lines) + "\n"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "strategy": self.strategy,
            "nav_points": self.nav_points,
            "trade_count": self.trade_count,
            "metrics": self.metrics.to_jsonable(),
            "period_codes": sorted(self.period_codes),
            "attribution": self.attribution.to_dict(),
        }


class ReportService:
    """报告编排（只依赖 Settings）。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    # ============================================================ 日报
    def daily(self, request: DailyReportRequest) -> DailyReport:
        output_dir = self._paper_dir(request.strategy_dir)
        ledger = JsonLedger(output_dir / DEFAULT_LEDGER_FILE)
        snapshot = ledger.load_broker()
        if snapshot is None:
            raise NotFoundError(
                "没有模拟盘账本，无法生成日报",
                detail={
                    "ledger": str(output_dir / DEFAULT_LEDGER_FILE),
                    "hint": "先跑一次 aimoney-paper run（或 aimoney-schedule once --job post）",
                },
            )

        trades = ledger.load_trades()
        orders = ledger.load_orders()
        day_trades = tuple(trade for trade in trades if trade.trade_time.date() == request.trade_date)
        day_orders = tuple(order for order in orders if order.created_at.date() == request.trade_date)
        nav_records = self._read_nav(output_dir)
        nav = next(
            (float(item["nav"]) for item in nav_records if item.get("trade_date") == request.trade_date.isoformat()),
            snapshot.account.nav,
        )
        report = DailyReport(
            trade_date=request.trade_date,
            strategy=request.strategy_dir.name,
            account=snapshot.account,
            positions=tuple(snapshot.positions),
            day_trades=day_trades,
            day_orders=day_orders,
            attribution=compute_attribution(trades, snapshot.positions),
            nav=nav,
            ledger_as_of=snapshot.last_trade_date,
        )
        report.output_dir = self._write(request.strategy_dir, f"daily_{request.trade_date:%Y%m%d}", report)
        _logger.info("%s", report.summary())
        return report

    # ============================================================ 区间报告
    def period(self, request: PeriodReportRequest) -> PeriodReport:
        if request.end < request.start:
            raise ValidationError(
                "区间结束日不能早于开始日",
                detail={"start": request.start.isoformat(), "end": request.end.isoformat()},
            )
        output_dir = self._paper_dir(request.strategy_dir)
        nav_records = self._read_nav(output_dir)
        window = [
            item
            for item in nav_records
            if request.start.isoformat() <= str(item.get("trade_date")) <= request.end.isoformat()
        ]
        if len(window) < 2:
            raise ValidationError(
                "区间内净值记录不足，无法计算绩效（至少需要 2 个交易日）",
                detail={
                    "start": request.start.isoformat(),
                    "end": request.end.isoformat(),
                    "nav_points": len(window),
                    "hint": "检查区间是否落在模拟盘已运行的交易日内（aimoney-paper status）",
                },
            )
        nav_series = pd.Series(
            [float(item["nav"]) for item in window],
            index=pd.DatetimeIndex([pd.Timestamp(str(item["trade_date"])) for item in window]),
            dtype=float,
        )
        metrics = compute_metrics(nav_series, risk_free_rate=request.risk_free_rate)

        ledger = JsonLedger(output_dir / DEFAULT_LEDGER_FILE)
        trades = ledger.load_trades()
        snapshot = ledger.load_broker()
        attribution = compute_attribution(trades, snapshot.positions if snapshot else ())
        period_codes = frozenset(
            trade.code for trade in trades if request.start <= trade.trade_time.date() <= request.end
        )
        report = PeriodReport(
            start=request.start,
            end=request.end,
            strategy=request.strategy_dir.name,
            metrics=metrics,
            attribution=attribution,
            period_codes=period_codes,
            trade_count=sum(1 for trade in trades if request.start <= trade.trade_time.date() <= request.end),
            nav_points=len(window),
        )
        report.output_dir = self._write(
            request.strategy_dir, f"period_{request.start:%Y%m%d}_{request.end:%Y%m%d}", report
        )
        _logger.info("%s", report.summary())
        return report

    # ============================================================ 内部
    def _paper_dir(self, strategy_dir: Path) -> Path:
        return Path(str(self.settings.data.root_dir)) / "paper" / strategy_dir.name

    @staticmethod
    def _read_nav(output_dir: Path) -> list[dict[str, Any]]:
        path = output_dir / NAV_FILE
        if not path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _write(self, strategy_dir: Path, stem: str, report: DailyReport | PeriodReport) -> Path:
        """写 Markdown + JSON。**同名覆盖**：一天/一个区间只有一份报告，
        可反复生成；历史序列由 `nav.jsonl` 与成交流水承担，不靠报告文件堆积。"""
        directory = Path(str(self.settings.data.root_dir)) / REPORTS_DIR / strategy_dir.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{stem}.md").write_text(report.to_markdown(), encoding="utf-8")
        (directory / f"{stem}.json").write_text(
            json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory


__all__ = [
    "NAV_FILE",
    "REPORTS_DIR",
    "DailyReport",
    "DailyReportRequest",
    "PeriodReport",
    "PeriodReportRequest",
    "ReportService",
]
