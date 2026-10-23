"""P4-4 报告测试：日报、区间报告与 CLI。

报告的正确性不在于"能生成文件"，而在于**口径**：
当日成交按成交日期筛、区间绩效只能来自 `nav.jsonl`、归因必须按完整流水重放成本。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import yaml

from app.core.config import Settings
from app.report_cli import EXIT_FAILED, EXIT_OK, main
from app.services.paper_service import PaperRunRequest, PaperTradingService
from app.services.report_service import (
    DailyReportRequest,
    PeriodReportRequest,
    ReportService,
)
from engine.datahub.quality import TradeGate
from engine.datahub.store import ParquetStore
from engine.errors import NotFoundError, ValidationError
from engine.trade.ledger import DEFAULT_LEDGER_FILE, JsonLedger

DAY1 = date(2026, 10, 12)
DAY2 = date(2026, 10, 13)
DAY3 = date(2026, 10, 14)
CODES = ("600000", "000001")

_STRATEGY_BODY = """
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class HalfPosition(BaseStrategy):
    def on_bar(self, context, bars):
        if not bars:
            return []
        code = sorted(bars)[0]
        return [
            Signal(
                code=code,
                strategy_id=self.strategy_id,
                trade_date=context.trade_date,
                direction=Direction.BUY,
                target_weight=0.30,
                reason="固定 30% 仓位",
            )
        ]


def create_strategy(params):
    return HalfPosition(strategy_id=str(params.get("strategy_id", "half_position")), params=params)
"""


def _frame() -> pd.DataFrame:
    rows = []
    for offset in range(5):
        day = DAY1 + timedelta(days=offset)
        for index, code in enumerate(CODES):
            price = 10.0 + index * 5
            rows.append(
                {
                    "code": code,
                    "date": day.isoformat(),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 5_000_000,
                    "amount": price * 5_000_000,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    resolved = Settings(
        data={"root": str(tmp_path / "data")},
        risk={"max_weight": 0.5, "max_order_value": "5000000"},
    )
    resolved.ensure_dirs()
    ParquetStore(resolved.parquet_dir).write("daily", _frame(), data_version="20261014.1")
    reference = resolved.data.root_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"code": code, "name": f"标的{i}", "is_st": False, "list_date": "2010-01-01", "delist_date": None}
            for i, code in enumerate(CODES)
        ]
    ).to_parquet(reference / "stock_info.parquet", index=False)
    TradeGate(resolved.gate_file).open(as_of=DAY1, note="测试用")
    return resolved


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "strategies" / "half_position"
    directory.mkdir(parents=True)
    (directory / "strategy.py").write_text(_STRATEGY_BODY, encoding="utf-8")
    (directory / "config.yaml").write_text(yaml.safe_dump({"strategy_id": "half_position"}), encoding="utf-8")
    return directory


def _run(settings: Settings, strategy_dir: Path, days: tuple[date, ...]) -> None:
    service = PaperTradingService(settings)
    for day in days:
        service.run(PaperRunRequest(strategy_dir=strategy_dir, trade_date=day, codes=CODES))


def _report_files(settings: Settings, strategy_dir: Path) -> list[Path]:
    directory = Path(str(settings.data.root_dir)) / "reports" / strategy_dir.name
    return sorted(directory.glob("*")) if directory.is_dir() else []


# ============================================================ 日报
def test_日报含账户成交持仓与归因(settings: Settings, strategy_dir: Path) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2))

    report = ReportService(settings).daily(DailyReportRequest(strategy_dir=strategy_dir, trade_date=DAY2))

    assert report.account.total_asset > 0
    assert report.positions, "DAY2 应已成交建仓"
    assert len(report.day_trades) == 1, "DAY1 的挂单在 DAY2 成交"
    assert all(trade.trade_time.date() == DAY2 for trade in report.day_trades)
    assert report.attribution.trade_count == 1
    assert report.nav > 0

    markdown = report.to_markdown()
    for section in ("# 模拟盘日报", "## 账户", "## 当日成交", "## 持仓", "## 归因（累计）"):
        assert section in markdown


def test_日报只统计当日成交与当日委托(settings: Settings, strategy_dir: Path) -> None:
    """跨日口径：DAY1 只有委托没有成交，DAY2 才有成交。"""
    _run(settings, strategy_dir, (DAY1, DAY2))

    first = ReportService(settings).daily(DailyReportRequest(strategy_dir=strategy_dir, trade_date=DAY1))

    assert first.day_trades == (), "DAY1 不应有成交（信号次日才成交）"
    assert first.day_orders, "DAY1 应有委托"
    assert "（当日无成交）" in first.to_markdown()


def test_报告落盘且同名覆盖(settings: Settings, strategy_dir: Path) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2))
    service = ReportService(settings)

    service.daily(DailyReportRequest(strategy_dir=strategy_dir, trade_date=DAY2))
    first_files = _report_files(settings, strategy_dir)
    service.daily(DailyReportRequest(strategy_dir=strategy_dir, trade_date=DAY2))
    second_files = _report_files(settings, strategy_dir)

    assert {path.name for path in first_files} == {path.name for path in second_files}, "同一天只保留一份报告"
    assert any(path.suffix == ".md" for path in second_files)
    assert any(path.suffix == ".json" for path in second_files)
    payload = json.loads(next(path for path in second_files if path.suffix == ".json").read_text(encoding="utf-8"))
    assert payload["trade_date"] == DAY2.isoformat()
    assert payload["attribution"]["trade_count"] == 1


def test_补看历史某天时标注账本已前进(settings: Settings, strategy_dir: Path) -> None:
    """口径陷阱：账本只存最新状态，看历史某天时账户/持仓不是当天的，必须在报告里说清。"""
    _run(settings, strategy_dir, (DAY1, DAY2, DAY3))

    report = ReportService(settings).daily(DailyReportRequest(strategy_dir=strategy_dir, trade_date=DAY2))

    assert report.stale, "账本已推进到 DAY3"
    assert report.ledger_as_of == DAY3
    assert "账本最新状态为" in report.to_markdown()
    payload = report.to_jsonable()
    assert payload["stale"] is True and payload["ledger_as_of"] == DAY3.isoformat()


def test_最后一天的日报不标stale(settings: Settings, strategy_dir: Path) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2))

    report = ReportService(settings).daily(DailyReportRequest(strategy_dir=strategy_dir, trade_date=DAY2))

    assert not report.stale
    assert "账本最新状态为" not in report.to_markdown()


def test_没有账本时明确报错(settings: Settings, tmp_path: Path) -> None:
    empty_dir = tmp_path / "strategies" / "never_run"
    empty_dir.mkdir(parents=True)

    with pytest.raises(NotFoundError) as excinfo:
        ReportService(settings).daily(DailyReportRequest(strategy_dir=empty_dir, trade_date=DAY2))

    assert "账本" in str(excinfo.value)


# ============================================================ 区间报告
def test_区间报告给出绩效与成本(settings: Settings, strategy_dir: Path) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2, DAY3))

    report = ReportService(settings).period(
        PeriodReportRequest(strategy_dir=strategy_dir, start=DAY1, end=DAY3)
    )

    assert report.nav_points == 3
    assert report.trade_count >= 1
    assert report.metrics.total_return is not None
    assert report.attribution.total_fees >= 0
    markdown = report.to_markdown()
    for section in ("# 区间报告", "## 绩效", "## 交易与成本", "## 归因"):
        assert section in markdown


def test_区间净值不足时报错(settings: Settings, strategy_dir: Path) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2))

    with pytest.raises(ValidationError) as excinfo:
        ReportService(settings).period(
            PeriodReportRequest(strategy_dir=strategy_dir, start=DAY2, end=DAY2)
        )

    assert "净值记录不足" in str(excinfo.value)


def test_区间起止颠倒时报错(settings: Settings, strategy_dir: Path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ReportService(settings).period(
            PeriodReportRequest(strategy_dir=strategy_dir, start=DAY3, end=DAY1)
        )

    assert "不能早于" in str(excinfo.value)


# ============================================================ 成交流水
def test_成交流水落盘并可读回(settings: Settings, strategy_dir: Path) -> None:
    """回归：成交流水曾只存在内存里，重启后归因与对账明细全部丢失。"""
    _run(settings, strategy_dir, (DAY1, DAY2))

    ledger = JsonLedger(settings.data.root_dir / "paper" / strategy_dir.name / DEFAULT_LEDGER_FILE)
    trades = ledger.load_trades()

    assert len(trades) == 1
    trade = trades[0]
    assert trade.code in CODES
    assert trade.qty > 0 and trade.price > 0
    assert trade.trade_time.date() == DAY2


# ============================================================ CLI
def test_CLI日报输出markdown(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2))

    code = main(
        ["--root", str(settings.data.root_dir), "daily", "--strategy", str(strategy_dir), "--date", DAY2.isoformat()]
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "# 模拟盘日报" in out
    assert "报告目录" in out


def test_CLI日报json可解析(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2))

    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "daily",
            "--strategy",
            str(strategy_dir),
            "--date",
            DAY2.isoformat(),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["trade_date"] == DAY2.isoformat()
    assert "attribution" in payload


def test_CLI区间报告(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(settings, strategy_dir, (DAY1, DAY2, DAY3))

    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "period",
            "--strategy",
            str(strategy_dir),
            "--start",
            DAY1.isoformat(),
            "--end",
            DAY3.isoformat(),
        ]
    )

    assert code == EXIT_OK
    assert "# 区间报告" in capsys.readouterr().out


def test_CLI没有账本时退出码为1(settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty_dir = tmp_path / "strategies" / "never_run"
    empty_dir.mkdir(parents=True)

    code = main(
        ["--root", str(settings.data.root_dir), "daily", "--strategy", str(empty_dir), "--date", DAY2.isoformat()]
    )

    assert code == EXIT_FAILED
    assert "账本" in capsys.readouterr().err


def test_CLI日期无法解析时报错(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "period",
            "--strategy",
            str(strategy_dir),
            "--start",
            "not-a-date",
            "--end",
            DAY3.isoformat(),
        ]
    )

    assert code == EXIT_FAILED
    assert "日期" in capsys.readouterr().err


# ============================================================ 手工构造的小账本
def test_归因在报告中按完整流水重放成本(settings: Settings, strategy_dir: Path) -> None:
    """区间内只展示期间有成交的标的，但成本基础必须来自**完整**流水。"""
    _run(settings, strategy_dir, (DAY1, DAY2, DAY3))

    report = ReportService(settings).period(
        PeriodReportRequest(strategy_dir=strategy_dir, start=DAY2, end=DAY3)
    )

    assert report.period_codes, "DAY2 有成交，应被纳入展示范围"
    assert all(book.code in report.period_codes for book in report.shown_symbols())
    assert isinstance(report.attribution.realized_pnl, Decimal)
