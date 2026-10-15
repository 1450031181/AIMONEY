"""P4-3 盘前/盘后调度测试。

重点验证三条"出错不会有人立刻发现"的行为：

1. **非交易日不得触发任何任务**（否则日志被噪声淹没，真问题被埋掉）；
2. **同步失败不能阻断质量校验**——否则门禁停留在旧状态，出现"数据坏了却还开着闸门"
   这种最危险的组合；
3. **质量 FAIL 不算调度失败**：检查本身成功执行了，结论是"今天不该交易"，
   真正的执行故障不该被它刷屏掩盖。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from app.core.config import Settings
from app.schedule_cli import EXIT_FAILED, EXIT_OK, EXIT_SKIPPED, main
from app.services.schedule_service import JobKind, ScheduleService, StepStatus, ensure_strategy_dir
from engine.datahub.calendar import TradingCalendarImpl
from engine.datahub.quality import TradeGate
from engine.datahub.store import ParquetStore
from engine.datasource.memory import InMemoryDataSource
from engine.errors import DataNotReadyError, DataSourceError, ValidationError

TRADE_DAY = date(2026, 10, 15)  # 周四
WEEKEND = date(2026, 10, 17)  # 周六
CODES = ("600000", "000001")
SESSION_DATES = tuple(
    date(2026, 10, 12) + timedelta(days=offset)
    for offset in range(5)
    if (date(2026, 10, 12) + timedelta(days=offset)).weekday() < 5
)

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
    for day in SESSION_DATES:
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


def _source() -> InMemoryDataSource:
    source = InMemoryDataSource(calendar=TradingCalendarImpl.from_dates(SESSION_DATES))
    source.add_daily(_frame())
    return source


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    resolved = Settings(
        data={"root": str(tmp_path / "data")},
        risk={"max_weight": 0.5, "max_order_value": "5000000"},
    )
    resolved.ensure_dirs()
    # 交易日历由 daily 数据集反推（数据集覆盖到哪，日历就到哪）
    ParquetStore(resolved.parquet_dir).write("daily", _frame(), data_version="20261015.1")
    reference = resolved.data.root_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"code": code, "name": f"标的{i}", "is_st": False, "list_date": "2010-01-01", "delist_date": None}
            for i, code in enumerate(CODES)
        ]
    ).to_parquet(reference / "stock_info.parquet", index=False)
    TradeGate(resolved.gate_file).open(as_of=TRADE_DAY, note="测试用")
    return resolved


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "strategies" / "half_position"
    directory.mkdir(parents=True)
    (directory / "strategy.py").write_text(_STRATEGY_BODY, encoding="utf-8")
    (directory / "config.yaml").write_text(yaml.safe_dump({"strategy_id": "half_position"}), encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def _offline_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 用例不得触网：把调度 CLI 的数据源工厂换成内存替身。

    CLI 生产路径会真实拉取交易日历（这是刻意的：数据源才是交易日权威），
    但测试必须离线——否则用例会依赖网络、还会为限流重试白白等上好几秒。
    """
    from app import schedule_cli

    monkeypatch.setattr(schedule_cli, "_source_factory", lambda settings: _source)


def _service(settings: Settings, *, strategy_dir: Path | None = None, source: Any = None) -> ScheduleService:
    return ScheduleService(
        settings=settings,
        source_factory=(lambda: source) if source is not None else None,
        strategy_dir=strategy_dir,
        codes=CODES,
    )


# ============================================================ 日历守卫
def test_非交易日不触发任何任务(settings: Settings, strategy_dir: Path) -> None:
    report = _service(settings, strategy_dir=strategy_dir, source=_source()).run(
        JobKind.POST_MARKET, day=WEEKEND
    )

    assert report.skipped
    assert report.steps == [], "守卫必须拦在最前面：连数据同步都不该触发"
    assert "非交易日" in (report.skip_reason or "")


def test_非交易日退出码为跳过(settings: Settings) -> None:
    code = main(["--root", str(settings.data.root_dir), "once", "--job", "pre", "--date", WEEKEND.isoformat()])

    assert code == EXIT_SKIPPED


def test_计划触发日只含交易日(settings: Settings) -> None:
    service = _service(settings, source=_source())

    days = service.upcoming_run_days(start=date(2026, 10, 12), days=5)

    assert all(day.weekday() < 5 for day in days), "周末绝不能被列进计划"
    assert days == SESSION_DATES


def test_无本地数据时明确报错而不是当作非交易日(settings: Settings, tmp_path: Path) -> None:
    """最危险的情况：没有行情/日历 → 每天都算"非交易日" → 永远不交易且无人察觉。

    所以这里必须**抛错**（带同步指引），而不是安静地返回 False。
    """
    empty = Settings(data={"root": str(tmp_path / "empty_data")})
    empty.ensure_dirs()

    with pytest.raises(DataNotReadyError):
        ScheduleService(settings=empty).is_trading_day(TRADE_DAY)


# ============================================================ 盘前链
def test_盘前同步与校验都执行(settings: Settings) -> None:
    report = _service(settings, source=_source()).run(JobKind.PRE_MARKET, day=TRADE_DAY)

    assert [step.name for step in report.steps] == ["sync", "check"]
    assert report.ok
    assert report.step("sync").status is StepStatus.OK
    # 质量校验：数据干净时门禁保持开启
    assert TradeGate(settings.gate_file).is_open()


def test_质量FAIL时门禁关闭但调度仍算成功(settings: Settings) -> None:
    """质量 FAIL 是"检查成功、结论是不该交易"，门禁关闭已拦住交易，调度层不必再报错。"""
    broken = Settings(data={"root": str(settings.data.root_dir)})
    frame = _frame()
    # 必须破坏**被检查的那一天**的数据：质量规则只判定 as_of 当日的切片
    frame.loc[(frame["date"] == TRADE_DAY.isoformat()) & (frame["code"] == CODES[0]), "close"] = -1.0
    ParquetStore(broken.parquet_dir).write("daily", frame, data_version="20261015.2")
    TradeGate(broken.gate_file).open(as_of=TRADE_DAY, note="先前通过")

    report = _service(broken, source=_source()).run(JobKind.PRE_MARKET, day=TRADE_DAY)

    assert report.ok, "调度本身成功（检查跑完了）"
    assert "FAIL" in (report.step("check").detail or "")
    assert not TradeGate(broken.gate_file).is_open(), "FAIL 必须关闭门禁"


def test_同步失败仍继续质量校验并让退出码非零(settings: Settings, tmp_path: Path) -> None:
    class _Broken(InMemoryDataSource):
        def daily(self, codes: list[str], start: date, end: date, *, adj: str = "post") -> pd.DataFrame:  # type: ignore[override]
            raise DataSourceError("模拟数据源不可用")

    # 本地数据只覆盖到 10-14：这样盘前同步会**真的**去取 10-15
    # （若本地已是最新，同步会幂等跳过，失败路径根本测不到）
    lagging = Settings(data={"root": str(tmp_path / "lag_data")})
    lagging.ensure_dirs()
    frame = _frame()
    ParquetStore(lagging.parquet_dir).write(
        "daily", frame[frame["date"] < TRADE_DAY.isoformat()], data_version="20261014.1"
    )
    TradeGate(lagging.gate_file).open(as_of=date(2026, 10, 14), note="先前通过")

    report = _service(lagging, source=_Broken(calendar=TradingCalendarImpl.from_dates(SESSION_DATES))).run(
        JobKind.PRE_MARKET, day=TRADE_DAY
    )

    assert report.step("sync").status is StepStatus.FAILED
    check = report.step("check")
    assert check is not None and check.status is not StepStatus.SKIPPED, (
        "同步失败不能阻断门禁更新（否则闸门停在旧状态，出现『数据坏了却还开着闸门』）"
    )
    assert not report.ok, "整体仍应标记失败，交给调度器告警"


def test_盘前同步在本地已最新时算成功(settings: Settings) -> None:
    """幂等空转不是故障：否则定时任务从第二天起每天都误告警，监控很快被脱敏。"""
    report = _service(settings, source=_source()).run(JobKind.PRE_MARKET, day=TRADE_DAY)

    assert report.step("sync").status is StepStatus.OK
    assert "无需同步" in (report.step("sync").detail or "")
    assert report.ok


def test_日历以数据源为准而非本地行情(settings: Settings, tmp_path: Path) -> None:
    """回归：本地行情推导的日历看不到"今天"（盘前还没同步），会让盘前任务永不触发（死锁）。"""
    lagging = Settings(data={"root": str(tmp_path / "lag_data")})
    lagging.ensure_dirs()
    frame = _frame()
    ParquetStore(lagging.parquet_dir).write(
        "daily", frame[frame["date"] < TRADE_DAY.isoformat()], data_version="20261014.1"
    )

    # 本地日历：10-15 不在其中
    local_only = ScheduleService(settings=lagging)
    assert not local_only.is_trading_day(TRADE_DAY), "本地行情里确实没有 10-15"

    # 注入数据源后：10-15 是交易日（否则盘前任务永远不会在 10-15 触发）
    with_source = ScheduleService(settings=lagging, source_factory=_source)
    assert with_source.is_trading_day(TRADE_DAY)


# ============================================================ 盘后链
def test_盘后跑策略并校准(settings: Settings, strategy_dir: Path) -> None:
    report = _service(settings, strategy_dir=strategy_dir, source=_source()).run(
        JobKind.POST_MARKET, day=TRADE_DAY
    )

    assert [step.name for step in report.steps] == ["paper", "reconcile"]
    assert report.ok, report.summary()
    assert "报单" in (report.step("paper").detail or "")


def test_盘后未指定策略时全部跳过(settings: Settings) -> None:
    report = _service(settings, source=_source()).run(JobKind.POST_MARKET, day=TRADE_DAY)

    assert report.skipped
    assert all(step.status is StepStatus.SKIPPED for step in report.steps)


def test_盘后重复运行被日级防重跳过(settings: Settings, strategy_dir: Path) -> None:
    service = _service(settings, strategy_dir=strategy_dir, source=_source())
    first = service.run(JobKind.POST_MARKET, day=TRADE_DAY)

    second = service.run(JobKind.POST_MARKET, day=TRADE_DAY)

    assert first.ok
    assert second.step("paper").status is StepStatus.SKIPPED
    assert "已处理过" in (second.step("paper").detail or "")
    # 校准是幂等的，重复跑不应产生任何"纠正/废单"
    assert second.step("reconcile").status is StepStatus.OK
    assert "纠正 0 笔" in (second.step("reconcile").detail or "")


def test_校准不会把在途委托误判为废单(settings: Settings, strategy_dir: Path) -> None:
    """回归：paper broker 的 restore 曾漏恢复委托簿，导致校准把在途委托全部判成废单。"""
    service = _service(settings, strategy_dir=strategy_dir, source=_source())
    service.run(JobKind.POST_MARKET, day=TRADE_DAY)

    report = service.run(JobKind.POST_MARKET, day=TRADE_DAY)

    detail = report.step("reconcile").detail or ""
    assert "废单 0 笔" in detail, f"在途委托不该被误判：{detail}"


# ============================================================ CLI
def test_CLI盘前任务(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "once",
            "--job",
            "pre",
            "--date",
            TRADE_DAY.isoformat(),
            "--codes",
            ",".join(CODES),  # 不传则按"全市场"同步，需要数据源提供股票列表
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK, payload["steps"]  # 失败时直接打印各步骤，省得再猜
    assert payload["status"] == "OK"
    assert [step["name"] for step in payload["steps"]] == ["sync", "check"]


def test_CLI策略目录不存在时退出码为1(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "once",
            "--job",
            "post",
            "--date",
            TRADE_DAY.isoformat(),
            "--strategy",
            str(settings.data.root_dir / "not-exists"),
        ]
    )

    assert code == EXIT_FAILED
    assert "策略目录不存在" in capsys.readouterr().err


def test_CLI查看计划触发日(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "plan",
            "--start",
            SESSION_DATES[0].isoformat(),
            "--days",
            "5",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["days"] == [day.isoformat() for day in SESSION_DATES]


def test_CLI参数格式非法时报错(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "once",
            "--job",
            "post",
            "--date",
            TRADE_DAY.isoformat(),
            "--params",
            "no_equals_sign",
        ]
    )

    assert code == EXIT_FAILED
    assert "key=value" in capsys.readouterr().err


# ============================================================ 辅助
def test_策略目录校验() -> None:
    with pytest.raises(ValidationError):
        ensure_strategy_dir(None)
    with pytest.raises(ValidationError):
        ensure_strategy_dir("not-a-dir")


def test_摘要可读(settings: Settings) -> None:
    report = _service(settings, source=_source()).run(JobKind.PRE_MARKET, day=TRADE_DAY)

    summary = report.summary()

    assert "sync ✓" in summary and "check ✓" in summary
    assert report.steps and all(step.duration_ms >= 0 for step in report.steps)


def test_账本现金为十进制而非浮点(settings: Settings, strategy_dir: Path) -> None:
    """账本落盘的金额一律以字符串保存（浮点会在多日复利后累积误差）。"""
    _service(settings, strategy_dir=strategy_dir, source=_source()).run(JobKind.POST_MARKET, day=TRADE_DAY)

    ledger_file = settings.data.root_dir / "paper" / strategy_dir.name / "paper_ledger.json"
    payload = json.loads(ledger_file.read_text(encoding="utf-8"))

    assert isinstance(payload["account"]["cash"], str)
    assert Decimal(payload["account"]["cash"]) > 0
