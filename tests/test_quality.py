"""T-11 数据质量与交易门禁测试。

最重要的两条：
1. **门禁默认关闭**（fail-safe）：全新部署没跑过质量校验时不允许交易；
2. **FAIL 即关门禁**：宁可当日空仓，也不拿脏数据下单。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from engine.datahub.quality import (
    QualityCheckTask,
    QualityStatus,
    QualityThresholds,
    Severity,
    TradeGate,
    check_quality,
)
from engine.datahub.store import ParquetStore
from engine.datahub.tasklog import MemoryTaskLog, TaskStatus

AS_OF = date(2026, 10, 14)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(code: str, close: float, *, volume: int = 100_000, day: str = "2026-10-14", suspended: bool = False) -> dict:
    return {
        "code": code,
        "date": day,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "adj_factor": 1.0,
        "is_suspended": suspended,
    }


def _clean_frame() -> pd.DataFrame:
    return _frame([_row("600000", 10.0), _row("600001", 20.0), _row("300750", 100.0)])


# ============================================================ 各条规则
def test_干净数据判定通过() -> None:
    report = check_quality(_clean_frame(), as_of=AS_OF, expected_codes=["600000", "600001", "300750"])

    assert report.status is QualityStatus.PASS
    assert report.findings == ()
    assert report.checked_rows == 3
    assert report.checked_codes == 3


def test_空数据直接失败() -> None:
    report = check_quality(pd.DataFrame(columns=["code", "date", "close"]), as_of=AS_OF)

    assert report.status is QualityStatus.FAIL
    assert report.failed_rules == ("empty_dataset",)


def test_非正价格判定失败() -> None:
    frame = _clean_frame()
    frame.loc[1, "close"] = 0.0
    report = check_quality(frame, as_of=AS_OF)

    assert report.status is QualityStatus.FAIL
    assert "non_positive_price" in report.failed_rules


def test_重复主键判定失败() -> None:
    frame = pd.concat([_clean_frame(), _clean_frame().iloc[[0]]], ignore_index=True)
    report = check_quality(frame, as_of=AS_OF)

    assert report.status is QualityStatus.FAIL
    assert "duplicate_primary_key" in report.failed_rules


def test_缺失率超限判定失败并给出样本() -> None:
    frame = _frame([_row("600000", 10.0)])
    report = check_quality(
        frame,
        as_of=AS_OF,
        expected_codes=[f"60000{i}" for i in range(10)],
        thresholds=QualityThresholds(max_missing_ratio=0.01),
    )

    finding = next(item for item in report.findings if item.rule == "missing_codes")
    assert finding.severity is Severity.FAIL
    assert finding.count == 9
    assert finding.samples[0]["code"] == "600001"


def test_未提供期望标的时缺失率检查标注未生效() -> None:
    report = check_quality(_clean_frame(), as_of=AS_OF)

    assert "缺失率检查（未提供期望标的列表）" in report.rules_not_applied
    assert report.status is QualityStatus.PASS, "规则没生效不等于数据有问题"


def test_创业板20个点正常波动不误报() -> None:
    frame = _frame(
        [
            _row("300750", 100.0, day="2026-10-13"),
            _row("300750", 120.0),  # +20%，创业板合法
        ]
    )
    assert check_quality(frame, as_of=AS_OF).status is QualityStatus.PASS


def test_未复权造成的跳变判定失败() -> None:
    """10 派 3 未乘因子 → 约 -23% 的假跌幅，主板不可能出现。"""
    frame = _frame([_row("600000", 10.0, day="2026-10-13"), _row("600000", 7.7)])
    report = check_quality(frame, as_of=AS_OF)

    assert report.status is QualityStatus.FAIL
    assert "price_continuity" in report.failed_rules


def test_非停牌零成交只告警不阻断() -> None:
    frame = _frame([_row("600000", 10.0), _row("600001", 20.0, volume=0)])

    report = check_quality(frame, as_of=AS_OF)

    assert report.status is QualityStatus.WARN
    assert report.warned_rules == ("zero_volume_not_suspended",)
    assert report.failed_rules == ()


def test_停牌零成交不告警() -> None:
    frame = _frame([_row("600000", 10.0, volume=0, suspended=True)])
    assert check_quality(frame, as_of=AS_OF).status is QualityStatus.PASS


def test_成交量突增告警() -> None:
    rows = [_row("600000", 10.0, volume=100_000, day=f"2026-09-{day:02d}") for day in range(1, 11)]
    rows.append(_row("600000", 10.0, volume=5_000_000))  # 50 倍
    report = check_quality(_frame(rows), as_of=AS_OF, thresholds=QualityThresholds(max_volume_jump_ratio=10.0))

    assert report.status is QualityStatus.WARN
    assert "volume_jump" in report.warned_rules


def test_报告可序列化且摘要含关键信息() -> None:
    frame = _clean_frame()
    frame.loc[0, "close"] = -1.0
    report = check_quality(frame, as_of=AS_OF, expected_codes=["600000"])

    payload = report.to_jsonable()
    assert payload["status"] == "FAIL"
    assert "non_positive_price" in payload["failed_rules"]
    assert json.dumps(payload, ensure_ascii=False)
    assert "FAIL" in report.summary()


def test_报告含pandas时间戳等标量时仍可序列化() -> None:
    """回归：从 Parquet 读出的 date 是 Timestamp，曾经让报告写盘直接 TypeError。"""
    frame = _clean_frame()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["volume"] = frame["volume"].astype("int64")
    frame.loc[0, "close"] = -1.0

    report = check_quality(frame, as_of=AS_OF)
    payload = report.to_jsonable()

    assert json.dumps(payload, ensure_ascii=False)
    assert isinstance(payload["findings"][0]["samples"][0]["date"], str)


# ============================================================ 交易门禁
def test_门禁默认关闭(tmp_path: Path) -> None:
    gate = TradeGate(tmp_path / "gate.json")

    assert not gate.is_open(), "全新部署必须先跑质量校验才开门（fail-safe）"
    assert gate.state()["as_of"] is None


def test_门禁开关与持久化(tmp_path: Path) -> None:
    gate = TradeGate(tmp_path / "gate.json")
    gate.open(as_of=AS_OF)
    assert gate.is_open()
    assert TradeGate(tmp_path / "gate.json").is_open(), "状态必须落盘，重启后仍有效"

    gate.close(as_of=AS_OF, reason="数据质量 FAIL：empty_dataset")
    assert not gate.is_open()
    assert "数据质量" in str(gate.state()["reason"])


# ============================================================ 质量校验任务
def _task(tmp_path: Path, *, log: MemoryTaskLog | None = None) -> tuple[QualityCheckTask, ParquetStore, TradeGate]:
    store = ParquetStore(tmp_path / "parquet")
    gate = TradeGate(tmp_path / "state" / "gate.json")
    task = QualityCheckTask(
        store=store,
        gate=gate,
        task_log=log or MemoryTaskLog(),
        report_dir=tmp_path / "quality",
    )
    return task, store, gate


def test_质量通过时开门禁并落报告(tmp_path: Path) -> None:
    log = MemoryTaskLog()
    task, store, gate = _task(tmp_path, log=log)
    store.write("daily", _clean_frame())

    report = task.run(as_of=AS_OF, expected_codes=["600000", "600001", "300750"])

    assert report.status is QualityStatus.PASS
    assert gate.is_open()
    report_file = tmp_path / "quality" / "quality_20261014.json"
    assert report_file.is_file()
    assert json.loads(report_file.read_text(encoding="utf-8"))["status"] == "PASS"
    assert log.records[-1].detail["gate_open"] is True


def test_质量失败时关门禁(tmp_path: Path) -> None:
    task, store, gate = _task(tmp_path)
    gate.open(as_of=date(2026, 10, 13))  # 昨天是开的
    bad = _clean_frame()
    bad.loc[0, "close"] = 0.0
    store.write("daily", bad)

    report = task.run(as_of=AS_OF)

    assert report.status is QualityStatus.FAIL
    assert not gate.is_open(), "FAIL 必须关门禁"


def test_当日无数据时关门禁(tmp_path: Path) -> None:
    task, _, gate = _task(tmp_path)
    gate.open(as_of=date(2026, 10, 13))

    report = task.run(as_of=AS_OF)

    assert report.status is QualityStatus.FAIL
    assert report.failed_rules == ("empty_dataset",)
    assert not gate.is_open()


def test_告警级别不关门禁(tmp_path: Path) -> None:
    task, store, gate = _task(tmp_path)
    store.write("daily", _frame([_row("600000", 10.0), _row("600001", 20.0, volume=0)]))

    report = task.run(as_of=AS_OF)

    assert report.status is QualityStatus.WARN
    assert gate.is_open(), "WARN 只告警，不阻断交易"


def test_任务日志记录校验结果(tmp_path: Path) -> None:
    log = MemoryTaskLog()
    task, store, _ = _task(tmp_path, log=log)
    store.write("daily", _clean_frame())

    task.run(as_of=AS_OF)

    record = log.records[-1]
    assert record.status is TaskStatus.SUCCESS
    assert record.detail["checked_rows"] == 3
    assert record.detail["status"] == "PASS"
    assert "缺失率检查" in " ".join(record.detail["rules_not_applied"])
