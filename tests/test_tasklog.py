"""T-08 任务日志测试：JSONL 往返、异常记录、日志写入失败不影响业务。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from engine.datahub.tasklog import JsonlTaskLog, MemoryTaskLog, TaskLogger, TaskRecord, TaskStatus


def test_成功任务被记录并含耗时(tmp_path: Path) -> None:
    log = JsonlTaskLog(tmp_path / "logs" / "tasks.jsonl")

    with TaskLogger(log, "demo", trade_date=date(2026, 10, 14), detail={"step": 1}) as task:
        task.update(rows=10)

    record = log.history()[-1]
    assert record.status is TaskStatus.SUCCESS
    assert record.detail == {"step": 1, "rows": 10}
    assert record.trade_date == date(2026, 10, 14)
    assert record.duration_seconds is not None
    assert (tmp_path / "logs" / "tasks.jsonl").is_file()


def test_跳过状态与失败可区分(tmp_path: Path) -> None:
    log = MemoryTaskLog()

    with TaskLogger(log, "demo") as task:
        task.skip("非交易日")

    assert log.records[-1].status is TaskStatus.SKIPPED
    assert log.records[-1].detail["skip_reason"] == "非交易日"


def test_异常被记录但不会被吞掉(tmp_path: Path) -> None:
    log = MemoryTaskLog()

    with pytest.raises(ValueError), TaskLogger(log, "demo"):
        raise ValueError("业务出错")

    record = log.records[-1]
    assert record.status is TaskStatus.FAILED
    assert record.error is not None and "业务出错" in record.error


def test_JSONL可往返解析复杂字段(tmp_path: Path) -> None:
    log = JsonlTaskLog(tmp_path / "tasks.jsonl")

    with TaskLogger(log, "demo", trade_date=date(2026, 10, 14)) as task:
        task.update(codes=("600000", "000001"), nested={"a": 1}, ratio=0.5, nothing=None)

    record = log.history()[-1]
    assert record.detail["codes"] == ["600000", "000001"], "元组应序列化为列表"
    assert record.detail["nested"] == {"a": 1}
    assert record.detail["ratio"] == 0.5
    assert record.detail["nothing"] is None


def test_历史查询按名称过滤并限制条数(tmp_path: Path) -> None:
    log = JsonlTaskLog(tmp_path / "tasks.jsonl")
    for name in ("sync_daily", "sync_daily", "check_quality"):
        with TaskLogger(log, name):
            pass

    assert len(log.history("sync_daily")) == 2
    assert log.history("sync_daily", limit=1)[0].name == "sync_daily"
    assert len(log.history()) == 3


def test_日志文件不存在时历史为空(tmp_path: Path) -> None:
    assert JsonlTaskLog(tmp_path / "missing.jsonl").history() == []


def test_损坏行被跳过而不是整份日志不可用(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    log = JsonlTaskLog(path)
    with TaskLogger(log, "demo"):
        pass
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ 这不是合法 JSON\n")
    with TaskLogger(log, "demo"):
        pass

    assert len(log.history()) == 2, "坏行应被跳过，其余记录仍可读"


def test_任务日志写入失败不影响业务(tmp_path: Path) -> None:
    class BrokenLog:
        def append(self, record: TaskRecord) -> None:
            raise OSError("磁盘只读")

        def history(self, name: str | None = None, *, limit: int = 50) -> list[TaskRecord]:
            return []

    with TaskLogger(BrokenLog(), "demo") as task:
        task.update(ok=True)

    # 不抛异常即为通过：日志是旁路，绝不能因为它失败而让同步任务失败
