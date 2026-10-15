"""任务日志（T-08）。

为什么单独做一层：盘前任务链一旦失败，当天就不该交易。要能回答"**哪个任务、
在哪一步、什么时候、因为什么**失败"，而不是只看到一个 traceback。

- ``TaskLogger`` 用上下文管理器自动记录开始/结束/耗时/异常，业务代码只写核心逻辑；
- **不吞异常**（``__exit__`` 返回 False）：日志是旁路，不能改变控制流；
- P0 落 JSONL（零依赖、append 安全）；后续接 PostgreSQL ``task_log`` 表时替换 ``TaskLogStore`` 实现即可。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol

from engine.datahub.jsonio import to_jsonable
from engine.ids import new_task_id

_logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    name: str
    status: TaskStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    trade_date: date | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": str(self.status),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": self.finished_at.isoformat(timespec="seconds") if self.finished_at else None,
            "duration_seconds": self.duration_seconds,
            "trade_date": self.trade_date.isoformat() if self.trade_date else None,
            "detail": to_jsonable(self.detail),
            "error": self.error,
        }

    @classmethod
    def from_jsonable(cls, payload: Mapping[str, Any]) -> TaskRecord:
        trade_date = payload.get("trade_date")
        finished_at = payload.get("finished_at")
        return cls(
            task_id=str(payload.get("task_id", "")),
            name=str(payload.get("name", "")),
            status=TaskStatus(str(payload.get("status", TaskStatus.RUNNING))),
            started_at=datetime.fromisoformat(str(payload.get("started_at"))),
            finished_at=datetime.fromisoformat(str(finished_at)) if finished_at else None,
            duration_seconds=payload.get("duration_seconds"),
            trade_date=date.fromisoformat(str(trade_date)) if trade_date else None,
            detail=payload.get("detail") or {},
            error=payload.get("error"),
        )


class TaskLogStore(Protocol):
    def append(self, record: TaskRecord) -> None: ...

    def history(self, name: str | None = None, *, limit: int = 50) -> list[TaskRecord]: ...


class JsonlTaskLog:
    """JSONL 任务日志（每行一条记录，append 即可，天然可被后续解析/采集）。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, record: TaskRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_jsonable(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def history(self, name: str | None = None, *, limit: int = 50) -> list[TaskRecord]:
        if not self.path.is_file():
            return []
        records: list[TaskRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    _logger.warning("任务日志存在无法解析的行，已跳过：%s", self.path)
                    continue
                record = TaskRecord.from_jsonable(payload)
                if name is None or record.name == name:
                    records.append(record)
        return records[-limit:]


class MemoryTaskLog:
    """内存任务日志（测试与单次运行场景）。"""

    def __init__(self) -> None:
        self.records: list[TaskRecord] = []

    def append(self, record: TaskRecord) -> None:
        self.records.append(record)

    def history(self, name: str | None = None, *, limit: int = 50) -> list[TaskRecord]:
        matched = [record for record in self.records if name is None or record.name == name]
        return matched[-limit:]


class TaskLogger:
    """任务上下文管理器：自动记录开始/结果/耗时/异常，并允许中途补充细节。"""

    def __init__(
        self,
        store: TaskLogStore,
        name: str,
        *,
        trade_date: date | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.name = name
        self.task_id = new_task_id()
        self.trade_date = trade_date
        self.started_at = datetime.now()
        self.detail: dict[str, Any] = dict(detail or {})
        self._status = TaskStatus.RUNNING
        self._monotonic_start = 0.0

    # -------------------------------------------------- 上下文协议
    def __enter__(self) -> TaskLogger:
        import time

        self._monotonic_start = time.monotonic()
        _logger.info("任务开始：%s", self.name, extra={"task_id": self.task_id})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        import time

        finished_at = datetime.now()
        if exc is not None:
            status = TaskStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
        # 无异常时：显式 skip 过则记为 SKIPPED，否则就是 SUCCESS
        elif self._status is TaskStatus.RUNNING:
            status = TaskStatus.SUCCESS
            error = None
        else:
            status = self._status
            error = None
        record = TaskRecord(
            task_id=self.task_id,
            name=self.name,
            status=status,
            started_at=self.started_at,
            finished_at=finished_at,
            duration_seconds=round(time.monotonic() - self._monotonic_start, 3),
            trade_date=self.trade_date,
            detail=dict(self.detail),
            error=error,
        )
        try:
            self.store.append(record)
        # 刻意兜底所有异常：日志是旁路，写入失败绝不能改变任务本身的结果
        except Exception:
            _logger.exception("任务日志写入失败（不影响任务结果）：%s", self.name)
        if status is TaskStatus.FAILED:
            _logger.error("任务失败：%s | %s", self.name, error)
        else:
            _logger.info("任务结束：%s | %s | %ss", self.name, status, record.duration_seconds)
        return False

    # -------------------------------------------------- 记录细节
    def update(self, **detail: Any) -> None:
        self.detail.update(detail)

    def skip(self, reason: str) -> None:
        """标记为跳过（非交易日、已是最新等）。跳过不是失败，不应告警。"""
        self._status = TaskStatus.SKIPPED
        self.detail.setdefault("skip_reason", reason)

    @property
    def status(self) -> TaskStatus:
        return self._status





__all__ = [
    "JsonlTaskLog",
    "MemoryTaskLog",
    "TaskLogStore",
    "TaskLogger",
    "TaskRecord",
    "TaskStatus",
]
