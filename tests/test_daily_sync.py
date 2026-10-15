"""T-08 日线同步任务测试（全离线）。

重点验证四件"出事就很难查"的事：
1. 增量续传：第二次同步只为增量区间请求数据，不重拉历史；
2. 单标的失败不拖垮整批，且成功部分已落盘（不是攒到最后一起丢）；
3. 补偿轮重试后仍失败的标的进 ``failed``，供告警与门禁使用；
4. 非交易日 / 已是最新必须"跳过"而不是"失败"——跳过不该触发告警。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from engine.datahub.calendar import weekday_calendar
from engine.datahub.store import ParquetStore
from engine.datahub.sync import DailySyncTask
from engine.datahub.tasklog import MemoryTaskLog, TaskStatus
from engine.datasource.memory import InMemoryDataSource
from engine.errors import DataSourceError, ValidationError

TRADE_DAY = date(2026, 10, 14)
START = date(2026, 10, 12)


class RecordingSource(InMemoryDataSource):
    """在内存源基础上记录每次请求的区间，并支持按标的注入失败。"""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.requests: list[tuple[tuple[str, ...], date, date]] = []
        self.fail_codes: dict[str, int] = {}

    def daily(
        self,
        codes: list[str],  # type: ignore[override]
        start: date,
        end: date,
        *,
        adj: str = "post",  # type: ignore[override]
    ) -> pd.DataFrame:
        self.requests.append((tuple(codes), start, end))
        for code in codes:
            remaining = self.fail_codes.get(code, 0)
            if remaining > 0:
                self.fail_codes[code] = remaining - 1
                raise DataSourceError(f"{code} 拉取失败", detail={"code": code})
        return super().daily(codes, start, end, adj=adj)  # type: ignore[arg-type]


def _source(*, fail_codes: dict[str, int] | None = None) -> RecordingSource:
    source = RecordingSource(calendar=weekday_calendar(date(2026, 1, 1), date(2026, 12, 31)))
    rows = [
        {
            "code": code,
            "date": day,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100_000,
        }
        for code in ("600000", "000001")
        for day in ("2026-10-12", "2026-10-13", "2026-10-14")
    ]
    source.add_daily(pd.DataFrame(rows))
    source.add_stock_info(pd.DataFrame([{"code": "600000", "name": "浦发银行"}, {"code": "000001", "name": "平安银行"}]))
    if fail_codes:
        source.fail_codes = dict(fail_codes)
    return source


def _task(tmp_path: Path, source: RecordingSource, *, log: MemoryTaskLog | None = None, **kwargs: object) -> DailySyncTask:
    return DailySyncTask(
        source=source,
        store=ParquetStore(tmp_path / "parquet"),
        task_log=log or MemoryTaskLog(),
        calendar=source.calendar,
        **kwargs,  # type: ignore[arg-type]
    )


def test_首次同步写入全部标的(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source)

    # 显式传 trade_date：数据版本号按锚定日期生成，不传会依赖"今天是几号"，用例就不稳定了
    result = task.run(codes=["600000", "000001"], start=START, end=TRADE_DAY, trade_date=TRADE_DAY)

    assert result.ok
    assert result.rows_written == 6
    assert task.store.coverage("daily").code_count == 2
    assert result.data_version == "20261014.1"


def test_增量同步只为增量区间请求数据(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source)
    task.run(codes=["600000"], start=START, end=date(2026, 10, 13))
    source.requests.clear()

    result = task.run(codes=["600000"], end=TRADE_DAY)

    assert result.start == date(2026, 10, 14), "起点应为本地最大日期的次日"
    assert source.requests == [(("600000",), date(2026, 10, 14), TRADE_DAY)]
    assert result.rows_written == 1


def test_已是最新时跳过而不是失败(tmp_path: Path) -> None:
    source = _source()
    log = MemoryTaskLog()
    task = _task(tmp_path, source, log=log)
    task.run(codes=["600000"], start=START, end=TRADE_DAY)

    result = task.run(codes=["600000"], end=TRADE_DAY)

    assert result.skipped
    assert "已是最新" in result.skip_reason
    assert log.records[-1].status is TaskStatus.SKIPPED, "跳过不该记为失败（否则天天误告警）"


def test_非交易日跳过(tmp_path: Path) -> None:
    source = _source()
    log = MemoryTaskLog()
    task = _task(tmp_path, source, log=log)

    result = task.run(codes=["600000"], start=START, end=date(2026, 10, 17), trade_date=date(2026, 10, 17))

    assert result.skipped
    assert "非交易日" in result.skip_reason
    assert log.records[-1].status is TaskStatus.SKIPPED
    assert not source.requests, "非交易日不应发出任何请求"


def test_单标的失败不影响其他标的且成功部分已落盘(tmp_path: Path) -> None:
    source = _source(fail_codes={"000001": 99})  # 一直失败
    task = _task(tmp_path, source)

    result = task.run(codes=["600000", "000001"], start=START, end=TRADE_DAY)

    assert [item.code for item in result.failed] == ["000001"]
    assert not result.ok
    assert result.rows_written == 3, "成功标的的数据必须已落盘"
    assert task.store.coverage("daily").code_count == 1
    assert task.store.read("daily", codes=["600000"]).shape[0] == 3


def test_补偿轮重试成功后不再计为失败(tmp_path: Path) -> None:
    source = _source(fail_codes={"000001": 1})  # 首轮失败一次，补偿轮成功
    task = _task(tmp_path, source, compensation_rounds=1)

    result = task.run(codes=["600000", "000001"], start=START, end=TRADE_DAY)

    assert result.failed == ()
    assert result.rows_written == 6
    assert task.store.coverage("daily").code_count == 2


def test_多次落盘保证部分进度可持久化(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source, flush_every=1)

    result = task.run(codes=["600000", "000001"], start=START, end=TRADE_DAY)

    assert result.flushes == 2, "flush_every=1 时应分两次落盘"


def test_本地无数据且未指定起点时拒绝长历史拉取(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source)

    with pytest.raises(ValidationError):
        task.run(codes=["600000"], end=TRADE_DAY)


def test_未指定标的时使用全部股票池并排除退市(tmp_path: Path) -> None:
    source = _source()
    source.add_stock_info(
        pd.DataFrame(
            [
                {"code": "600000", "name": "浦发银行", "delist_date": None},
                {"code": "000001", "name": "平安银行", "delist_date": None},
                {"code": "000002", "name": "已退市", "delist_date": "2020-01-01"},
            ]
        )
    )
    task = _task(tmp_path, source)

    result = task.run(start=START, end=TRADE_DAY)

    assert result.requested_codes == 2, "退市标的没有新行情，不应进入同步范围"
    requested = {code for codes, _, _ in source.requests for code in codes}
    assert requested == {"600000", "000001"}


def test_数据版本号按同日递增(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source)
    # 两次同步锚定同一交易日：版本号前缀相同，序号才应递增
    task.run(codes=["600000"], start=START, end=date(2026, 10, 13), trade_date=TRADE_DAY)
    second = task.run(codes=["600000"], end=TRADE_DAY, trade_date=TRADE_DAY)

    assert task.store.data_version("daily") == "20261014.2"
    assert second.data_version == "20261014.2"


def test_重复同步保持幂等(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source)
    task.run(codes=["600000"], start=START, end=TRADE_DAY)

    task.run(codes=["600000"], start=START, end=TRADE_DAY)

    assert task.store.read("daily").shape[0] == 3, "重复同步不得让数据翻倍"


def test_无数据的标的记入空集合而不算失败(tmp_path: Path) -> None:
    source = _source()
    task = _task(tmp_path, source)

    result = task.run(codes=["600000", "300750"], start=START, end=TRADE_DAY)

    assert result.empty_codes == ("300750",)
    assert result.failed == ()
    assert result.ok


def test_任务日志记录成功详情(tmp_path: Path) -> None:
    source = _source()
    log = MemoryTaskLog()
    task = _task(tmp_path, source, log=log)

    task.run(codes=["600000", "000001"], start=START, end=TRADE_DAY, trade_date=TRADE_DAY)

    record = log.records[-1]
    assert record.status is TaskStatus.SUCCESS
    assert record.name == "sync_daily"
    assert record.duration_seconds is not None and record.duration_seconds >= 0
    assert record.detail["rows_written"] == 6
    assert record.detail["data_version"] == "20261014.1"


def test_任务日志在异常时记为失败并保留异常(tmp_path: Path) -> None:
    source = _source()
    log = MemoryTaskLog()

    class ExplodingStore(ParquetStore):
        def write(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
            raise RuntimeError("磁盘写满")

    task = DailySyncTask(
        source=source,
        store=ExplodingStore(tmp_path / "parquet"),
        task_log=log,
        calendar=source.calendar,
    )

    with pytest.raises(RuntimeError):
        task.run(codes=["600000"], start=START, end=TRADE_DAY)

    record = log.records[-1]
    assert record.status is TaskStatus.FAILED
    assert "磁盘写满" in (record.error or "")
