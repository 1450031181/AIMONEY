"""T-12 DuckDB 查询层测试：分区裁剪、列裁剪、代码过滤、覆盖率聚合。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from engine.datahub.reader import DuckDBReader
from engine.datahub.store import ParquetStore
from engine.errors import DataNotReadyError, ValidationError


def _frame(*, code: str = "600000", days: list[str], close: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": code,
                "date": value,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100,
                "adj_factor": 1.0,
            }
            for value in days
        ]
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store = ParquetStore(tmp_path / "parquet")
    store.write("daily", _frame(days=["2026-09-30"]), data_version="20260930.1")
    store.write("daily", _frame(days=["2026-10-12", "2026-10-13", "2026-10-14"]), data_version="20261014.2")
    store.write("daily", _frame(code="000001", days=["2026-10-13", "2026-10-14"]))
    return tmp_path / "parquet"


def test_按区间与代码查询(root: Path) -> None:
    with DuckDBReader(root) as reader:
        frame = reader.daily(codes=["600000"], start=date(2026, 10, 13), end=date(2026, 10, 14))

    assert len(frame) == 2
    assert set(frame["code"]) == {"600000"}
    assert list(pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")) == ["2026-10-13", "2026-10-14"]


def test_不传代码时返回全部并排序(root: Path) -> None:
    with DuckDBReader(root) as reader:
        frame = reader.daily(start=date(2026, 10, 1))

    # 600000 在 10 月有 3 行（9-30 那行属 9 月分区，被裁剪掉），000001 有 2 行
    assert list(frame["code"]) == ["000001", "000001", "600000", "600000", "600000"]


def test_列裁剪只取需要的列(root: Path) -> None:
    with DuckDBReader(root) as reader:
        frame = reader.daily(columns=["close"], start=date(2026, 10, 12), end=date(2026, 10, 12))

    assert list(frame.columns) == ["close"]


def test_分区裁剪只命中相关月份(root: Path) -> None:
    with DuckDBReader(root) as reader:
        files = reader.plan_files("daily", start=date(2026, 10, 1), end=date(2026, 10, 31))

    assert len(files) == 1
    assert "month=10" in files[0].as_posix()


def test_分区裁剪影响实际查询结果范围(root: Path) -> None:
    with DuckDBReader(root) as reader:
        frame = reader.daily(start=date(2026, 10, 1), end=date(2026, 10, 31))

    assert min(pd.to_datetime(frame["date"]).dt.date) == date(2026, 10, 12), "9 月分区不应被读入"


def test_覆盖率聚合正确(root: Path) -> None:
    with DuckDBReader(root) as reader:
        coverage = reader.coverage("daily")

    assert coverage.row_count == 6
    assert coverage.code_count == 2
    assert coverage.start == date(2026, 9, 30)
    assert coverage.end == date(2026, 10, 14)
    assert coverage.partitions == 2
    assert coverage.data_version == "20261014.2"
    assert coverage.updated_at is not None


def test_空数据集的覆盖率返回零值(tmp_path: Path) -> None:
    with DuckDBReader(tmp_path / "parquet") as reader:
        coverage = reader.coverage("daily")
        frame = reader.daily(start=date(2026, 10, 1))

    assert coverage.row_count == 0
    assert coverage.start is None
    assert coverage.data_version is None
    assert frame.empty
    assert reader.plan_files("daily") == []


def test_数据门禁_缺数据时报错(tmp_path: Path) -> None:
    with DuckDBReader(tmp_path / "parquet") as reader, pytest.raises(DataNotReadyError):
        reader.require_data("daily")


def test_数据集名非法被拒(root: Path) -> None:
    with DuckDBReader(root) as reader, pytest.raises(ValidationError):
        reader.plan_files("bad/name")


def test_摘要含文件数与版本(root: Path) -> None:
    with DuckDBReader(root) as reader:
        attributes = reader.attributes("daily")

    assert attributes["dataset"] == "daily"
    assert attributes["code_count"] == 2
    assert attributes["files"] >= 1
    assert attributes["start"] == "2026-09-30"


def test_可自写SQL并引用数据集(root: Path) -> None:
    with DuckDBReader(root) as reader:
        frame = reader.query(
            f"SELECT code, COUNT(*) AS bars FROM {reader.dataset_sql('daily')} GROUP BY code ORDER BY code"
        )

    assert list(frame["code"]) == ["000001", "600000"]
    assert list(frame["bars"]) == [2, 4]


def test_连接可显式关闭(tmp_path: Path) -> None:
    reader = DuckDBReader(tmp_path / "parquet")
    reader.close()
    with pytest.raises(Exception):  # noqa: B017 —— duckdb 关闭后执行会抛错，类型随版本变化
        reader.query("SELECT 1")
