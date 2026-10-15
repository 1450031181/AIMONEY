"""T-07 Parquet 列存测试：幂等 upsert、分区裁剪、原子落盘、覆盖率统计。

这些性质直接决定"能不能安心重跑同步任务"：
如果 upsert 不幂等，补偿任务就会把数据翻倍；如果分区裁剪失效，回测会慢一个量级。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from engine.datahub.store import ParquetStore
from engine.errors import DataNotReadyError, DataQualityError, ValidationError


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "parquet")


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rows(*, code: str = "600000", dates: list[str], close: float = 10.0) -> list[dict[str, object]]:
    return [
        {"code": code, "date": value, "open": close, "high": close, "low": close, "close": close, "volume": 100}
        for value in dates
    ]


def test_写入后可读出相同数据(store: ParquetStore) -> None:
    frame = _frame(_rows(dates=["2026-10-12", "2026-10-13"]))
    result = store.write("daily", frame)

    assert result.input_rows == 2
    assert result.written_rows == 2
    assert result.partitions == ("year=2026/month=10",)

    loaded = store.read("daily")
    assert len(loaded) == 2
    assert list(loaded["close"]) == [10.0, 10.0]


def test_按年月分区落盘(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(dates=["2026-09-30", "2026-10-12", "2026-11-02"])))

    assert store.partitions("daily") == [(2026, 9), (2026, 10), (2026, 11)]
    assert (store.dataset_dir("daily") / "year=2026" / "month=10" / "part-000.parquet").is_file()


def test_重复写入同一批数据保持幂等(store: ParquetStore) -> None:
    frame = _frame(_rows(dates=["2026-10-12", "2026-10-13"]))
    store.write("daily", frame)
    second = store.write("daily", frame)

    assert second.written_rows == 2, "重复写入后分区内仍应只有 2 行"
    assert second.duplicates_dropped == 2
    assert len(store.read("daily")) == 2


def test_同一主键新值覆盖旧值(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(dates=["2026-10-13"], close=10.0)))
    store.write("daily", _frame(_rows(dates=["2026-10-13"], close=12.5)))

    loaded = store.read("daily")
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == pytest.approx(12.5)


def test_分区裁剪只打开命中月份的parquet(store: ParquetStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.write("daily", _frame(_rows(dates=["2026-06-15"])))
    store.write("daily", _frame(_rows(dates=["2026-10-13"])))

    opened: list[str] = []
    real_read_parquet = pd.read_parquet

    def spy(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        partition = Path(str(path)).parent
        opened.append(f"{partition.parent.name}/{partition.name}")
        return real_read_parquet(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", spy)
    result = store.read("daily", start=date(2026, 10, 1), end=date(2026, 10, 31))

    assert len(result) == 1
    assert opened == ["year=2026/month=10"], f"分区裁剪失效，打开了：{opened}"


def test_列裁剪只返回请求的列(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(dates=["2026-10-13"])))
    result = store.read("daily", columns=["close"])
    assert list(result.columns) == ["close"]


def test_按代码与区间过滤(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(code="600000", dates=["2026-10-12", "2026-10-13", "2026-10-14"])))
    store.write("daily", _frame(_rows(code="000001", dates=["2026-10-12", "2026-10-13", "2026-10-14"])))

    result = store.read("daily", codes=["600000"], start=date(2026, 10, 13), end=date(2026, 10, 14))
    assert len(result) == 2
    assert set(result["code"]) == {"600000"}


def test_读取结果按代码与日期排序(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(code="600000", dates=["2026-10-14", "2026-10-12"])))
    store.write("daily", _frame(_rows(code="000001", dates=["2026-10-13"])))

    result = store.read("daily")
    assert list(result["code"]) == ["000001", "600000", "600000"]
    assert list(result["date"].dt.strftime("%Y-%m-%d")) == ["2026-10-13", "2026-10-12", "2026-10-14"]


def test_无数据时读取返回空表(store: ParquetStore) -> None:
    assert store.read("daily").empty
    assert not store.exists("daily")


def test_写入空表不做任何落盘(store: ParquetStore) -> None:
    result = store.write("daily", pd.DataFrame(columns=["code", "date", "close"]))
    assert result.written_rows == 0
    assert result.partitions == ()
    assert not store.exists("daily")


def test_缺少主键列时拒绝写入(store: ParquetStore) -> None:
    with pytest.raises(DataQualityError):
        store.write("daily", pd.DataFrame([{"close": 10.0}]))


def test_原子写不留下临时文件(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(dates=["2026-10-13"])))
    leftovers = list(store.dataset_dir("daily").rglob("*.tmp"))
    assert leftovers == []


def test_数据版本与覆盖率(store: ParquetStore) -> None:
    store.write("daily", _frame(_rows(code="600000", dates=["2026-09-30"])), data_version="20260930.1")
    store.write("daily", _frame(_rows(code="000001", dates=["2026-10-13"])), data_version="20261014.1")

    assert store.data_version("daily") == "20261014.1"

    coverage = store.coverage("daily")
    assert coverage.row_count == 2
    assert coverage.code_count == 2
    assert coverage.start == date(2026, 9, 30)
    assert coverage.end == date(2026, 10, 13)
    assert coverage.partitions == 2
    assert coverage.updated_at is not None


def test_覆盖率对空数据集返回零值(store: ParquetStore) -> None:
    coverage = store.coverage("daily")
    assert coverage.row_count == 0
    assert coverage.start is None
    assert coverage.data_version is None
    assert coverage.updated_at is None


def test_数据门禁_缺数据时抛错而不是降级(store: ParquetStore) -> None:
    with pytest.raises(DataNotReadyError):
        store.require_data("daily")

    store.write("daily", _frame(_rows(dates=["2026-10-13"])))
    store.require_data("daily")  # 不再抛错


def test_数据集名非法被拒(store: ParquetStore) -> None:
    for bad in ["", "a/b", "a\\b", ".hidden"]:
        with pytest.raises(ValidationError):
            store.dataset_dir(bad)
