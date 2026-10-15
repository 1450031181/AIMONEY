"""DuckDB 查询层（T-12）。

为什么不直接用 pandas 拼文件读：

1. **分析型查询下沉到 DuckDB**：覆盖率统计（行数/标的数/起止日期）在存储层原本要
   逐个分区读主键列再聚合，标的多了就是几百次小 IO；DuckDB 一条 SQL 搞定，
   且只读 Parquet footer 与必要的列。
2. **列裁剪 + 谓词下推由数据库负责**：回测读 10 年全市场时，只取需要的列远比
   "读进来再扔"便宜一个量级。
3. **分区裁剪仍由我们控制**：DuckDB 的 hive 分区裁剪依赖对 ``year/month`` 的谓词，
   而业务过滤条件是 ``date`` 列，因此这里显式把命中文件列表传给 ``read_parquet``，
   裁剪逻辑与 ``ParquetStore.files()`` 共用一份实现。

用法::

    with DuckDBReader(root / "parquet") as reader:
        frame = reader.daily(codes=["600000"], start=date(2026, 10, 1), end=date(2026, 10, 31))
        coverage = reader.coverage("daily")
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb
import pandas as pd

from engine.datahub.store import DatasetCoverage, ParquetStore
from engine.errors import DataNotReadyError, ValidationError

DEFAULT_DATASET = "daily"


@dataclass
class DuckDBReader:
    """基于 DuckDB 的只读查询层（存储层负责写，读全部走这里）。"""

    root: Path | str
    database: Path | str | None = None
    read_only: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._store = ParquetStore(self.root)
        if self.database is None:
            self._connection = duckdb.connect(":memory:")
        else:
            Path(self.database).parent.mkdir(parents=True, exist_ok=True)
            self._connection = duckdb.connect(str(self.database), read_only=self.read_only)

    # -------------------------------------------------- 生命周期
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> DuckDBReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -------------------------------------------------- 分区裁剪
    def plan_files(
        self, dataset: str = DEFAULT_DATASET, *, start: date | None = None, end: date | None = None
    ) -> list[Path]:
        """本次查询会实际打开的文件（与存储层共用同一份分区裁剪逻辑）。"""
        return self._store.files(dataset, start=start, end=end)

    # -------------------------------------------------- 读行情
    def daily(
        self,
        *,
        codes: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        columns: Sequence[str] | None = None,
        dataset: str = DEFAULT_DATASET,
    ) -> pd.DataFrame:
        files = self.plan_files(dataset, start=start, end=end)
        if not files:
            return pd.DataFrame(columns=list(columns) if columns else None)

        selection = ", ".join(f'"{column}"' for column in columns) if columns else "*"
        conditions: list[str] = []
        params: list[Any] = []
        if codes:
            conditions.append("list_contains(?, code)")
            params.append([str(code) for code in codes])
        if start is not None:
            conditions.append("date >= ?")
            params.append(start)
        if end is not None:
            conditions.append("date <= ?")
            params.append(end)

        sql = f"SELECT {selection} FROM {self._read_parquet(files)}"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY code, date"
        return self._connection.execute(sql, params).fetch_df()

    # -------------------------------------------------- 覆盖率
    def coverage(self, dataset: str = DEFAULT_DATASET) -> DatasetCoverage:
        """一条 SQL 完成覆盖率统计（行数 / 标的数 / 起止日期）。

        空数据集不报错而是返回零值——"还没同步"是正常状态，
        是否允许交易由交易门禁决定（见 T-11），不在这里混为一谈。
        """
        files = self.plan_files(dataset)
        data_version = self._store.data_version(dataset)
        updated_at = self._store.updated_at(dataset)
        partitions = len(self._store.partitions(dataset))
        if not files:
            return DatasetCoverage(
                dataset=dataset,
                row_count=0,
                code_count=0,
                start=None,
                end=None,
                partitions=0,
                data_version=str(data_version) if data_version is not None else None,
                updated_at=updated_at,
            )

        sql = (
            "SELECT COUNT(*) AS row_count, COUNT(DISTINCT code) AS code_count, "
            "MIN(date) AS start_date, MAX(date) AS end_date "
            f"FROM {self._read_parquet(files)}"
        )
        row = self._connection.execute(sql).fetchone()
        if row is None:  # pragma: no cover —— 聚合查询必然返回一行
            raise ValidationError("覆盖率聚合查询未返回结果", detail={"dataset": dataset})
        row_count, code_count, start_date, end_date = row
        return DatasetCoverage(
            dataset=dataset,
            row_count=int(row_count),
            code_count=int(code_count),
            start=_to_date(start_date),
            end=_to_date(end_date),
            partitions=partitions,
            data_version=data_version,
            updated_at=updated_at,
        )

    def require_data(self, dataset: str = DEFAULT_DATASET) -> None:
        if not self._store.exists(dataset):
            raise DataNotReadyError(
                f"数据集 {dataset} 尚未同步，禁止在缺数据的情况下继续", detail={"dataset": dataset}
            )

    # -------------------------------------------------- 逃生舱
    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """执行自定义 SQL（分析脚本用）。表名约定见 ``dataset_sql()``。"""
        return self._connection.execute(sql, list(params or [])).fetch_df()

    def dataset_sql(self, dataset: str = DEFAULT_DATASET) -> str:
        """把数据集暴露为可直接引用的 SQL 表达式，便于自写查询。"""
        return self._read_parquet(self.plan_files(dataset))

    def attributes(self, dataset: str = DEFAULT_DATASET) -> Mapping[str, Any]:
        """数据集摘要（CLI 与接口的公共数据来源）。"""
        coverage = self.coverage(dataset)
        return {
            "dataset": coverage.dataset,
            "row_count": coverage.row_count,
            "code_count": coverage.code_count,
            "start": coverage.start.isoformat() if coverage.start else None,
            "end": coverage.end.isoformat() if coverage.end else None,
            "partitions": coverage.partitions,
            "data_version": coverage.data_version,
            "updated_at": coverage.updated_at.isoformat() if coverage.updated_at else None,
            "files": len(self.plan_files(dataset)),
        }

    # -------------------------------------------------- 内部
    @staticmethod
    def _read_parquet(files: Sequence[Path]) -> str:
        # Windows 路径含反斜杠会被 DuckDB 当作转义符，统一成正斜杠
        paths = ", ".join(f"'{Path(file).as_posix()}'" for file in files)
        return f"read_parquet([{paths}])"


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    return pd.Timestamp(value).date()


__all__ = ["DEFAULT_DATASET", "DuckDBReader"]
