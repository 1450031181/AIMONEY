"""Parquet 列存（T-07）。

三个必须做对的地方：

1. **幂等 upsert**：同一 ``(code, date)`` 重复写入必须覆盖而不是堆积。
   日线增量同步一定会重叠（补偿任务、手工重跑、数据源修订历史），
   幂等性保证"重复同步 N 次 == 同步 1 次"，这是能安心重跑补偿任务的前提。
2. **分区裁剪**：按 ``year=YYYY/month=MM`` 分区，回测只读命中分区。
   全 A 股 10 年日线约 1200 万行，全量扫描会让每次回测都白等几十秒。
3. **原子落盘**：先写 ``*.tmp`` 再 ``replace``。中途断网/断电若留下半个 parquet，
   整个分区都会读不出来——宁可本次同步失败，也不能污染已有数据。

目录约定::

    <root>/daily/year=2026/month=10/part-000.parquet
    <root>/daily/_meta.json                 # data_version / updated_at
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd

from engine.datahub.schema import ensure_columns
from engine.errors import DataNotReadyError, ValidationError

_PARTITION_GLOB: Final[str] = "year=*/month=*/part-*.parquet"
_META_FILE: Final[str] = "_meta.json"
_PARTITION_COLUMN: Final[str] = "date"
#: 与 pandas-stubs 的 compression 字面量保持一致
CompressionType = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]


@dataclass(frozen=True, slots=True)
class WriteResult:
    """写入结果：供同步任务记录日志与判断是否需要告警。

    注意 ``written_rows`` 是**本次落盘的行数**（即本次输入中去重后的行数），
    不是"写入后分区内的总行数"——后者用 ``partition_rows``。
    这两个语义混用会让"累计落盘行数"随着重复写入虚高。
    """

    dataset: str
    input_rows: int
    written_rows: int
    duplicates_dropped: int
    partition_rows: int
    partitions: tuple[str, ...]
    data_version: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetCoverage:
    """数据集覆盖情况（供 `GET /data/coverage` 与数据质量日报使用）。"""

    dataset: str
    row_count: int
    code_count: int
    start: date | None
    end: date | None
    partitions: int
    data_version: str | None
    updated_at: datetime | None


class ParquetStore:
    """基于 pyarrow 的列存实现（写入 / 读取 / 覆盖率）。

    只在 engine 内部使用，不感知 HTTP、不感知业务语义。
    """

    def __init__(self, root: Path | str, *, compression: CompressionType = "zstd") -> None:
        self.root = Path(root)
        self.compression = compression

    # ============================================================ 路径
    def dataset_dir(self, dataset: str) -> Path:
        _ensure_dataset_name(dataset)
        return self.root / dataset

    def partition_path(self, dataset: str, year: int, month: int, index: int = 0) -> Path:
        return self.dataset_dir(dataset) / f"year={year:04d}" / f"month={month:02d}" / f"part-{index:03d}.parquet"

    def exists(self, dataset: str) -> bool:
        return any(self.dataset_dir(dataset).glob(_PARTITION_GLOB))

    def partitions(self, dataset: str) -> list[tuple[int, int]]:
        """已落盘的分区列表（升序）。"""
        found: set[tuple[int, int]] = set()
        for path in self.dataset_dir(dataset).glob(_PARTITION_GLOB):
            year = _parse_partition_value(path.parent.parent.name, "year=")
            month = _parse_partition_value(path.parent.name, "month=")
            if year is not None and month is not None:
                found.add((year, month))
        return sorted(found)

    # ============================================================ 写入
    def write(
        self,
        dataset: str,
        frame: pd.DataFrame,
        *,
        key_columns: Sequence[str] = ("code", "date"),
        data_version: str | None = None,
    ) -> WriteResult:
        """幂等写入：按 ``key_columns`` 去重（同主键保留本次值），按年月分区落盘。"""
        keys = tuple(key_columns)
        if frame.empty:
            return WriteResult(
                dataset=dataset,
                input_rows=0,
                written_rows=0,
                duplicates_dropped=0,
                partition_rows=0,
                partitions=(),
                data_version=data_version,
            )
        ensure_columns(frame, keys, dataset=dataset)
        ensure_columns(frame, (_PARTITION_COLUMN,), dataset=dataset)

        normalized = frame.copy()
        normalized[_PARTITION_COLUMN] = pd.to_datetime(normalized[_PARTITION_COLUMN]).dt.normalize()
        # 用 "YYYY-MM" 字符串作为分组键：既避免 groupby 多键返回 Hashable 元组的类型噪音，
        # 又天然按时间升序（零填充保证字典序 == 时间序）
        normalized["_period"] = normalized[_PARTITION_COLUMN].dt.strftime("%Y-%m")

        touched: list[str] = []
        written_rows = 0
        partition_rows = 0
        dropped = 0
        for period, group in normalized.groupby("_period", sort=True):
            year_text, month_text = str(period).split("-")
            year, month = int(year_text), int(month_text)
            part = group.drop(columns=["_period"]).drop_duplicates(subset=list(keys), keep="last")
            dropped += len(group) - len(part)  # 本批次内部重复
            path = self.partition_path(dataset, year, month)
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, part], ignore_index=True)
                before = len(merged)
                merged = merged.drop_duplicates(subset=list(keys), keep="last")
                dropped += before - len(merged)  # 与已有数据重叠、被本次覆盖的行
            else:
                merged = part
            merged = merged.sort_values(list(keys)).reset_index(drop=True)
            self._atomic_write(path, merged)
            touched.append(f"year={year:04d}/month={month:02d}")
            written_rows += len(part)  # 本次实际落盘的行数（keep="last" 保证它们都在）
            partition_rows += len(merged)

        if data_version is not None:
            self.set_data_version(dataset, data_version)

        return WriteResult(
            dataset=dataset,
            input_rows=len(frame),
            written_rows=written_rows,
            duplicates_dropped=dropped,
            partition_rows=partition_rows,
            partitions=tuple(touched),
            data_version=data_version,
        )

    def meta(self, dataset: str) -> dict[str, Any]:
        """数据集元信息（``data_version`` / ``updated_at``），供覆盖率接口与回测快照记录使用。"""
        return self._read_meta(dataset)

    def updated_at(self, dataset: str) -> datetime | None:
        return _parse_datetime(self._read_meta(dataset).get("updated_at"))

    def files(self, dataset: str, *, start: date | None = None, end: date | None = None) -> list[Path]:
        """按年月分区裁剪后命中的 parquet 文件列表。

        分区裁剪是**唯一实现**（此处），存储读取与 DuckDB 查询层共用，
        避免两处各写一份阈值/边界逻辑而不一致。
        """
        paths: list[Path] = []
        for year, month in self.partitions(dataset):
            if start is not None and _month_index(year, month) < _month_index(start.year, start.month):
                continue
            if end is not None and _month_index(year, month) > _month_index(end.year, end.month):
                continue
            paths.extend(
                sorted(self.dataset_dir(dataset).glob(f"year={year:04d}/month={month:02d}/part-*.parquet"))
            )
        return paths

    # ============================================================ 读取
    def read(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        codes: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """读取行情。``start/end`` 会命中分区裁剪，只打开相关月份的 parquet。"""
        paths = self.files(dataset, start=start, end=end)
        if not paths:
            return pd.DataFrame(columns=list(columns) if columns else None)

        # 内部必须拿到主键列才能做行级过滤，最后再按调用方要求裁剪列
        internal_columns = list(columns) if columns else None
        if internal_columns is not None:
            for column in ("code", _PARTITION_COLUMN):
                if column not in internal_columns:
                    internal_columns.append(column)

        frames = [pd.read_parquet(path, columns=internal_columns) for path in paths]
        result = pd.concat(frames, ignore_index=True)

        if codes is not None:
            result = result.loc[result["code"].isin(list(codes))]
        if start is not None:
            result = result.loc[result[_PARTITION_COLUMN] >= pd.Timestamp(start)]
        if end is not None:
            result = result.loc[result[_PARTITION_COLUMN] <= pd.Timestamp(end)]

        result = result.sort_values(["code", _PARTITION_COLUMN]).reset_index(drop=True)
        if columns is not None:
            result = result.loc[:, [column for column in columns if column in result.columns]]
        return result

    # ============================================================ 元信息
    def data_version(self, dataset: str) -> str | None:
        meta = self._read_meta(dataset)
        version = meta.get("data_version")
        return str(version) if version is not None else None

    def set_data_version(self, dataset: str, version: str) -> None:
        meta = self._read_meta(dataset)
        meta.update({"data_version": version, "updated_at": datetime.now().isoformat(timespec="seconds")})
        self.dataset_dir(dataset).mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(self.dataset_dir(dataset) / _META_FILE, json.dumps(meta, ensure_ascii=False, indent=2))

    def require_data(self, dataset: str) -> None:
        """数据门禁：数据不存在时抛 ``DataNotReadyError``。

        调用方（盘前任务链）必须让这个异常阻止当日交易，而不是降级用旧数据。
        """
        if not self.exists(dataset):
            raise DataNotReadyError(f"数据集 {dataset} 尚未同步，禁止在缺数据的情况下继续", detail={"dataset": dataset})

    def coverage(self, dataset: str) -> DatasetCoverage:
        """数据集覆盖情况（只读主键列，代价很小）。"""
        paths = sorted(self.dataset_dir(dataset).glob(_PARTITION_GLOB))
        meta = self._read_meta(dataset)
        if not paths:
            return DatasetCoverage(
                dataset=dataset,
                row_count=0,
                code_count=0,
                start=None,
                end=None,
                partitions=0,
                data_version=meta.get("data_version"),
                updated_at=_parse_datetime(meta.get("updated_at")),
            )

        row_count = 0
        codes: set[str] = set()
        start: date | None = None
        end: date | None = None
        for path in paths:
            frame = pd.read_parquet(path, columns=["code", _PARTITION_COLUMN])
            row_count += len(frame)
            codes.update(frame["code"].astype(str).unique().tolist())
            if not frame.empty:
                part_start = frame[_PARTITION_COLUMN].min().date()
                part_end = frame[_PARTITION_COLUMN].max().date()
                start = part_start if start is None else min(start, part_start)
                end = part_end if end is None else max(end, part_end)

        return DatasetCoverage(
            dataset=dataset,
            row_count=row_count,
            code_count=len(codes),
            start=start,
            end=end,
            partitions=len(paths),
            data_version=meta.get("data_version"),
            updated_at=_parse_datetime(meta.get("updated_at")),
        )

    # ============================================================ 内部
    def _atomic_write(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            frame.to_parquet(tmp, engine="pyarrow", compression=self.compression, index=False)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def _read_meta(self, dataset: str) -> dict[str, Any]:
        meta_path = self.dataset_dir(dataset) / _META_FILE
        if not meta_path.is_file():
            return {}
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"数据集元信息损坏：{meta_path}", detail={"error": str(exc)}) from exc
        if not isinstance(loaded, dict):
            raise ValidationError(f"数据集元信息必须是 JSON 对象：{meta_path}")
        return loaded


# ============================================================ 模块级工具
def _ensure_dataset_name(dataset: str) -> None:
    if not dataset or "/" in dataset or "\\" in dataset or dataset.startswith("."):
        raise ValidationError(
            "数据集名非法（禁止空值、路径分隔符与前缀点）", detail={"dataset": dataset}
        )


def _month_index(year: int, month: int) -> int:
    return year * 12 + month


def _parse_partition_value(segment: str, prefix: str) -> int | None:
    if not segment.startswith(prefix):
        return None
    try:
        return int(segment[len(prefix) :])
    except ValueError:
        return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


__all__ = ["DatasetCoverage", "ParquetStore", "WriteResult"]
