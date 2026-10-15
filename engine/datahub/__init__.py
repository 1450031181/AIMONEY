"""数据访问层：交易日历、列存、查询、复权、质量校验。"""

from __future__ import annotations

from engine.datahub.adjust import (
    ContinuityViolation,
    assert_continuity,
    ensure_factor_column,
    find_continuity_violations,
    restore_real_frame,
    restore_real_price,
    to_pre_adjusted,
)
from engine.datahub.calendar import TradingCalendarImpl, weekday_calendar
from engine.datahub.quality import (
    QualityCheckTask,
    QualityFinding,
    QualityReport,
    QualityStatus,
    QualityThresholds,
    TradeGate,
    check_quality,
)
from engine.datahub.reader import DEFAULT_DATASET, DuckDBReader
from engine.datahub.reference import STOCK_INFO, ReferenceStore
from engine.datahub.rules import (
    infer_board,
    limit_prices,
    price_limit_ratio,
)
from engine.datahub.store import DatasetCoverage, ParquetStore, WriteResult
from engine.datahub.sync import (
    DailySyncTask,
    IndexMembersResult,
    IndexMembersSyncTask,
    SyncFailure,
    SyncResult,
)
from engine.datahub.tasklog import JsonlTaskLog, MemoryTaskLog, TaskLogger, TaskRecord, TaskStatus
from engine.datahub.universe import (
    JsonUniverseStore,
    UniverseFilter,
    UniverseHistory,
    UniverseSelection,
    select_universe,
)

__all__ = [
    "DEFAULT_DATASET",
    "STOCK_INFO",
    "ContinuityViolation",
    "DailySyncTask",
    "DatasetCoverage",
    "DuckDBReader",
    "IndexMembersResult",
    "IndexMembersSyncTask",
    "JsonUniverseStore",
    "JsonlTaskLog",
    "MemoryTaskLog",
    "ParquetStore",
    "QualityCheckTask",
    "QualityFinding",
    "QualityReport",
    "QualityStatus",
    "QualityThresholds",
    "ReferenceStore",
    "SyncFailure",
    "SyncResult",
    "TaskLogger",
    "TaskRecord",
    "TaskStatus",
    "TradeGate",
    "TradingCalendarImpl",
    "UniverseFilter",
    "UniverseHistory",
    "UniverseSelection",
    "WriteResult",
    "assert_continuity",
    "check_quality",
    "ensure_factor_column",
    "find_continuity_violations",
    "infer_board",
    "limit_prices",
    "price_limit_ratio",
    "restore_real_frame",
    "restore_real_price",
    "select_universe",
    "to_pre_adjusted",
    "weekday_calendar",
]
