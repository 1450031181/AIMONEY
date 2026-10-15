"""日线增量同步任务（T-08）。

数据链路的第一个"生产级"环节，四个必须做对的地方：

1. **逐标的取数 + 批量落盘**，而不是"批量取数"。底层 akshare 接口本来就是逐标的请求，
   批量取数一旦某只票重试耗尽就整体抛错、已拉到的数据全丢。逐标的取数把失败影响
   限制在单只票，且每 ``flush_every`` 只票就落盘一次——中途崩溃也能保住已完成的部分。
2. **增量续传**：起点默认取本地最大日期的次日。重复同步不会翻倍（靠 ParquetStore 的幂等 upsert），
   但也不会白拉十年历史。
3. **补偿轮**：首轮失败的标的在收尾时再试若干轮，仍失败才计入 ``SyncResult.failed`` 并告警。
4. **数据版本号**：每次成功同步写 ``YYYYMMDD.N``，回测任务据此记录所用数据快照。

**不做的事**：不在这里做数据质量判定（T-11），也不在这里决定是否允许交易——
本任务只负责"把数据搬进来"，判定与门禁由 T-11 独立完成。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from engine.datahub.reference import STOCK_INFO, ReferenceStore
from engine.datahub.store import ParquetStore
from engine.datahub.tasklog import TaskLogger, TaskLogStore
from engine.datahub.universe import JsonUniverseStore, UniverseHistory
from engine.errors import DataSourceError, ValidationError
from engine.ports import AdjType, DataSource, TradingCalendar

_logger = logging.getLogger(__name__)

DEFAULT_TASK_NAME = "sync_daily"


@dataclass(frozen=True, slots=True)
class SyncFailure:
    code: str
    reason: str


@dataclass
class SyncResult:
    """同步结果。``failed`` 非空即需要告警——缺数据的交易日绝不允许照常交易。"""

    dataset: str
    skipped: bool = False
    skip_reason: str = ""
    start: date | None = None
    end: date | None = None
    requested_codes: int = 0
    empty_codes: tuple[str, ...] = ()
    rows_fetched: int = 0
    rows_written: int = 0
    failed: tuple[SyncFailure, ...] = ()
    data_version: str | None = None
    flushes: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped

    def summary(self) -> str:
        if self.skipped:
            return f"跳过：{self.skip_reason}"
        return (
            f"{self.dataset} {self.start}~{self.end} 标的 {self.requested_codes} 只，"
            f"取数 {self.rows_fetched} 行，落盘 {self.rows_written} 行，"
            f"失败 {len(self.failed)} 只，版本 {self.data_version}"
        )


@dataclass
class DailySyncTask:
    """日线同步任务（实现上只依赖端口，回测/实盘/CLI 均可复用）。"""

    source: DataSource
    store: ParquetStore
    task_log: TaskLogStore
    dataset: str = "daily"
    #: 复权口径：回测计算统一存后复权，展示层再转前复权（设计文档 5.4）
    adj: AdjType = "post"
    #: 每取够这么多标的就落盘一次（控制内存，同时保证部分进度可持久化）
    flush_every: int = 200
    #: 收尾补偿轮数
    compensation_rounds: int = 1
    calendar: TradingCalendar | None = None
    task_name: str = DEFAULT_TASK_NAME
    #: 无本地数据时要求显式指定回填起点，避免误触发十年全量拉取
    full_backfill_start: date | None = None
    #: 传入后会顺带把股票基础信息落到 data/reference（离线场景不再依赖网络）
    reference_store: ReferenceStore | None = None
    #: 首轮失败、待补偿重试的标的 {code: reason}
    _pending: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # ============================================================ 主流程
    def run(
        self,
        *,
        codes: Sequence[str] | None = None,
        start: date | None = None,
        end: date | None = None,
        trade_date: date | None = None,
    ) -> SyncResult:
        anchor = trade_date or date.today()
        result = SyncResult(dataset=self.dataset)
        with TaskLogger(self.task_log, self.task_name, trade_date=anchor) as task:
            calendar = self.calendar or self._source_calendar()
            if calendar is not None and not calendar.is_trade_day(anchor):
                result.skipped = True
                result.skip_reason = f"{anchor.isoformat()} 非交易日"
                task.skip(result.skip_reason)
                return result

            stock_info = self._stock_info_frame() if (codes is None or self.reference_store) else None
            resolved_codes = list(codes) if codes else self._all_codes(stock_info)
            result.requested_codes = len(resolved_codes)
            if not resolved_codes:
                result.skipped = True
                result.skip_reason = "没有可同步的标的"
                task.skip(result.skip_reason)
                return result

            resolved_start, resolved_end = self._resolve_range(start=start, end=end, anchor=anchor)
            if resolved_start > resolved_end:
                result.skipped = True
                result.skip_reason = f"已是最新（本地截至 {resolved_start - timedelta(days=1)}）"
                task.skip(result.skip_reason)
                return result
            result.start, result.end = resolved_start, resolved_end

            self._fetch_and_write(resolved_codes, resolved_start, resolved_end, result, task)
            self._compensate(resolved_start, resolved_end, result, task)

            if result.rows_written > 0:
                result.data_version = self._next_data_version(anchor)
                self.store.set_data_version(self.dataset, result.data_version)
            self._save_reference(stock_info, task)

            task.update(
                start=resolved_start.isoformat(),
                end=resolved_end.isoformat(),
                requested_codes=result.requested_codes,
                rows_fetched=result.rows_fetched,
                rows_written=result.rows_written,
                flushes=result.flushes,
                empty_codes=list(result.empty_codes),
                failed=[{"code": item.code, "reason": item.reason} for item in result.failed],
                data_version=result.data_version,
            )
            if result.failed:
                _logger.error("同步存在失败标的：%s", [item.code for item in result.failed])
            return result

    # ============================================================ 取数与落盘
    def _fetch_and_write(
        self,
        codes: Sequence[str],
        start: date,
        end: date,
        result: SyncResult,
        task: TaskLogger,
    ) -> None:
        buffer: list[pd.DataFrame] = []
        empty: list[str] = []
        pending: dict[str, str] = {}

        for code in codes:
            try:
                frame = self.source.daily([code], start, end, adj=self.adj)
            except DataSourceError as exc:
                pending[code] = str(exc)
                continue
            if frame.empty:
                empty.append(code)
                continue
            buffer.append(frame)
            result.rows_fetched += len(frame)
            if len(buffer) >= self.flush_every:
                result.flushes += 1
                result.rows_written += self._flush(buffer, task)

        if buffer:
            result.flushes += 1
            result.rows_written += self._flush(buffer, task)

        result.empty_codes = tuple(empty)
        self._pending = pending

    def _flush(self, buffer: list[pd.DataFrame], task: TaskLogger) -> int:
        merged = pd.concat(buffer, ignore_index=True)
        buffer.clear()
        write_result = self.store.write(self.dataset, merged)
        _logger.info(
            "落盘 %s 行（输入 %s 行，去重 %s 行）",
            write_result.written_rows,
            write_result.input_rows,
            write_result.duplicates_dropped,
        )
        task.update(last_flush_rows=write_result.written_rows)
        return write_result.written_rows

    # -------------------------------------------------- 补偿
    def _compensate(
        self,
        start: date,
        end: date,
        result: SyncResult,
        task: TaskLogger,
    ) -> None:
        """对首轮失败的标的做补偿重试；仍失败才计入结果。"""
        pending = dict(self._pending)
        for round_index in range(1, self.compensation_rounds + 1):
            if not pending:
                break
            _logger.warning("第 %s 轮补偿重试，待重试标的 %s 只", round_index, len(pending))
            retry_codes = list(pending)
            pending = {}
            buffer: list[pd.DataFrame] = []
            for code in retry_codes:
                try:
                    frame = self.source.daily([code], start, end, adj=self.adj)
                except DataSourceError as exc:
                    pending[code] = str(exc)
                    continue
                if frame.empty:
                    result.empty_codes = (*result.empty_codes, code)
                    continue
                buffer.append(frame)
                result.rows_fetched += len(frame)
            if buffer:
                result.flushes += 1
                result.rows_written += self._flush(buffer, task)
            task.update(compensation_rounds_used=round_index)

        result.failed = tuple(SyncFailure(code=code, reason=reason) for code, reason in sorted(pending.items()))
        task.update(pending_after_compensation=len(pending))

    # ============================================================ 区间与版本
    def _resolve_range(
        self,
        *,
        start: date | None,
        end: date | None,
        anchor: date,
    ) -> tuple[date, date]:
        resolved_end = end or anchor
        if start is not None:
            return start, resolved_end

        coverage = self.store.coverage(self.dataset)
        if coverage.end is not None:
            return coverage.end + timedelta(days=1), resolved_end
        if self.full_backfill_start is None:
            raise ValidationError(
                "本地无数据且未指定回填起点，拒绝自动拉取长历史",
                detail={"hint": "显式传入 start，或设置 full_backfill_start"},
            )
        return self.full_backfill_start, resolved_end

    def _next_data_version(self, anchor: date) -> str:
        prefix = f"{anchor:%Y%m%d}"
        current = self.store.data_version(self.dataset)
        sequence = 1
        if current and current.startswith(f"{prefix}."):
            try:
                sequence = int(current.split(".", 1)[1]) + 1
            except ValueError:
                sequence = 1
        return f"{prefix}.{sequence}"

    def _source_calendar(self) -> TradingCalendar | None:
        """数据源自带日历时直接复用（AkShare 源有，内存源也有）。"""
        getter = getattr(self.source, "trading_calendar", None)
        if getter is None:
            return None
        try:
            calendar: TradingCalendar = getter()
        # 刻意兜底所有异常：日历是可选增强，拉不到不应阻断同步（降级为不做交易日判断）
        except Exception:
            _logger.warning("获取交易日历失败，本次同步跳过交易日判断", exc_info=True)
            return None
        return calendar

    def _stock_info_frame(self) -> pd.DataFrame | None:
        """取股票基础信息；失败时返回 None（是否致命由调用方决定）。"""
        try:
            frame = self.source.stock_info()
        # 刻意兜底：基础信息是可选增强，拉不到不应让整次同步失败
        except Exception:
            _logger.warning("获取股票基础信息失败", exc_info=True)
            return None
        return None if frame.empty else frame

    def _all_codes(self, info: pd.DataFrame | None) -> list[str]:
        if info is None or "code" not in info.columns:
            raise ValidationError(
                "未能获取股票基础信息，无法确定同步标的范围",
                detail={
                    "hint": (
                        "数据源拉不到股票列表时，显式指定标的即可（sync --codes 600000,000001）；"
                        "调度场景建议固定标的列表，避免因上游接口变动让盘前任务整体失败"
                    )
                },
            )
        codes = info["code"].astype(str).tolist()
        # 已退市标的没有新行情，排除以免每次都失败进补偿队列
        if "delist_date" in info.columns:
            listed = info["delist_date"].isna()
            codes = [code for code, keep in zip(codes, listed, strict=True) if keep]
        return sorted(set(codes))

    def _save_reference(self, info: pd.DataFrame | None, task: TaskLogger) -> None:
        if self.reference_store is None or info is None:
            return
        path = self.reference_store.save(STOCK_INFO, info)
        task.update(stock_info_path=str(path))


# ============================================================ 成分快照
DEFAULT_INDEX_CODES: tuple[str, ...] = ("000300", "000905", "000852")


@dataclass(frozen=True, slots=True)
class IndexMembersResult:
    as_of: date
    added: Mapping[str, int] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    empty: tuple[str, ...] = ()
    failed: tuple[SyncFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.empty and not self.failed


@dataclass
class IndexMembersSyncTask:
    """抓取指数**当前**成分并以抓取日为生效日登记。

    为什么必须以抓取日为生效日：akshare 等免费接口只提供"今天的成分"，
    没有历史成分。如果拿今天的成分去回填三年前，回测就会用"今天才知道的赢家"，
    这是典型的未来函数。因此本任务只做"打点"：每次运行留一个带日期的快照，
    跑得越久，历史越完整；没有快照的日期宁可报错也不用最新成分兜底。
    """

    source: DataSource
    history: UniverseHistory
    store: JsonUniverseStore
    task_log: TaskLogStore
    index_codes: Sequence[str] = DEFAULT_INDEX_CODES
    task_name: str = "sync_index_members"

    def run(self, *, on: date | None = None) -> IndexMembersResult:
        as_of = on or date.today()
        added: dict[str, int] = {}
        skipped: list[str] = []
        empty: list[str] = []
        failed: list[SyncFailure] = []

        with TaskLogger(self.task_log, self.task_name, trade_date=as_of) as task:
            for index_code in self.index_codes:
                if self.history.effective_date(index_code, as_of) == as_of:
                    # 当日已有快照：幂等跳过，避免重复打点与无谓的接口请求
                    skipped.append(index_code)
                    continue
                try:
                    members = tuple(self.source.index_members(index_code))
                except DataSourceError as exc:
                    failed.append(SyncFailure(code=index_code, reason=str(exc)))
                    continue
                if not members:
                    empty.append(index_code)
                    continue
                self.history.add(index_code, as_of, members)
                added[index_code] = len(members)

            if added:
                self.store.save(self.history)

            result = IndexMembersResult(
                as_of=as_of,
                added=added,
                skipped=tuple(skipped),
                empty=tuple(empty),
                failed=tuple(failed),
            )
            task.update(
                as_of=as_of.isoformat(),
                added=dict(added),
                skipped=list(skipped),
                empty=list(empty),
                failed=[{"code": item.code, "reason": item.reason} for item in failed],
            )
            if not added and skipped and not empty and not failed:
                task.skip("当日成分快照已存在")
            if empty or failed:
                _logger.error("指数成分快照存在问题：空 %s，失败 %s", empty, [item.code for item in failed])
            return result


__all__ = [
    "DEFAULT_INDEX_CODES",
    "DEFAULT_TASK_NAME",
    "DailySyncTask",
    "IndexMembersResult",
    "IndexMembersSyncTask",
    "SyncFailure",
    "SyncResult",
]
