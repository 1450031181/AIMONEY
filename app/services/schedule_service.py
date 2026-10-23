"""盘前/盘后调度（P4-3）：把"每天一条命令"接上交易日历守卫。

它不引入常驻进程，而是提供**可被外部调度器调用**的一次性任务
（Windows 任务计划 / cron / APScheduler 都可以）：每次 `run(job, day)` 是幂等的，退出码即契约。

两条任务链（顺序即正确性）：

    盘前 PRE_MARKET ：同步最新行情 → 质量校验（决定门禁开关）
    盘后 POST_MARKET：跑模拟盘一个交易日 → 查单校准

三个必须做对的地方：

1. **交易日守卫**：非交易日直接跳过（退出码 3）。缺了它，周末与节假日会跑到空任务链上，
   日志塞满"无数据"噪声，真正的问题反而被淹没。
2. **同步失败不阻断质量校验**：同步失败说明数据可能是旧的，但门禁必须照常更新——
   否则门禁会停留在旧状态，出现"数据坏了却还开着闸门"这种最危险的组合。
   因此同步失败记为 FAILED 并让整体退出码非零，但**继续**跑质量校验。
3. **盘后校准独立于策略运行**：校准只对账、不下单。即使当天没有跑策略（或跑失败），
   只要账本存在就应当校准——账本可能因昨天的异常而分叉。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.services.paper_service import PaperRunReport, PaperRunRequest, PaperTradingService
from app.services.report_service import DailyReportRequest, ReportService
from engine.datahub.quality import QualityCheckTask, QualityThresholds, TradeGate
from engine.datahub.reference import ReferenceStore
from engine.datahub.store import ParquetStore
from engine.datahub.sync import DailySyncTask
from engine.datahub.tasklog import JsonlTaskLog
from engine.datasource.local_source import LocalDataSource
from engine.errors import AimoneyError, NotFoundError, ValidationError
from engine.ports import DataSource, TradingCalendar

_logger = logging.getLogger(__name__)

TASKS_LOG = "tasks.jsonl"
DEFAULT_DATASET = "daily"


class JobKind(StrEnum):
    PRE_MARKET = "pre"
    POST_MARKET = "post"


class StepStatus(StrEnum):
    OK = "OK"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class StepResult:
    name: str
    status: StepStatus
    detail: str = ""
    duration_ms: int = 0

    @property
    def failed(self) -> bool:
        return self.status is StepStatus.FAILED

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
        }


@dataclass
class JobReport:
    job: JobKind
    trade_date: date
    status: StepStatus
    steps: list[StepResult] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not StepStatus.FAILED

    @property
    def skipped(self) -> bool:
        return self.status is StepStatus.SKIPPED

    def step(self, name: str) -> StepResult | None:
        return next((item for item in self.steps if item.name == name), None)

    def summary(self) -> str:
        if self.skipped and not self.steps:
            return f"[{self.job.value}] {self.trade_date} 跳过：{self.skip_reason}"
        marks = {StepStatus.OK: "✓", StepStatus.SKIPPED: "－", StepStatus.FAILED: "✗"}
        body = "；".join(
            f"{step.name} {marks[step.status]}{f'（{step.detail}）' if step.detail else ''}" for step in self.steps
        )
        head = f"[{self.job.value}] {self.trade_date} {self.status.value}：{body}"
        return head if self.ok else f"{head}  ⚠ 存在失败步骤"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "job": self.job.value,
            "trade_date": self.trade_date.isoformat(),
            "status": self.status.value,
            "ok": self.ok,
            "skip_reason": self.skip_reason,
            "steps": [step.to_jsonable() for step in self.steps],
        }


@dataclass
class ScheduleService:
    """盘前/盘后任务编排（只依赖 Settings + 可注入的数据源工厂）。"""

    settings: Settings
    #: 数据源工厂（测试注入内存替身；生产环境默认走 AkShare）
    source_factory: Callable[[], DataSource] | None = None
    dataset: str = DEFAULT_DATASET
    #: 盘后要跑的模拟盘策略目录（None 表示只做校准）
    strategy_dir: Path | None = None
    codes: tuple[str, ...] = ()
    params_override: dict[str, Any] = field(default_factory=dict)
    factors: tuple[str, ...] = ()

    # ============================================================ 日历
    def calendar(self) -> TradingCalendar:
        """交易日历：**优先用数据源**，本地行情推导只作降级兜底。

        为什么不能只用本地行情反推（这曾导致一个死锁）：本地日历等于"行情数据里出现过的日期"，
        而**盘前那一刻今天的数据本来就不存在**——于是"今天不是交易日"成立，
        盘前任务被守卫拦下，数据永远同步不了，明天同样如此。数据源才是交易日的权威。

        数据源不可用（网络故障/限流）时降级为本地推导并打 WARNING：
        此时"今天"可能被误判，但至少不会让历史区间的调度完全停摆。
        """
        source = self._optional_source()
        if source is not None:
            try:
                return source.trading_calendar()
            except AimoneyError as exc:
                _logger.warning("数据源日历不可用（%s），降级为本地行情推导的日历", exc)
        store = ParquetStore(self.settings.parquet_dir)
        store.require_data(self.dataset)
        return LocalDataSource(store, dataset=self.dataset).trading_calendar()

    def _optional_source(self) -> DataSource | None:
        """构造数据源（仅当显式提供工厂时）。

        刻意**不默认**构造 AkShare：`plan` 这类只读命令不该隐式发起网络请求。
        CLI 会显式注入 ``build_source``，测试注入内存替身。
        """
        if self.source_factory is None:
            return None
        try:
            return self.source_factory()
        except AimoneyError as exc:  # pragma: no cover —— 工厂自身失败
            _logger.warning("数据源构造失败（%s），降级为本地日历", exc)
            return None

    def is_trading_day(self, day: date) -> bool:
        return self.calendar().is_trade_day(day)

    def upcoming_run_days(self, *, start: date, days: int = 5) -> tuple[date, ...]:
        """从 ``start``（含）起接下来会真正触发任务的交易日（供运维核对）。"""
        if days <= 0:
            return ()
        # 只用 TradingCalendar 协议上的方法（sessions/between），不依赖实现类的 last_day：
        # 换了日历实现（如接券商日历）这里不该需要改动。
        sessions = tuple(day for day in self.calendar().sessions() if day >= start)
        return sessions[:days]

    def last_known_session(self) -> date | None:
        """日历里最后一个已知交易日（用于提示"日历覆盖到哪天"）。"""
        sessions = self.calendar().sessions()
        return sessions[-1] if sessions else None

    # ============================================================ 主入口
    def run(self, job: JobKind, *, day: date | None = None) -> JobReport:
        target = day or date.today()
        if not self.is_trading_day(target):
            # 守卫放在最前面：非交易日连数据同步都不该触发
            report = JobReport(
                job=job,
                trade_date=target,
                status=StepStatus.SKIPPED,
                skip_reason="非交易日（交易日历守卫）",
            )
            _logger.info("%s", report.summary())
            return report
        return self._pre_market(target) if job is JobKind.PRE_MARKET else self._post_market(target)

    # ============================================================ 盘前
    def _pre_market(self, day: date) -> JobReport:
        # 1) 同步：失败不阻断后续质量校验（见模块 docstring 第 2 条）
        sync_step = self._step("sync", lambda: self._sync(day))
        # 2) 质量校验 → 决定门禁开关
        check_step = self._step("check", lambda: self._check(day))

        steps = [sync_step, check_step]
        report = JobReport(job=JobKind.PRE_MARKET, trade_date=day, status=StepStatus.OK, steps=steps)
        if any(step.failed for step in steps):
            report.status = StepStatus.FAILED
        _logger.info("%s", report.summary())
        return report

    def _sync(self, day: date) -> tuple[StepStatus, str]:
        task = DailySyncTask(
            source=self._source(),
            store=ParquetStore(self.settings.parquet_dir),
            task_log=self._task_log(),
            dataset=self.dataset,
            reference_store=ReferenceStore(Path(str(self.settings.data.root_dir)) / "reference"),
        )
        result = task.run(codes=list(self.codes) or None, end=day, trade_date=day)
        if result.failed:
            failed = ", ".join(item.code for item in result.failed[:5])
            return StepStatus.FAILED, f"部分标的失败：{failed}（成功写入 {result.rows_written} 行）"
        if result.skipped:
            # "本地已是最新、无需同步"是**幂等空转，不是故障**。
            # 若当成失败，定时任务从第二天起每天都会误告警，监控很快就会被脱敏。
            return StepStatus.OK, f"无需同步（{result.skip_reason}）"
        detail = f"写入 {result.rows_written} 行，版本 {result.data_version or '-'}"
        if result.empty_codes:
            detail += f"，{len(result.empty_codes)} 只无数据"
        return StepStatus.OK, detail

    def _check(self, day: date) -> tuple[StepStatus, str]:
        task = QualityCheckTask(
            store=ParquetStore(self.settings.parquet_dir),
            gate=TradeGate(self.settings.gate_file),
            task_log=self._task_log(),
            report_dir=self.settings.quality_dir,
            thresholds=QualityThresholds.from_mapping(self.settings.data.quality.model_dump()),
            dataset=self.dataset,
            close_gate_on_fail=bool(self.settings.data.quality.trade_gate_on_fail),
        )
        report = task.run(as_of=day, expected_codes=list(self.codes) or None)
        gate = "开启" if TradeGate(self.settings.gate_file).is_open() else "关闭"
        # 质量 FAIL **不算调度失败**：检查本身成功执行了，结论是"今天不该交易"，
        # 而门禁关闭已经拦住了交易。若这里也报错，真正的执行故障会被刷屏噪声淹没。
        return StepStatus.OK, f"{report.status.value}，门禁{gate}"

    # ============================================================ 盘后
    def _post_market(self, day: date) -> JobReport:
        steps: list[StepResult] = []
        if self.strategy_dir is None:
            steps.append(StepResult(name="paper", status=StepStatus.SKIPPED, detail="未指定策略目录"))
        else:
            steps.append(self._step("paper", lambda: self._paper(day)))
        # 校准独立于策略运行：只要账本存在就应当对账（账本可能因昨天的异常而分叉）
        if self.strategy_dir is None:
            steps.append(StepResult(name="reconcile", status=StepStatus.SKIPPED, detail="未指定策略目录"))
        else:
            steps.append(self._step("reconcile", lambda: self._reconcile(day)))
        # 日报放在最后：它只是把已落盘的事实整理成人能读的形式，失败不影响交易与对账
        if self.strategy_dir is None:
            steps.append(StepResult(name="report", status=StepStatus.SKIPPED, detail="未指定策略目录"))
        else:
            steps.append(self._step("report", lambda: self._daily_report(day)))

        report = JobReport(job=JobKind.POST_MARKET, trade_date=day, status=StepStatus.OK, steps=steps)
        if any(step.failed for step in steps):
            report.status = StepStatus.FAILED
        elif all(step.status is StepStatus.SKIPPED for step in steps):
            report.status = StepStatus.SKIPPED
            report.skip_reason = "所有步骤均跳过（未指定策略或该交易日已处理）"
        _logger.info("%s", report.summary())
        return report

    def _paper(self, day: date) -> tuple[StepStatus, str]:
        request = PaperRunRequest(
            strategy_dir=self._strategy_dir(),
            trade_date=day,
            codes=self.codes,
            params_override=dict(self.params_override),
            factors=self.factors,
        )
        report: PaperRunReport = PaperTradingService(self.settings).run(request)
        if report.skipped:
            return StepStatus.SKIPPED, report.skipped_reason or "跳过"
        return StepStatus.OK, report.summary()

    def _daily_report(self, day: date) -> tuple[StepStatus, str]:
        """生成当日报告。没有账本（门禁关闭/尚未跑过）时**跳过**，不当作失败。"""
        try:
            report = ReportService(self.settings).daily(
                DailyReportRequest(strategy_dir=self._strategy_dir(), trade_date=day)
            )
        except NotFoundError:
            return StepStatus.SKIPPED, "无模拟盘账本，跳过日报"
        return StepStatus.OK, f"{report.summary()}；报告目录 {report.output_dir}"

    def _reconcile(self, day: date) -> tuple[StepStatus, str]:
        request = PaperRunRequest(strategy_dir=self._strategy_dir(), trade_date=day, codes=self.codes)
        result = PaperTradingService(self.settings).reconcile(request)
        if result.checked == 0:
            return StepStatus.SKIPPED, "无账本或无可校准委托"
        return StepStatus.OK, result.summary()

    # ============================================================ 内部
    def _strategy_dir(self) -> Path:
        assert self.strategy_dir is not None  # 调用方已判空
        return self.strategy_dir

    def _source(self) -> DataSource:
        """同步步骤使用的数据源（必须存在；缺了直接报配置错误而不是静默跳过）。"""
        source = self._optional_source()
        if source is None:
            raise ValidationError(
                "未配置数据源，无法执行盘前同步",
                detail={"hint": "调度服务需要 source_factory；CLI 会自动注入 AkShare 数据源"},
            )
        return source

    def _task_log(self) -> JsonlTaskLog:
        return JsonlTaskLog(self.settings.log_dir / TASKS_LOG)

    def _step(self, name: str, action: Callable[[], tuple[StepStatus, str]]) -> StepResult:
        """执行一步并压成 ``StepResult``（异常也转成失败步骤，不让整条链崩掉）。"""
        started = time.monotonic()
        try:
            status, detail = action()
        except AimoneyError as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            _logger.error("任务步骤失败：%s | %s", name, exc)
            return StepResult(name=name, status=StepStatus.FAILED, detail=str(exc), duration_ms=elapsed)
        elapsed = int((time.monotonic() - started) * 1000)
        _logger.info("任务步骤完成：%s（%s ms）", name, elapsed)
        return StepResult(name=name, status=status, detail=detail, duration_ms=elapsed)


def ensure_strategy_dir(path: Path | str | None) -> Path:
    """校验策略目录存在（把配置错误挡在跑任务之前）。"""
    if path is None:
        raise ValidationError("缺少 --strategy 参数", detail={"hint": "指定包含 strategy.py 的策略目录"})
    directory = Path(path)
    if not directory.is_dir():
        raise ValidationError(
            "策略目录不存在",
            detail={"strategy_dir": str(directory), "hint": "用 --strategy 指定包含 strategy.py 的目录"},
        )
    return directory


__all__ = [
    "TASKS_LOG",
    "JobKind",
    "JobReport",
    "ScheduleService",
    "StepResult",
    "StepStatus",
    "ensure_strategy_dir",
]
