"""数据质量校验与交易门禁（T-11）。

这是 P0 里唯一直接决定"今天能不能交易"的模块。设计原则一句话：
**宁可当日空仓，也不拿脏数据下单。**

六类规则（前四类 FAIL 即关门禁，后两类仅告警）：

| 规则 | 严重度 | 为什么 |
| --- | --- | --- |
| 数据为空 | FAIL | 没有数据就没有策略，谈不上"降级使用" |
| 缺失率超限 | FAIL | 全市场大面积缺数据往往意味着接口改版或限流，不是个别标的的问题 |
| 价格非正 | FAIL | 0 价会让市值/收益率计算出现除零与无穷大 |
| 主键重复 | FAIL | 重复行会让成交量、成交额、因子权重全部翻倍 |
| 复权连续性 | FAIL | 漏乘复权因子会造出"平稳多赚一点"的假收益，肉眼无法发现 |
| 非停牌零成交 | WARN | 可能是停牌标记错误，先告警不阻断 |
| 成交量突增 | WARN | 可能是数据源口径变化（如单位从手变股），需人工确认 |

另外两条与 T-10 一致的工程约定：
- 规则缺少输入（如没传期望标的数）时，记入 ``rules_not_applied`` 并在日报里显式列出，
  绝不"看起来检查了、其实没检查"；
- 门禁状态持久化到文件（P0），默认**关闭**（fail-safe）：必须先跑质量校验才允许交易。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from engine.datahub.adjust import find_continuity_violations
from engine.datahub.jsonio import to_jsonable
from engine.datahub.schema import DAILY_PRIMARY_KEY
from engine.datahub.store import ParquetStore
from engine.datahub.tasklog import TaskLogger, TaskLogStore
from engine.errors import DataQualityError

DAILY_DATASET = "daily"
DEFAULT_TASK_NAME = "check_quality"
_VOLUME_BASELINE_WINDOW = 20
_VOLUME_MIN_PERIODS = 5


class QualityStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Severity(StrEnum):
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """与 ``app.core.config.QualitySection`` 一一对应（engine 不依赖 app，故此处独立定义）。"""

    max_missing_ratio: float = 0.01
    continuity_tolerance: float = 0.02
    max_volume_jump_ratio: float = 10.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> QualityThresholds:
        return cls(
            max_missing_ratio=float(mapping.get("max_missing_ratio", 0.01)),
            continuity_tolerance=float(mapping.get("continuity_tolerance", 0.02)),
            max_volume_jump_ratio=float(mapping.get("max_volume_jump_ratio", 10.0)),
        )


@dataclass(frozen=True, slots=True)
class QualityFinding:
    rule: str
    severity: Severity
    message: str
    count: int = 0
    samples: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class QualityReport:
    dataset: str
    as_of: date | None
    checked_rows: int
    checked_codes: int
    findings: tuple[QualityFinding, ...] = ()
    rules_not_applied: tuple[str, ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def status(self) -> QualityStatus:
        if any(finding.severity is Severity.FAIL for finding in self.findings):
            return QualityStatus.FAIL
        if any(finding.severity is Severity.WARN for finding in self.findings):
            return QualityStatus.WARN
        return QualityStatus.PASS

    @property
    def failed_rules(self) -> tuple[str, ...]:
        return tuple(finding.rule for finding in self.findings if finding.severity is Severity.FAIL)

    @property
    def warned_rules(self) -> tuple[str, ...]:
        return tuple(finding.rule for finding in self.findings if finding.severity is Severity.WARN)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "status": str(self.status),
            "checked_rows": self.checked_rows,
            "checked_codes": self.checked_codes,
            "failed_rules": list(self.failed_rules),
            "warned_rules": list(self.warned_rules),
            "rules_not_applied": list(self.rules_not_applied),
            "thresholds": to_jsonable(dict(self.thresholds)),
            "generated_at": self.generated_at.isoformat(timespec="seconds"),
            "findings": [
                {
                    "rule": finding.rule,
                    "severity": str(finding.severity),
                    "message": finding.message,
                    "count": finding.count,
                    # 样本可能含 pandas Timestamp / numpy 标量（从 Parquet 读出），必须归一化，
                    # 否则报告写盘会抛 TypeError —— 偏偏报告是排查 FAIL 的唯一依据
                    "samples": to_jsonable(list(finding.samples)),
                }
                for finding in self.findings
            ],
        }

    def summary(self) -> str:
        parts = [
            f"[{self.status}] {self.dataset} {self.as_of} "
            f"检查 {self.checked_codes} 只 / {self.checked_rows} 行"
        ]
        if self.failed_rules:
            parts.append(f"失败规则：{', '.join(self.failed_rules)}")
        if self.warned_rules:
            parts.append(f"告警规则：{', '.join(self.warned_rules)}")
        if self.rules_not_applied:
            parts.append("未生效：" + "；".join(self.rules_not_applied))
        return "；".join(parts)


# ============================================================ 各条规则
def _sample_records(
    frame: pd.DataFrame, mask: pd.Series, columns: Sequence[str], limit: int = 5
) -> tuple[dict[str, Any], ...]:
    """抽取违规样本行（键统一为 str，便于 JSON 序列化与人工阅读）。"""
    records = frame.loc[mask, list(columns)].head(limit).to_dict(orient="records")
    return tuple({str(key): value for key, value in record.items()} for record in records)


def _rule_empty(frame: pd.DataFrame) -> QualityFinding | None:
    if frame.empty:
        return QualityFinding(
            rule="empty_dataset",
            severity=Severity.FAIL,
            message="当日没有任何行情数据",
        )
    return None


def _rule_missing(
    frame: pd.DataFrame,
    expected_codes: Sequence[str] | None,
    *,
    max_missing_ratio: float,
) -> tuple[QualityFinding | None, bool]:
    """返回 ``(findings, 是否真的执行了检查)``。

    第二个返回值用于区分"检查通过"与"没提供期望标的、检查没做"——
    后者必须进 ``rules_not_applied``，否则会给人"缺失率已检查"的错觉。
    """
    if expected_codes is None:
        return None, False
    expected = {str(code) for code in expected_codes}
    if not expected:
        return None, False
    actual = set(frame["code"].astype(str).unique())
    missing = sorted(expected - actual)
    missing_ratio = len(missing) / len(expected)
    if missing_ratio <= max_missing_ratio:
        return None, True
    return (
        QualityFinding(
            rule="missing_codes",
            severity=Severity.FAIL,
            message=(
                f"缺失 {len(missing)}/{len(expected)} 只（{missing_ratio:.2%}），"
                f"超过阈值 {max_missing_ratio:.2%}"
            ),
            count=len(missing),
            samples=tuple({"code": code} for code in missing[:10]),
        ),
        True,
    )


def _rule_non_positive_price(frame: pd.DataFrame) -> QualityFinding | None:
    price_columns = [column for column in ("open", "high", "low", "close") if column in frame.columns]
    if not price_columns or frame.empty:
        return None
    values = frame[price_columns].apply(pd.to_numeric, errors="coerce")
    bad = (values <= 0).any(axis=1)
    if not bad.any():
        return None
    return QualityFinding(
        rule="non_positive_price",
        severity=Severity.FAIL,
        message=f"{int(bad.sum())} 行存在非正价格（价格 ≤ 0）",
        count=int(bad.sum()),
        samples=_sample_records(frame, bad, ["code", "date", *price_columns]),
    )


def _rule_duplicate_key(frame: pd.DataFrame) -> QualityFinding | None:
    if frame.empty:
        return None
    keys = [column for column in DAILY_PRIMARY_KEY if column in frame.columns]
    duplicated = frame.duplicated(subset=keys)
    if not duplicated.any():
        return None
    return QualityFinding(
        rule="duplicate_primary_key",
        severity=Severity.FAIL,
        message=f"存在 {int(duplicated.sum())} 行重复主键（{'+'.join(keys)}）",
        count=int(duplicated.sum()),
        samples=_sample_records(frame, duplicated, keys),
    )


def _rule_continuity(frame: pd.DataFrame, *, tolerance: float) -> QualityFinding | None:
    if frame.empty:
        return None
    violations = find_continuity_violations(frame, tolerance=tolerance)
    if not violations:
        return None
    return QualityFinding(
        rule="price_continuity",
        severity=Severity.FAIL,
        message=(
            f"{len(violations)} 处相邻交易日跳变超出该板块涨跌停限制，"
            "疑似复权因子缺失或价格未复权"
        ),
        count=len(violations),
        samples=tuple(
            {"code": item.code, "date": item.date, "jump_ratio": item.jump_ratio, "allowed": item.allowed_ratio}
            for item in violations[:5]
        ),
    )


def _rule_zero_volume_not_suspended(frame: pd.DataFrame) -> QualityFinding | None:
    if frame.empty or "volume" not in frame.columns or "is_suspended" not in frame.columns:
        return None
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    bad = (volume <= 0) & (~frame["is_suspended"].fillna(False).astype(bool))
    if not bad.any():
        return None
    return QualityFinding(
        rule="zero_volume_not_suspended",
        severity=Severity.WARN,
        message=f"{int(bad.sum())} 行成交量为 0 却未标记停牌，可能是停牌标记错误",
        count=int(bad.sum()),
        samples=_sample_records(frame, bad, ["code", "date", "volume"]),
    )


def _rule_volume_jump(
    frame: pd.DataFrame,
    *,
    max_ratio: float,
    window: int = _VOLUME_BASELINE_WINDOW,
    min_periods: int = _VOLUME_MIN_PERIODS,
) -> QualityFinding | None:
    if frame.empty or "volume" not in frame.columns:
        return None
    ordered = frame.sort_values(["code", "date"])
    volume = pd.to_numeric(ordered["volume"], errors="coerce")
    baseline = volume.groupby(ordered["code"]).transform(
        lambda series: series.shift(1).rolling(window, min_periods=min_periods).median()
    )
    ratio = volume / baseline
    bad = (baseline > 0) & (ratio > max_ratio)
    if not bad.any():
        return None
    return QualityFinding(
        rule="volume_jump",
        severity=Severity.WARN,
        message=f"{int(bad.sum())} 行成交量超过前 {window} 日中位数的 {max_ratio} 倍（疑似单位口径变化）",
        count=int(bad.sum()),
        samples=tuple(
            {
                "code": str(row["code"]),
                "date": str(row["date"]),
                "volume": float(row["volume"]),
                "ratio": round(float(row["_ratio"]), 2),
            }
            for _, row in ordered.assign(_ratio=ratio).loc[bad].head(5).iterrows()
        ),
    )


def check_quality(
    frame: pd.DataFrame,
    *,
    dataset: str = DAILY_DATASET,
    as_of: date | None = None,
    thresholds: QualityThresholds | None = None,
    expected_codes: Sequence[str] | None = None,
) -> QualityReport:
    """对一批行情数据执行全部质量规则，产出报告。"""
    resolved = thresholds or QualityThresholds()

    not_applied: list[str] = []
    findings: list[QualityFinding] = []

    empty = _rule_empty(frame)
    if empty is not None:
        findings.append(empty)
        return QualityReport(
            dataset=dataset,
            as_of=as_of,
            checked_rows=0,
            checked_codes=0,
            findings=tuple(findings),
            thresholds=_thresholds_mapping(resolved),
        )

    missing_finding, missing_applied = _rule_missing(
        frame, expected_codes, max_missing_ratio=resolved.max_missing_ratio
    )
    if not missing_applied:
        not_applied.append("缺失率检查（未提供期望标的列表）")
    elif missing_finding is not None:
        findings.append(missing_finding)

    for finding in (
        _rule_non_positive_price(frame),
        _rule_duplicate_key(frame),
        _rule_continuity(frame, tolerance=resolved.continuity_tolerance),
    ):
        if finding is not None:
            findings.append(finding)

    if "volume" in frame.columns and "is_suspended" not in frame.columns:
        not_applied.append("停牌一致性检查（缺少 is_suspended 列）")

    for finding in (
        _rule_zero_volume_not_suspended(frame),
        _rule_volume_jump(frame, max_ratio=resolved.max_volume_jump_ratio),
    ):
        if finding is not None:
            findings.append(finding)

    return QualityReport(
        dataset=dataset,
        as_of=as_of,
        checked_rows=len(frame),
        checked_codes=int(frame["code"].astype(str).nunique()) if "code" in frame.columns else 0,
        findings=tuple(findings),
        rules_not_applied=tuple(not_applied),
        thresholds=_thresholds_mapping(resolved),
    )


def _thresholds_mapping(thresholds: QualityThresholds) -> dict[str, float]:
    return {
        "max_missing_ratio": thresholds.max_missing_ratio,
        "continuity_tolerance": thresholds.continuity_tolerance,
        "max_volume_jump_ratio": thresholds.max_volume_jump_ratio,
    }


# ============================================================ 交易门禁
DEFAULT_GATE_FILE = "state/trade_gate.json"


class TradeGate:
    """交易门禁：状态持久化到 JSON 文件（P0；接 Redis 后替换实现即可）。

    **默认关闭**（fail-safe）：没有状态文件时 ``is_open()`` 返回 False。
    这意味着全新部署必须先跑一次质量校验才会开门——宁可少交易一天，
    也不能在"不知道数据是否可用"的情况下自动下单。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def state(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"open": False, "reason": "尚未执行数据质量校验", "as_of": None, "updated_at": None}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise DataQualityError("交易门禁状态文件格式非法", detail={"path": str(self.path)})
        return payload

    def is_open(self) -> bool:
        return bool(self.state().get("open", False))

    def open(self, *, as_of: date, note: str = "") -> None:
        self._write(open_=True, reason=note or "数据质量校验通过", as_of=as_of)

    def close(self, *, as_of: date, reason: str) -> None:
        self._write(open_=False, reason=reason, as_of=as_of)

    def _write(self, *, open_: bool, reason: str, as_of: date) -> None:
        payload = {
            "open": open_,
            "reason": reason,
            "as_of": as_of.isoformat(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)


# ============================================================ 质量校验任务
@dataclass
class QualityCheckTask:
    """盘前质量校验任务：跑规则 → 落报告 → 开关交易门禁。"""

    store: ParquetStore
    gate: TradeGate
    task_log: TaskLogStore
    report_dir: Path
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    dataset: str = DAILY_DATASET
    task_name: str = DEFAULT_TASK_NAME
    close_gate_on_fail: bool = True

    def run(
        self,
        *,
        as_of: date,
        expected_codes: Sequence[str] | None = None,
    ) -> QualityReport:
        with TaskLogger(self.task_log, self.task_name, trade_date=as_of) as task:
            frame = self.store.read(self.dataset, start=as_of, end=as_of)
            report = check_quality(
                frame,
                dataset=self.dataset,
                as_of=as_of,
                thresholds=self.thresholds,
                expected_codes=expected_codes,
            )
            report_path = self._write_report(report, as_of=as_of)

            if report.status is QualityStatus.FAIL and self.close_gate_on_fail:
                self.gate.close(
                    as_of=as_of,
                    reason=f"数据质量 FAIL：{', '.join(report.failed_rules)}",
                )
            elif report.status is not QualityStatus.FAIL:
                self.gate.open(as_of=as_of, note=f"数据质量 {report.status}")

            task.update(
                status=str(report.status),
                checked_rows=report.checked_rows,
                checked_codes=report.checked_codes,
                failed_rules=list(report.failed_rules),
                warned_rules=list(report.warned_rules),
                rules_not_applied=list(report.rules_not_applied),
                report_path=str(report_path),
                gate_open=self.gate.is_open(),
            )
            if report.status is QualityStatus.FAIL:
                task.update(gate_closed_reason="数据质量 FAIL")
            return report

    def _write_report(self, report: QualityReport, *, as_of: date) -> Path:
        path = Path(self.report_dir) / f"quality_{as_of:%Y%m%d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            tmp.write_text(
                json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return path


__all__ = [
    "DAILY_DATASET",
    "DEFAULT_GATE_FILE",
    "DEFAULT_TASK_NAME",
    "QualityCheckTask",
    "QualityFinding",
    "QualityReport",
    "QualityStatus",
    "QualityThresholds",
    "Severity",
    "TradeGate",
    "check_quality",
]
