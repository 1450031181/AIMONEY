"""股票池 / 指数成分历史（T-06 提取，T-10 扩展）。

**为什么成分必须带生效日**：指数成分每半年调整一次，若用「当前成分」回填历史回测，
就会把"今天才知道的赢家"放进三年前的持仓——这是最隐蔽、也最常见的一类未来函数，
而且它在净值曲线上表现为"平稳地多赚一点"，很难事后发现。

因此本模块只按 ``effective_date <= on`` 解析成分；早于首次快照的日期**返回空**，
调用方（数据源适配器）据此抛 ``DataNotReadyError``，而不是静默用最新成分。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from engine.datahub.schema import STOCK_INFO_REQUIRED_COLUMNS, ensure_columns
from engine.errors import DataNotReadyError, ValidationError

_logger = logging.getLogger(__name__)


@dataclass
class UniverseHistory:
    """指数成分历史：一个指数对应若干「生效日 → 成分」快照（按生效日升序）。"""

    _snapshots: dict[str, list[tuple[date, tuple[str, ...]]]] = field(default_factory=dict, repr=False)

    def add(self, index_code: str, effective_date: date, codes: Iterable[str]) -> None:
        """登记一次成分快照（同一生效日重复登记以最后一次为准）。"""
        entries = self._snapshots.setdefault(index_code, [])
        entries[:] = [entry for entry in entries if entry[0] != effective_date]
        entries.append((effective_date, tuple(dict.fromkeys(codes))))
        entries.sort(key=lambda entry: entry[0])

    def indices(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    def has(self, index_code: str) -> bool:
        return bool(self._snapshots.get(index_code))

    def effective_date(self, index_code: str, on: date) -> date | None:
        """``on`` 当日生效的成分快照日期；不存在（早于首次快照）返回 None。"""
        candidates = [entry[0] for entry in self._snapshots.get(index_code, []) if entry[0] <= on]
        return max(candidates) if candidates else None

    def resolve(self, index_code: str, on: date) -> tuple[str, ...]:
        effective = self.effective_date(index_code, on)
        if effective is None:
            return ()
        for entry_date, codes in self._snapshots[index_code]:
            if entry_date == effective:
                return codes
        return ()

    def latest(self, index_code: str) -> tuple[str, ...]:
        entries = self._snapshots.get(index_code, [])
        return entries[-1][1] if entries else ()

    def snapshots(self, index_code: str) -> Sequence[tuple[date, tuple[str, ...]]]:
        return tuple(self._snapshots.get(index_code, []))


# ============================================================ 股票池筛选
#: 上市日缺失时的处置策略：
#: - warn（默认）：跳过该规则并记入 notes —— 明确可见，不会让人误以为过滤生效
#: - error：直接报错，适合对策略有强要求（如打新类策略）的场景
ListDatePolicy = Literal["warn", "error"]

RULE_DELISTED = "delisted"
RULE_ST = "st"
RULE_NEW_LISTING = "new_listing"
RULE_SUSPENDED = "suspended"
RULE_NO_INFO = "no_stock_info"
RULE_MISSING_LIST_DATE = "missing_list_date"


@dataclass(frozen=True, slots=True)
class UniverseFilter:
    """股票池筛选条件（对应策略 config.yaml 的 universe 段）。"""

    index_code: str | None = None
    exclude_st: bool = True
    exclude_delisted: bool = True
    exclude_suspended: bool = True
    #: 上市满 N 个自然日才纳入（次新股过滤）；0 表示不过滤
    min_list_days: int = 0
    list_date_policy: ListDatePolicy = "warn"

    def __post_init__(self) -> None:
        if self.min_list_days < 0:
            raise ValidationError("min_list_days 不能为负", detail={"min_list_days": self.min_list_days})


@dataclass(frozen=True, slots=True)
class UniverseSelection:
    """筛选结果。``dropped`` 与 ``notes`` 是**可观测性**的核心：

    策略只看到"30 只候选"，必须能回答"另外 270 只为什么被排除"，
    否则过滤出问题（例如某天全被 ST 规则剔掉）时完全无从排查。
    """

    as_of: date
    index_code: str | None
    codes: tuple[str, ...]
    dropped: Mapping[str, tuple[str, ...]]
    notes: tuple[str, ...] = ()

    @property
    def dropped_total(self) -> int:
        unique: set[str] = set()
        for codes in self.dropped.values():
            unique.update(codes)
        return len(unique)

    @property
    def rules_not_applied(self) -> tuple[str, ...]:
        """因数据缺失而**未生效**的规则（如上市日缺失导致次新过滤被跳过）。"""
        return tuple(note for note in self.notes if note.startswith("未生效:"))

    def summary(self) -> str:
        parts = [f"{len(self.codes)} 只入选（{self.as_of.isoformat()}"]
        if self.index_code:
            parts.append(f"，{self.index_code}")
        parts.append(f"，排除 {self.dropped_total} 只)")
        if self.rules_not_applied:
            parts.append("；" + "；".join(self.rules_not_applied))
        return "".join(parts)


def select_universe(
    history: UniverseHistory,
    *,
    as_of: date,
    stock_info: pd.DataFrame,
    universe_filter: UniverseFilter | None = None,
    suspended: Iterable[str] = (),
) -> UniverseSelection:
    """按筛选条件产出当日股票池。

    ``suspended`` 由调用方传入当日停牌集合（来自行情数据，而不是基础信息表），
    这样"停牌"判定与真实交易状态一致。
    """
    resolved = universe_filter or UniverseFilter()
    ensure_columns(stock_info, STOCK_INFO_REQUIRED_COLUMNS, dataset="stock_info")
    if stock_info.empty:
        raise ValidationError("股票基础信息为空，无法构建股票池")

    if resolved.index_code:
        effective = history.effective_date(resolved.index_code, as_of)
        if effective is None:
            raise DataNotReadyError(
                "该日期没有可用的指数成分快照，禁止用最新成分回填历史",
                detail={"index_code": resolved.index_code, "as_of": as_of.isoformat()},
            )
        candidates = list(history.resolve(resolved.index_code, as_of))
    else:
        candidates = stock_info["code"].astype(str).tolist()

    info = stock_info.copy()
    info["code"] = info["code"].astype(str)
    lookup = info.drop_duplicates(subset=["code"], keep="last").set_index("code")

    also_suspended = {str(code) for code in suspended}
    dropped: dict[str, list[str]] = {rule: [] for rule in (
        RULE_NO_INFO,
        RULE_DELISTED,
        RULE_ST,
        RULE_SUSPENDED,
        RULE_NEW_LISTING,
        RULE_MISSING_LIST_DATE,
    )}
    notes: list[str] = []
    has_list_date = "list_date" in lookup.columns
    has_is_st = "is_st" in lookup.columns
    has_delist_date = "delist_date" in lookup.columns
    if resolved.min_list_days > 0 and not has_list_date:
        notes.append("未生效:次新过滤（基础信息缺少 list_date 列）")
    if resolved.exclude_st and not has_is_st:
        notes.append("未生效:ST 过滤（基础信息缺少 is_st 列）")

    missing_list_dates: list[str] = []
    selected: list[str] = []
    for code in candidates:
        if code not in lookup.index:
            dropped[RULE_NO_INFO].append(code)
            continue
        row = lookup.loc[code]
        if isinstance(row, pd.DataFrame):  # 理论上不会出现，防御性处理
            row = row.iloc[0]

        if resolved.exclude_delisted and has_delist_date and pd.notna(row.get("delist_date")):
            delist_date = pd.to_datetime(row["delist_date"]).date()
            if delist_date <= as_of:
                dropped[RULE_DELISTED].append(code)
                continue

        if resolved.exclude_st and has_is_st and bool(row.get("is_st", False)):
            dropped[RULE_ST].append(code)
            continue

        if resolved.exclude_suspended and code in also_suspended:
            dropped[RULE_SUSPENDED].append(code)
            continue

        if resolved.min_list_days > 0 and has_list_date:
            list_date = row.get("list_date")
            if pd.isna(list_date):
                missing_list_dates.append(code)
                if resolved.list_date_policy == "error":
                    dropped[RULE_MISSING_LIST_DATE].append(code)
                    continue
            else:
                listed_days = (as_of - pd.to_datetime(list_date).date()).days
                if listed_days < resolved.min_list_days:
                    dropped[RULE_NEW_LISTING].append(code)
                    continue

        selected.append(code)

    if missing_list_dates and resolved.list_date_policy == "warn":
        notes.append(
            f"未生效:次新过滤（{len(missing_list_dates)} 只标的缺少上市日，已按『不排除』处理）"
        )
        _logger.warning(
            "次新过滤未生效：%s 只标的缺少 list_date。补齐数据源前该规则形同虚设。",
            len(missing_list_dates),
        )

    return UniverseSelection(
        as_of=as_of,
        index_code=resolved.index_code,
        codes=tuple(sorted(selected)),
        dropped={rule: tuple(codes) for rule, codes in dropped.items() if codes},
        notes=tuple(notes),
    )


# ============================================================ 成分历史持久化
class JsonUniverseStore:
    """成分历史的 JSON 持久化（P0 用；接 PostgreSQL ``dim_index_member`` 后替换）。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> UniverseHistory:
        history = UniverseHistory()
        if not self.path.is_file():
            return history
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValidationError("成分历史文件格式非法", detail={"path": str(self.path)})
        for index_code, snapshots in payload.items():
            for snapshot in snapshots:
                history.add(index_code, date.fromisoformat(str(snapshot["effective_date"])), snapshot["codes"])
        return history

    def save(self, history: UniverseHistory) -> Path:
        payload: dict[str, list[dict[str, Any]]] = {
            index_code: [
                {"effective_date": effective_date.isoformat(), "codes": list(codes)}
                for effective_date, codes in history.snapshots(index_code)
            ]
            for index_code in history.indices()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return self.path


__all__ = [
    "RULE_DELISTED",
    "RULE_MISSING_LIST_DATE",
    "RULE_NEW_LISTING",
    "RULE_NO_INFO",
    "RULE_ST",
    "RULE_SUSPENDED",
    "JsonUniverseStore",
    "ListDatePolicy",
    "UniverseFilter",
    "UniverseHistory",
    "UniverseSelection",
    "select_universe",
]
