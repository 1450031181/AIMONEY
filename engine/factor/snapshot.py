"""因子快照（P2-F5）：预计算因子值按日切片，喂给策略与回测。

**为什么不是把整张因子表交给策略**：策略若能自由查询任意日期，
"只用当时可见的数据"就只是一句口头约定，迟早有人写出 `factors[["date"] > today]`。
把它切成"当日可见"的视图后，越界在**接口上不存在**——与 ``bars_today`` 的处理方式完全一致。

快照里存的是**截面标准化后**的因子值（winsorize → rank → zscore）。
原因：多因子合成要把不同量纲的因子加权相加，原始值直接相加等于让量纲大的因子独占权重。

因果性说明：本项目全部内置因子都只用**向后看**的滚动/位移窗口计算，
因此 ``date=t`` 的因子值天然只依赖 ``≤ t`` 的数据，快照不会引入未来信息。
新增因子若违反该约定（例如使用了 ``shift(-n)``），必须在此处被拦截——
``assert_causal_factors`` 提供了一个可执行的检查入口。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from engine.errors import ValidationError
from engine.factor.preprocess import DEFAULT_STEPS

MISSING = None


@dataclass
class FactorSnapshot:
    """``(date, code) → {factor: value}`` 的只读视图（NaN 不进入快照）。"""

    frame: pd.DataFrame
    _by_day: dict[date, dict[str, dict[str, float]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        missing = [column for column in ("date", "code") if column not in self.frame.columns]
        if missing:
            raise ValidationError("因子快照缺少必要列", detail={"missing": missing})
        if self.frame.empty:
            return
        normalized = self.frame.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        factor_columns = [column for column in normalized.columns if column not in ("date", "code")]
        for day, group in normalized.groupby("date", sort=True):
            per_code: dict[str, dict[str, float]] = {}
            for _, row in group.iterrows():
                values = {}
                for name in factor_columns:
                    value = row[name]
                    if pd.notna(value):
                        values[name] = float(value)
                if values:
                    per_code[str(row["code"])] = values
            # groupby 的键被存根标注为 Hashable，先转字符串再转 Timestamp
            self._by_day[pd.Timestamp(str(day)).date()] = per_code

    # -------------------------------------------------- 元信息
    @property
    def is_empty(self) -> bool:
        return not self._by_day

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(column for column in self.frame.columns if column not in ("date", "code"))

    def dates(self) -> tuple[date, ...]:
        return tuple(sorted(self._by_day))

    # -------------------------------------------------- 按日访问
    def values_on(self, day: date) -> dict[str, dict[str, float]]:
        """当日可见的 ``code → {factor: value}``；无当日数据返回空（调用方自行决定是否跳过）。"""
        return dict(self._by_day.get(day, {}))

    def has_day(self, day: date) -> bool:
        return day in self._by_day

    def value(self, name: str, code: str, *, on: date) -> float | None:
        """单点取值（测试与按需查询用；策略侧应经 Context 访问以保留越界保护）。"""
        return self._by_day.get(on, {}).get(code, {}).get(name, MISSING)


def build_snapshot(values: pd.DataFrame, *, factors: Sequence[str]) -> FactorSnapshot:
    """由因子值长表（``date``/``code``/因子列）构建快照。"""
    missing = [name for name in factors if name not in values.columns]
    if missing:
        raise ValidationError("因子列不存在", detail={"missing": missing})
    return FactorSnapshot(frame=values.loc[:, ["date", "code", *factors]])


def build_snapshot_from_prices(
    prices: pd.DataFrame,
    *,
    factors: Sequence[str],
    preprocess_steps: Sequence[str] = DEFAULT_STEPS,
) -> FactorSnapshot:
    """一步到位：行情长表 → 计算因子 → 截面预处理 → 快照。

    **注意调用方需要自行准备预热期数据**：若 ``prices`` 只覆盖 ``[start, end]``，
    那么 ``start`` 之后的前 ``window`` 个交易日的因子值会是 NaN（进而被策略跳过）。
    因子服务与回测服务都通过 ``calendar.warmup_start()`` 前推读取起点来避免这个问题。
    """
    from engine.factor.engine import compute_factors
    from engine.factor.preprocess import PreprocessConfig, preprocess_cross_section

    values = compute_factors(prices, list(factors))
    values["date"] = pd.to_datetime(values["date"])
    processed = preprocess_cross_section(values, list(factors), PreprocessConfig(steps=tuple(preprocess_steps)))
    return build_snapshot(processed, factors=factors)


def assert_causal_factors(
    prices: pd.DataFrame,
    factors: Sequence[str],
    *,
    codes: Sequence[str] | None = None,
) -> None:
    """因果性抽查：把价格序列截断到 ``t`` 前后分别计算，值应完全一致。

    这是给"新因子是否偷看未来"准备的自动化检查入口：任何使用 ``shift(-n)``、
    全局均值、未来窗口的因子都会在截断后得到不同的值，从而在此暴露。

    代价是每个因子要算两次（截断前/截断后），因此只在因子注册与校验流程调用，
    不进入日常计算路径。
    """
    from engine.factor.engine import compute_factors

    if prices.empty:
        raise ValidationError("因果性检查需要非空行情")
    ordered = prices.copy()
    ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
    days = sorted(pd.Index(ordered["date"]).unique())
    if len(days) < 3:
        raise ValidationError("行情过短，无法进行因果性检查", detail={"days": len(days)})

    cutoff = days[len(days) // 2]
    selected = list(codes) if codes else sorted(ordered["code"].astype(str).unique())
    scoped = ordered.loc[ordered["code"].astype(str).isin(selected)]

    full = compute_factors(scoped, list(factors))
    truncated = compute_factors(scoped.loc[scoped["date"] <= cutoff], list(factors))

    full["date"] = pd.to_datetime(full["date"])
    truncated["date"] = pd.to_datetime(truncated["date"])
    left = full.loc[full["date"] == cutoff, ["code", *factors]].set_index("code").sort_index()
    right = truncated.loc[truncated["date"] == cutoff, ["code", *factors]].set_index("code").sort_index()

    for name in factors:
        pair = pd.concat([left[name], right[name]], axis=1, keys=["full", "truncated"])
        both_nan = pair["full"].isna() & pair["truncated"].isna()
        same = (pair["full"] == pair["truncated"]) | both_nan
        if not bool(same.all()):
            raise ValidationError(
                "因子使用了未来信息（截断前后取值不一致）",
                detail={"factor": name, "cutoff": cutoff.date().isoformat(), "mismatch": int((~same).sum())},
            )


def snapshot_summary(snapshot: FactorSnapshot) -> dict[str, Any]:
    """快照摘要（日志与报告用）。"""
    return {
        "factor_names": list(snapshot.factor_names),
        "days": len(snapshot.dates()),
        "first_day": snapshot.dates()[0].isoformat() if not snapshot.is_empty else None,
        "last_day": snapshot.dates()[-1].isoformat() if not snapshot.is_empty else None,
    }


__all__ = [
    "MISSING",
    "FactorSnapshot",
    "assert_causal_factors",
    "build_snapshot",
    "build_snapshot_from_prices",
    "snapshot_summary",
]
