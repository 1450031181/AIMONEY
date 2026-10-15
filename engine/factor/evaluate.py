"""因子评价（P2-F2）。

判据体系（与需求文档附录 A 一致）：

- **IC**：每个截面日算因子值与未来 ``horizon`` 日收益的相关性（默认 Spearman = Rank IC）。
  用秩相关而不是皮尔逊：因子分布厚尾、与收益多为单调而非线性关系。
- **ICIR**：``mean(IC) / std(IC)``。IC 均值高但波动大同样不可用，ICIR 才是稳定性指标。
- **分层回测**：按因子值把截面分成 Q 层，看各层未来收益是否单调。
  单调性比"多空收益高"更重要——它说明因子确实在给标的排序，而不是靠某一层运气。
- **多空组合**：顶层减底层的净值，理想情况下应稳定向上。
- **换手率**：顶层/底层成员变动带来的组合换手。高换手因子即使 IC 漂亮，
  扣掉交易成本后也可能一文不值，所以换手必须与 IC 一起看。

**唯一的未来函数例外**：评价必须使用未来收益（这正是"预测能力"的定义），
因此 ``forward_returns`` 会引用未来价格。这是**离线研究**场景的刻意设计，
绝不能出现在策略或回测的决策路径上——回测里读未来数据会抛 ``LookaheadError``。
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from engine.errors import ValidationError

SUPPORTED_IC_METHODS = ("spearman", "pearson")
#: 标准差小于该值视为常量（浮点误差会让 [0.1,0.1,0.1] 的 std 是 1e-17 而不是 0）
_ZERO_TOLERANCE = 1e-12


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """Spearman 秩相关 = 排名后的皮尔逊相关。

    自己实现而不是 ``Series.corr(method="spearman")``：后者会引入 scipy 依赖，
    而因子评价只用到这一个统计量，不值得为它拉进整个科学计算栈。
    """
    if len(left) < 3:
        return float("nan")
    ranked_left = left.rank()
    ranked_right = right.rank()
    if ranked_left.nunique() < 2 or ranked_right.nunique() < 2:
        return float("nan")
    return float(ranked_left.corr(ranked_right))


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    #: 未来收益的持有天数
    horizon: int = 1
    #: 分层数量（需求文档默认 5/10）
    quantiles: int = 5
    #: 单个截面至少多少标的才参与 IC / 分层（样本太少的相关性没有意义）
    min_cross_section: int = 5
    ic_method: str = "spearman"

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValidationError("horizon 必须为正", detail={"horizon": self.horizon})
        if self.quantiles < 2:
            raise ValidationError("分层数至少为 2", detail={"quantiles": self.quantiles})
        if self.min_cross_section < 3:
            raise ValidationError("最小截面样本数至少为 3", detail={"min_cross_section": self.min_cross_section})
        if self.ic_method not in SUPPORTED_IC_METHODS:
            raise ValidationError(
                "不支持的 IC 方法", detail={"method": self.ic_method, "supported": list(SUPPORTED_IC_METHODS)}
            )


@dataclass
class FactorEvaluation:
    factor: str
    horizon: int
    quantiles: int
    observations: int
    ic_mean: float
    ic_std: float
    icir: float
    ic_positive_ratio: float
    ic_series: pd.Series
    quantile_mean_returns: dict[int, float] = field(default_factory=dict)
    quantile_cum_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    long_short_mean_return: float = 0.0
    long_short_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    top_turnover: float = 0.0
    bottom_turnover: float = 0.0
    monotonicity: float = 0.0
    dropped_rows: int = 0

    @property
    def sufficient(self) -> bool:
        """是否有足够观测支撑结论。为 False 时任何"因子有效/无效"的判断都不成立。"""
        return self.observations > 0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "horizon": self.horizon,
            "quantiles": self.quantiles,
            "observations": self.observations,
            "sufficient": self.sufficient,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "ic_positive_ratio": self.ic_positive_ratio,
            "long_short_mean_return": self.long_short_mean_return,
            "top_turnover": self.top_turnover,
            "bottom_turnover": self.bottom_turnover,
            "monotonicity": self.monotonicity,
            "dropped_rows": self.dropped_rows,
            "quantile_mean_returns": {str(key): value for key, value in self.quantile_mean_returns.items()},
            "ic_recent": {
                str(pd.Timestamp(str(key)).date()): float(value) for key, value in self.ic_series.tail(60).items()
            },
        }

    def summary(self) -> str:
        if not self.sufficient:
            return f"[{self.factor}] 有效截面不足，无法评价（丢弃 {self.dropped_rows} 行）"
        return (
            f"[{self.factor}] IC {self.ic_mean:+.4f}，ICIR {self.icir:+.2f}，"
            f"IC>0 占比 {self.ic_positive_ratio:.1%}，多空均值 {self.long_short_mean_return:+.3%}，"
            f"顶层换手 {self.top_turnover:.1%}，单调性 {self.monotonicity:+.2f}（{self.observations} 个截面）"
        )


# ============================================================ 未来收益
def forward_returns(prices: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """逐标的计算未来 ``horizon`` 日收益（``close.shift(-h)/close - 1``）。

    末尾 ``horizon`` 个交易日**必然为 NaN**（没有未来价格），调用方会在合并后被剔除——
    这正是"评价不能用不存在的数据"的体现。
    """
    if horizon <= 0:
        raise ValidationError("horizon 必须为正", detail={"horizon": horizon})
    missing = [column for column in ("code", "date", "close") if column not in prices.columns]
    if missing:
        raise ValidationError("未来收益计算缺少必要列", detail={"missing": missing})

    parts: list[pd.DataFrame] = []
    for code, group in prices.groupby("code", sort=True):
        ordered = group.sort_values("date")
        close = pd.to_numeric(ordered["close"], errors="coerce")
        parts.append(
            pd.DataFrame(
                {
                    "date": ordered["date"].to_numpy(),
                    "code": code,
                    "forward_return": (close.shift(-horizon) / close - 1).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


# ============================================================ IC
def information_coefficient(
    merged: pd.DataFrame,
    *,
    factor_column: str,
    method: str = "spearman",
    min_cross_section: int = 5,
) -> pd.Series:
    """逐截面计算 IC，返回 ``date → IC`` 的序列（样本不足的截面直接跳过）。"""
    values: dict[Any, float] = {}
    for day, group in merged.groupby("date", sort=True):
        sample = group[[factor_column, "forward_return"]].dropna()
        if len(sample) < min_cross_section:
            continue
        if sample[factor_column].nunique() < 2 or sample["forward_return"].nunique() < 2:
            continue  # 常数截面无法计算相关性
        correlation = (
            spearman_correlation(sample[factor_column], sample["forward_return"])
            if method == "spearman"
            else float(sample[factor_column].corr(sample["forward_return"]))
        )
        if pd.notna(correlation):
            values[day] = float(correlation)
    series = pd.Series(values, dtype=float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def icir(ic_series: pd.Series) -> float:
    """IC 信息比率。IC 波动为零时返回 0（无波动的"稳定"没有意义）。"""
    if len(ic_series) < 2:
        return 0.0
    std = float(ic_series.std(ddof=1))
    if std <= _ZERO_TOLERANCE:
        return 0.0
    return float(ic_series.mean() / std)


# ============================================================ 分层
def assign_quantiles(values: pd.Series, quantiles: int) -> pd.Series:
    """按因子值分层（1 = 最低层）。用 rank 后再 qcut，避免并列值被分到不同层。"""
    ranks = values.rank(method="first")
    labels = pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
    return labels.astype(int)


def quantile_analysis(
    merged: pd.DataFrame,
    *,
    factor_column: str,
    quantiles: int = 5,
    min_cross_section: int = 5,
) -> tuple[dict[int, float], pd.DataFrame]:
    """分层收益与各层净值。返回 ``(各层平均未来收益, 各层累计净值表)``。"""
    records: list[pd.DataFrame] = []
    for _, group in merged.groupby("date", sort=True):
        sample = group[[factor_column, "forward_return"]].dropna()
        if len(sample) < max(min_cross_section, quantiles):
            continue
        frame = sample.copy()
        frame["quantile"] = assign_quantiles(sample[factor_column], quantiles)
        frame["date"] = group["date"].iloc[0]
        records.append(frame[["date", "quantile", "forward_return"]])

    if not records:
        return {}, pd.DataFrame()

    stacked = pd.concat(records, ignore_index=True)
    # groupby 的键被 pandas-stubs 标注为 Hashable，故显式转成 int（层号本就是 1..Q 的整数）
    mean_returns = {
        int(str(quantile)): float(value["forward_return"].mean())
        for quantile, value in stacked.groupby("quantile", sort=True)
    }
    pivot = stacked.pivot_table(index="date", columns="quantile", values="forward_return", aggfunc="mean")
    cumulative = (1.0 + pivot.fillna(0.0)).cumprod()
    return mean_returns, cumulative


def quantile_turnover(
    merged: pd.DataFrame,
    *,
    factor_column: str,
    quantiles: int = 5,
    min_cross_section: int = 5,
) -> dict[int, float]:
    """各层成员变动的平均换手率：``0.5 × Σ|w_t - w_{t-1}|`` 在多日上的均值。

    高换手意味着成本会吃掉相当一部分 IC——所以换手必须与 IC 同时看。
    """
    holdings: dict[Any, dict[str, set[Any]]] = {}
    for day, group in merged.groupby("date", sort=True):
        sample = group[["code", factor_column, "forward_return"]].dropna()
        if len(sample) < max(min_cross_section, quantiles):
            continue
        labels = assign_quantiles(sample[factor_column], quantiles)
        holdings[day] = {
            str(quantile): set(sample.loc[labels == quantile, "code"].tolist())
            for quantile in range(1, quantiles + 1)
        }

    ordered_days = sorted(holdings)
    if len(ordered_days) < 2:
        return {}

    turnover: dict[int, float] = {}
    for quantile in range(1, quantiles + 1):
        values: list[float] = []
        for previous_day, day in itertools.pairwise(ordered_days):
            previous = holdings[previous_day][str(quantile)]
            current = holdings[day][str(quantile)]
            if not previous or not current:
                continue
            weights_previous = {code: 1.0 / len(previous) for code in previous}
            weights_current = {code: 1.0 / len(current) for code in current}
            delta = sum(
                abs(weights_current.get(code, 0.0) - weights_previous.get(code, 0.0))
                for code in set(weights_previous) | set(weights_current)
            )
            values.append(delta / 2.0)
        turnover[quantile] = float(np.mean(values)) if values else 0.0
    return turnover


def factor_correlation(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    min_cross_section: int = 5,
) -> pd.DataFrame:
    """因子间的**平均截面秩相关**矩阵。

    为什么必须在合成多因子前看这个：两个 IC 都不错的因子若相关性 0.9，
    合起来几乎不会带来增量信息，只会把同一个风险敞口放大一倍。
    逐日算截面秩相关再取均值，比"全样本混算"更贴近实际使用方式
    （每日重新排序选股，而不是用跨时间的绝对水平）。
    """
    names = list(columns)
    if len(names) < 2:
        raise ValidationError("相关性矩阵至少需要两个因子", detail={"columns": names})
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValidationError("因子列不存在", detail={"missing": missing})

    totals = pd.DataFrame(0.0, index=names, columns=names)
    sections = 0
    for _, group in frame.groupby("date", sort=True):
        sample = group[names].dropna()
        if len(sample) < min_cross_section:
            continue
        # 排名后的皮尔逊相关即 Spearman（同样避免 scipy 依赖）
        totals += sample.rank().corr().fillna(0.0)
        sections += 1
    if sections == 0:
        return pd.DataFrame(index=names, columns=names, dtype=float)
    return (totals / sections).astype(float)


def monotonicity_score(quantile_mean_returns: Mapping[int, float]) -> float:
    """分层单调性：各层收益序与层号的 Spearman 相关（+1 = 完全单调递增）。

    比"多空收益高"更能说明因子在**给标的排序**而不是靠某一层运气。
    """
    if len(quantile_mean_returns) < 2:
        return 0.0
    layers = pd.Series(sorted(quantile_mean_returns))
    returns = pd.Series([quantile_mean_returns[int(layer)] for layer in layers])
    if layers.nunique() < 2 or returns.nunique() < 2:
        return 0.0
    score = spearman_correlation(layers, returns)
    return 0.0 if pd.isna(score) else float(score)


# ============================================================ 主入口
def evaluate_factor(
    prices: pd.DataFrame,
    factor_values: pd.DataFrame,
    *,
    factor: str,
    config: EvaluationConfig | None = None,
) -> FactorEvaluation:
    """评价单个因子：IC / ICIR / 分层 / 多空 / 换手 / 单调性。

    ``prices`` 为行情长表（``code``/``date``/``close``），
    ``factor_values`` 为因子长表（``date``/``code``/``<factor>`` 列）。
    """
    resolved = config or EvaluationConfig()
    if factor not in factor_values.columns:
        raise ValidationError("因子列不存在", detail={"factor": factor, "columns": list(factor_values.columns)})

    forward = forward_returns(prices, horizon=resolved.horizon)
    merged = factor_values[["date", "code", factor]].merge(forward, on=["date", "code"], how="inner")
    if merged.empty:
        raise ValidationError("因子值与行情无法对齐（检查 date/code 口径）")

    before = len(merged)
    merged = merged.dropna(subset=[factor, "forward_return"]).reset_index(drop=True)
    dropped = before - len(merged)
    if merged.empty:
        return FactorEvaluation(
            factor=factor,
            horizon=resolved.horizon,
            quantiles=resolved.quantiles,
            observations=0,
            ic_mean=0.0,
            ic_std=0.0,
            icir=0.0,
            ic_positive_ratio=0.0,
            ic_series=pd.Series(dtype=float),
            dropped_rows=dropped,
        )

    ic_series = information_coefficient(
        merged,
        factor_column=factor,
        method=resolved.ic_method,
        min_cross_section=resolved.min_cross_section,
    )
    quantile_means, quantile_cum = quantile_analysis(
        merged, factor_column=factor, quantiles=resolved.quantiles, min_cross_section=resolved.min_cross_section
    )
    turnovers = quantile_turnover(
        merged, factor_column=factor, quantiles=resolved.quantiles, min_cross_section=resolved.min_cross_section
    )

    long_short_nav, long_short_mean = _long_short(quantile_cum, quantiles=resolved.quantiles)

    return FactorEvaluation(
        factor=factor,
        horizon=resolved.horizon,
        quantiles=resolved.quantiles,
        observations=len(ic_series),
        ic_mean=float(ic_series.mean()) if len(ic_series) else 0.0,
        ic_std=float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0,
        icir=icir(ic_series),
        ic_positive_ratio=float((ic_series > 0).mean()) if len(ic_series) else 0.0,
        ic_series=ic_series,
        quantile_mean_returns=quantile_means,
        quantile_cum_returns=quantile_cum,
        long_short_mean_return=long_short_mean,
        long_short_nav=long_short_nav,
        top_turnover=turnovers.get(resolved.quantiles, 0.0),
        bottom_turnover=turnovers.get(1, 0.0),
        monotonicity=monotonicity_score(quantile_means),
        dropped_rows=dropped,
    )


def _long_short(quantile_cum: pd.DataFrame, *, quantiles: int) -> tuple[pd.Series, float]:
    """多空组合（顶层 - 底层）：按日收益差复利得到净值。

    层号由 ``assign_quantiles`` 保证为 1..Q，故顶层/底层直接取自配置，
    不去猜列标签（顺带避开 pandas 列标签的联合类型）。

    首日没有前值因而无收益（记为 0），故多空统计从第二个截面开始——这一点在
    ``long_short_mean_return`` 的语义里是明确的，而不是把首日当成 0 收益混进均值。
    """
    if quantile_cum.empty or quantiles < 2:
        return pd.Series(dtype=float), 0.0
    top, bottom = quantiles, 1
    if top not in quantile_cum.columns or bottom not in quantile_cum.columns:
        return pd.Series(dtype=float), 0.0
    returns = quantile_cum.pct_change().fillna(0.0)
    spread = returns[top] - returns[bottom]
    nav = (1.0 + spread).cumprod()
    mean_return = float(spread.iloc[1:].mean()) if len(spread) > 1 else 0.0
    return nav, mean_return


__all__ = [
    "SUPPORTED_IC_METHODS",
    "EvaluationConfig",
    "FactorEvaluation",
    "assign_quantiles",
    "evaluate_factor",
    "factor_correlation",
    "forward_returns",
    "icir",
    "information_coefficient",
    "monotonicity_score",
    "quantile_analysis",
    "quantile_turnover",
    "spearman_correlation",
]
