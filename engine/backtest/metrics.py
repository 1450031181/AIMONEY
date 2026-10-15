"""绩效指标（P1-B2）——回测报告的输出层，全部纯函数。

三条工程约定：

1. **公式集中且可手算验证**：年化用 `(1+总收益)^(252/n) - 1`（n 为收益观测数），
   夏普用日频计算后年化，与需求文档附录 A 一致；每条公式都有手工计算的用例。
2. **输入校验 fail-fast**：净值必须非空且全为正——0 或负净值意味着上游结算已经错了，
   在这里静默算出"负回撤 100%"之类的指标只会掩盖上游 bug。
3. **基准相关指标可选**：没有基准时 alpha/beta/信息比率/跟踪误差为 None，
   而不是填 0 冒充"无超额"。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from engine.errors import ValidationError

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """回测绩效指标集。基准相关字段在未提供基准时为 None。"""

    total_return: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    max_drawdown_start: date | None = None
    max_drawdown_end: date | None = None
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    daily_win_rate: float = 0.0
    profit_loss_ratio: float | None = None
    monthly_returns: Mapping[str, float] = field(default_factory=dict)
    benchmark_alpha: float | None = None
    benchmark_beta: float | None = None
    information_ratio: float | None = None
    tracking_error: float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_start": self.max_drawdown_start.isoformat() if self.max_drawdown_start else None,
            "max_drawdown_end": self.max_drawdown_end.isoformat() if self.max_drawdown_end else None,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "daily_win_rate": self.daily_win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "monthly_returns": dict(self.monthly_returns),
            "benchmark_alpha": self.benchmark_alpha,
            "benchmark_beta": self.benchmark_beta,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error,
        }


# ============================================================ 中间量
def validate_nav(nav: pd.Series) -> pd.Series:
    """校验并规范净值序列（float、按日期升序）。"""
    if nav is None or len(nav) == 0:
        raise ValidationError("净值序列为空，无法计算绩效指标")
    values = pd.to_numeric(nav, errors="coerce")
    if values.isna().any():
        raise ValidationError("净值序列存在非数值", detail={"bad_rows": int(values.isna().sum())})
    if (values <= 0).any():
        raise ValidationError(
            "净值必须全为正：出现非正净值说明上游结算有 bug，不能继续计算指标",
            detail={"min": float(values.min())},
        )
    result = values.astype(float)
    result.index = pd.to_datetime(pd.Series(result.index))
    return result.sort_index()


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def drawdown_series(nav: pd.Series) -> pd.Series:
    """回撤序列：``nav / 历史最高 - 1``（≤ 0）。"""
    return nav / nav.cummax() - 1.0


def monthly_returns(nav: pd.Series) -> dict[str, float]:
    """按自然月聚合收益（复利）。键形如 ``2026-10``。"""
    returns = daily_returns(nav)
    if returns.empty:
        return {}
    periods = pd.DatetimeIndex(returns.index).to_period("M")
    monthly: dict[str, float] = {}
    for period, values in returns.groupby(np.asarray(periods)):
        compounded = float(np.prod([1.0 + float(value) for value in values]) - 1.0)
        monthly[str(period)] = round(compounded, 6)
    return monthly


# ============================================================ 主计算
def compute_metrics(
    nav: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceMetrics:
    """由净值序列计算全部绩效指标。

    ``nav`` 索引需为日期且升序（不强制连续交易日——指标只依赖观测数）。
    """
    clean = validate_nav(nav)
    returns = daily_returns(clean)
    observations = len(returns)

    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    annual_return = _annualize(total_return, observations, periods_per_year)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if observations >= 2 else 0.0

    drawdown = drawdown_series(clean)
    max_drawdown = float(drawdown.min())
    drawdown_values = np.asarray(drawdown, dtype=float)
    nav_values = np.asarray(clean, dtype=float)
    # 用模块级 np.argmin/argmax：ndarray.argmin() 的无参形式在 numpy-stubs 中签名不完整
    dd_end_position = int(np.argmin(drawdown_values)) if max_drawdown < 0 else None
    dd_start: date | None = None
    dd_end: date | None = None
    if dd_end_position is not None:
        peak_position = int(np.argmax(nav_values[: dd_end_position + 1]))
        dd_start = _as_date(clean.index[peak_position])
        dd_end = _as_date(clean.index[dd_end_position])

    daily_rf = risk_free_rate / periods_per_year
    if observations == 0:
        # 单点净值：无收益观测，风险类指标全部无定义，显式置零而不是让 numpy 对空序列求均值
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=0.0,
            annual_volatility=0.0,
            max_drawdown=max_drawdown,
            max_drawdown_start=dd_start,
            max_drawdown_end=dd_end,
            monthly_returns=monthly_returns(clean),
        )

    active_daily = returns - daily_rf
    sharpe = _ratio(float(active_daily.mean()), float(returns.std(ddof=1)), periods_per_year)
    downside = np.minimum(active_daily.to_numpy(dtype=float), 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2)))
    sortino = _ratio(float(active_daily.mean()), downside_std, periods_per_year)
    calmar = _simple_ratio(annual_return, abs(max_drawdown))
    win_rate = float((returns > 0).sum() / observations) if observations else 0.0
    profit_loss_ratio = _profit_loss_ratio(returns)

    alpha = beta = info_ratio = tracking_error = None
    if benchmark is not None:
        clean_benchmark = validate_nav(benchmark)
        aligned_nav, aligned_benchmark = clean.align(clean_benchmark, join="inner")
        if len(aligned_nav) >= 3:
            strategy_returns = daily_returns(aligned_nav)
            benchmark_returns = daily_returns(aligned_benchmark)
            beta = float(_covariance(strategy_returns, benchmark_returns) / _variance(benchmark_returns))
            alpha = float(
                (strategy_returns.mean() - beta * benchmark_returns.mean()) * periods_per_year
            )
            active = strategy_returns - benchmark_returns
            tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
            info_ratio = _ratio(float(active.mean()), float(active.std(ddof=1)), periods_per_year)

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        max_drawdown=max_drawdown,
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        daily_win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        monthly_returns=monthly_returns(clean),
        benchmark_alpha=alpha,
        benchmark_beta=beta,
        information_ratio=info_ratio,
        tracking_error=tracking_error,
    )


# ============================================================ 内部工具
def _annualize(total_return: float, observations: int, periods_per_year: int) -> float:
    if observations <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0  # 本金归零，不再外推
    return float((1.0 + total_return) ** (periods_per_year / observations) - 1.0)


def _ratio(mean_active: float, std: float, periods_per_year: int) -> float:
    """日频均值/标准差 → 年化比率。波动为零时返回 0（没有风险就没有超额的口径）。"""
    if std == 0:
        return 0.0
    return float(mean_active / std * np.sqrt(periods_per_year))


def _simple_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _profit_loss_ratio(returns: pd.Series) -> float | None:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if wins.empty or losses.empty:
        return None  # 单边行情下该指标无意义，显式缺失而不是填 0
    return float(abs(wins.mean() / losses.mean()))


def _covariance(left: pd.Series, right: pd.Series) -> float:
    return float(np.cov(left.to_numpy(dtype=float), right.to_numpy(dtype=float), ddof=1)[0, 1])


def _variance(series: pd.Series) -> float:
    return float(np.var(series.to_numpy(dtype=float), ddof=1))


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "PerformanceMetrics",
    "compute_metrics",
    "daily_returns",
    "drawdown_series",
    "monthly_returns",
    "validate_nav",
]
