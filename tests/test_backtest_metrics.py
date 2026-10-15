"""P1-B2 绩效指标测试：公式全部用手工可算的净值序列验证。

指标算错不会报错，只会让"看起来很好的策略"通过评审——
所以这里宁可多写几条手算用例，也不依赖"跑起来没报错"。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.backtest.metrics import (
    compute_metrics,
    drawdown_series,
    monthly_returns,
    validate_nav,
)
from engine.errors import ValidationError


def _nav(values: list[float], *, start: date = date(2026, 10, 12)) -> pd.Series:
    """构造逐日递增的净值序列（每个值间隔一个自然日，足够指标计算）。"""
    index = pd.DatetimeIndex([pd.Timestamp(start) + pd.Timedelta(days=offset) for offset in range(len(values))])
    return pd.Series(values, index=index, dtype=float)


# ============================================================ 输入校验
def test_空序列被拒() -> None:
    with pytest.raises(ValidationError):
        validate_nav(pd.Series(dtype=float))


def test_非正净值被拒() -> None:
    with pytest.raises(ValidationError):
        validate_nav(_nav([1.0, 0.5, 0.0]))


def test_含NaN被拒() -> None:
    series = _nav([1.0, float("nan"), 1.1])
    with pytest.raises(ValidationError):
        validate_nav(series)


def test_乱序输入自动按日期排序() -> None:
    series = _nav([1.0, 1.1])
    shuffled = series.iloc[[1, 0]]
    assert list(validate_nav(shuffled)) == [1.0, 1.1]


# ============================================================ 基础指标
def test_总收益与年化() -> None:
    metrics = compute_metrics(_nav([1.0, 1.1, 0.99, 1.2]), risk_free_rate=0.0)

    assert metrics.total_return == pytest.approx(0.20)
    # 年化 = 1.2^(252/3) - 1
    assert metrics.annual_return == pytest.approx(1.2 ** (252 / 3) - 1)


def test_本金归零时年化下限为负100() -> None:
    metrics = compute_metrics(_nav([1.0, 0.5, 0.05]))
    assert metrics.total_return == pytest.approx(-0.95)
    assert metrics.annual_return == -1.0


def test_单点序列年化为零() -> None:
    metrics = compute_metrics(_nav([1.0]))
    assert metrics.total_return == 0.0
    assert metrics.annual_return == 0.0
    assert metrics.annual_volatility == 0.0


def test_常数净值的风险与收益全为零() -> None:
    metrics = compute_metrics(_nav([1.0, 1.0, 1.0, 1.0]), risk_free_rate=0.0)

    assert metrics.max_drawdown == 0.0
    assert metrics.annual_volatility == 0.0
    assert metrics.sharpe_ratio == 0.0, "波动为零时夏普定义为 0，而不是除零"
    assert metrics.calmar_ratio == 0.0


# ============================================================ 回撤
def test_最大回撤与起止日期() -> None:
    nav = _nav([1.0, 1.1, 0.99, 1.05, 1.2])
    metrics = compute_metrics(nav)

    assert metrics.max_drawdown == pytest.approx(-(1.0 - 0.99 / 1.1)), "回撤以负数表示"
    assert metrics.max_drawdown_start == date(2026, 10, 13)  # 峰值 1.1
    assert metrics.max_drawdown_end == date(2026, 10, 14)  # 谷值 0.99


def test_回撤序列上限为零() -> None:
    drawdown = drawdown_series(_nav([1.0, 1.1, 0.99, 1.2]))
    assert (drawdown <= 0).all()
    assert drawdown.min() == pytest.approx(-(1.0 - 0.99 / 1.1))


def test_无回撤时起止日期为空() -> None:
    metrics = compute_metrics(_nav([1.0, 1.01, 1.02]))
    assert metrics.max_drawdown == 0.0
    assert metrics.max_drawdown_start is None
    assert metrics.max_drawdown_end is None


# ============================================================ 风险调整收益
def test_夏普比率手工验证() -> None:
    nav = _nav([1.0, 1.02, 1.04])  # 日收益 2%、1.96%
    returns = np.array([0.02, 0.02 / 1.02])
    expected = (returns.mean() - 0.02 / 252) / returns.std(ddof=1) * np.sqrt(252)

    metrics = compute_metrics(nav, risk_free_rate=0.02)
    assert metrics.sharpe_ratio == pytest.approx(expected)


def test_卡玛比率() -> None:
    metrics = compute_metrics(_nav([1.0, 1.1, 0.99, 1.2]), risk_free_rate=0.0)
    expected = metrics.annual_return / abs(metrics.max_drawdown)
    assert metrics.calmar_ratio == pytest.approx(expected)


def test_索提诺只用下行波动() -> None:
    nav = _nav([1.0, 1.02, 1.00, 1.04])
    returns = np.array([0.02, -0.02 / 1.02, 0.04])
    downside = np.minimum(returns - 0.02 / 252, 0.0)
    expected = (returns - 0.02 / 252).mean() / np.sqrt((downside**2).mean()) * np.sqrt(252)

    metrics = compute_metrics(nav, risk_free_rate=0.02)
    assert metrics.sortino_ratio == pytest.approx(expected)
    assert metrics.sortino_ratio > metrics.sharpe_ratio, "下行波动小于总波动时索提诺应更大"


# ============================================================ 胜率与盈亏比
def test_日胜率() -> None:
    metrics = compute_metrics(_nav([1.0, 1.1, 0.99, 1.2]))
    assert metrics.daily_win_rate == pytest.approx(2 / 3)


def test_盈亏比单边行情时显式缺失() -> None:
    assert compute_metrics(_nav([1.0, 1.1, 1.2])).profit_loss_ratio is None, "没有亏损日时该指标无意义"
    assert compute_metrics(_nav([1.0, 0.9, 0.8])).profit_loss_ratio is None


def test_盈亏比手工验证() -> None:
    metrics = compute_metrics(_nav([1.0, 1.10, 0.99, 1.20]))
    wins = np.array([0.10, 1.20 / 0.99 - 1.0])
    losses = np.array([0.99 / 1.10 - 1.0])
    expected = abs(wins.mean() / losses.mean())
    assert metrics.profit_loss_ratio == pytest.approx(expected)


# ============================================================ 月度收益
def test_月度收益跨月复利() -> None:
    values = [1.0, 1.1, 1.1, 1.21]
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-09-30"),
            pd.Timestamp("2026-09-30"),
            pd.Timestamp("2026-10-08"),
            pd.Timestamp("2026-10-09"),
        ]
    )
    nav = pd.Series(values, index=index)

    monthly = monthly_returns(nav)
    assert monthly["2026-09"] == pytest.approx(0.10)
    assert monthly["2026-10"] == pytest.approx(0.10)
    assert compute_metrics(nav).monthly_returns == monthly


# ============================================================ 基准对照
def test_基准指标_完全跟踪时超额为零() -> None:
    nav = _nav([1.0, 1.1, 0.99, 1.2])
    metrics = compute_metrics(nav, benchmark=nav.copy())

    assert metrics.benchmark_beta == pytest.approx(1.0)
    assert metrics.benchmark_alpha == pytest.approx(0.0, abs=1e-9)
    assert metrics.information_ratio == 0.0
    assert metrics.tracking_error == 0.0


def test_基准指标_beta与信息比率() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2026-10-12") + pd.Timedelta(days=i) for i in range(60)])
    rng = np.random.default_rng(42)
    benchmark_returns = pd.Series(rng.normal(0.0005, 0.01, 59), index=dates[1:])
    strategy_returns = benchmark_returns * 1.5  # beta = 1.5
    nav = pd.concat([pd.Series([1.0], index=dates[:1]), (1 + strategy_returns).cumprod()])
    benchmark = pd.concat([pd.Series([1.0], index=dates[:1]), (1 + benchmark_returns).cumprod()])

    metrics = compute_metrics(nav, benchmark=benchmark)

    assert metrics.benchmark_beta == pytest.approx(1.5, abs=0.05)
    # 杠杆放大会产生主动收益：active = 0.5 × 基准收益，TE 与 IR 必须与之吻合
    active = strategy_returns - benchmark_returns
    assert metrics.tracking_error == pytest.approx(
        float(active.std(ddof=1) * np.sqrt(252)), rel=1e-6
    )
    assert metrics.information_ratio == pytest.approx(
        float(active.mean() / active.std(ddof=1) * np.sqrt(252)), rel=1e-6
    )


def test_无基准时基准指标为None() -> None:
    metrics = compute_metrics(_nav([1.0, 1.1, 1.2]))
    assert metrics.benchmark_alpha is None
    assert metrics.benchmark_beta is None
    assert metrics.information_ratio is None
    assert metrics.tracking_error is None


# ============================================================ 序列化
def test_指标可JSON序列化且日期为ISO格式() -> None:
    metrics = compute_metrics(_nav([1.0, 1.1, 0.99, 1.2]))
    payload = metrics.to_jsonable()

    assert payload["max_drawdown_start"] == "2026-10-13"
    assert payload["max_drawdown_end"] == "2026-10-14"
    assert isinstance(payload["monthly_returns"], dict)
