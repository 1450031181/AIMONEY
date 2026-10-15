"""回测内核：撮合、成本、指标与主循环。"""

from __future__ import annotations

from engine.backtest.context import BacktestContext
from engine.backtest.data import DataFrameBarProvider
from engine.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from engine.backtest.match import FillAttempt, FillStatus, MatchEngine, MatcherConfig
from engine.backtest.metrics import (
    PerformanceMetrics,
    compute_metrics,
    daily_returns,
    drawdown_series,
    monthly_returns,
)
from engine.backtest.planner import PlannerConfig, TargetWeightPlanner
from engine.backtest.slippage import FixedBpsSlippage, NoSlippage

__all__ = [
    "BacktestConfig",
    "BacktestContext",
    "BacktestEngine",
    "BacktestResult",
    "DataFrameBarProvider",
    "FillAttempt",
    "FillStatus",
    "FixedBpsSlippage",
    "MatchEngine",
    "MatcherConfig",
    "NoSlippage",
    "PerformanceMetrics",
    "PlannerConfig",
    "TargetWeightPlanner",
    "compute_metrics",
    "daily_returns",
    "drawdown_series",
    "monthly_returns",
]
