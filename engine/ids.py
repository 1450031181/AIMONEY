"""ID 生成：所有链路 ID 便于人工肉眼排序与追溯。

格式：``<prefix>-<yyyymmddHHMMSS>-<8位随机>``，例如 ``ord-20261014093500-a1b2c3d4``。
可读前缀让日志里一眼看出这是信号、订单还是成交。
"""

from __future__ import annotations

import uuid
from datetime import datetime

# 链路 ID 前缀常量，禁止散落硬编码
PREFIX_SIGNAL = "sig"
PREFIX_PLAN = "plan"
PREFIX_ORDER = "ord"
PREFIX_TRADE = "trd"
PREFIX_TASK = "task"
PREFIX_BACKTEST = "bt"


def new_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def new_trace_id() -> str:
    """链路追踪 ID（日志用）。"""
    return uuid.uuid4().hex[:16]


def new_signal_id() -> str:
    return new_id(PREFIX_SIGNAL)


def new_plan_id() -> str:
    return new_id(PREFIX_PLAN)


def new_order_id() -> str:
    return new_id(PREFIX_ORDER)


def new_trade_id() -> str:
    return new_id(PREFIX_TRADE)


def new_task_id() -> str:
    return new_id(PREFIX_TASK)


def new_backtest_id() -> str:
    return new_id(PREFIX_BACKTEST)


__all__ = [
    "PREFIX_BACKTEST",
    "PREFIX_ORDER",
    "PREFIX_PLAN",
    "PREFIX_SIGNAL",
    "PREFIX_TASK",
    "PREFIX_TRADE",
    "new_backtest_id",
    "new_id",
    "new_order_id",
    "new_plan_id",
    "new_signal_id",
    "new_task_id",
    "new_trace_id",
    "new_trade_id",
]
