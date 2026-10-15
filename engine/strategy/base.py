"""策略基类（P1-B5）。

实现 ``engine.ports.Strategy`` 协议的公共部分，把样板代码收敛到这里：
- 参数管理（默认值 + 外部覆盖）；
- 历史数据访问（委托 Context，天然受防未来函数约束）；
- ``init`` / ``on_order_filled`` 的默认空实现。

模板策略只写 ``on_bar`` 的信号逻辑。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from engine.errors import StrategyError
from engine.ports import Context
from engine.types import Trade

_logger = logging.getLogger(__name__)


class BaseStrategy:
    """所有内置与用户策略的基类（结构化满足 Strategy 协议，无需继承引擎侧任何东西）。"""

    def __init__(self, *, strategy_id: str, params: Mapping[str, Any] | None = None) -> None:
        self.strategy_id = strategy_id
        self.params: dict[str, Any] = dict(params or {})

    # -------------------------------------------------- 协议方法
    def init(self, context: Context) -> None:  # noqa: ARG002 —— 协议签名；子类按需使用
        """首次运行前调用一次（预加载数据等）。默认无操作。"""
        return

    def on_order_filled(self, context: Context, trade: Trade) -> None:  # noqa: ARG002 —— 协议签名
        """成交回报（日内止损等场景用）。默认无操作。"""
        return

    # -------------------------------------------------- 参数与工具
    def param(self, key: str) -> Any:
        """读取必需参数，缺失即报错（必需参数不该有静默默认值）。"""
        if key not in self.params:
            raise StrategyError(
                f"缺少策略参数：{key}",
                detail={"strategy_id": self.strategy_id, "required": key, "given": sorted(self.params)},
            )
        return self.params[key]

    def param_or(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    @staticmethod
    def mean(values: Sequence[Decimal | float]) -> float:
        if not values:
            raise StrategyError("均值计算需要非空序列")
        return float(sum(float(value) for value in values) / len(values))

    def closes(self, context: Context, code: str, window: int) -> list[Decimal]:
        """截至当日的收盘价序列（升序，最多 window 个）。"""
        return context.history(code, field="close", window=window)


__all__ = ["BaseStrategy"]
