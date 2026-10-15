"""策略上下文（P1-B4 核心）。

``BacktestContext`` 实现 ``engine.ports.Context``：**只读、无下单方法、历史严格受时点约束**。

防未来函数的三道防线（全部有测试）：
1. ``bars()`` 只给当日行情——策略想看明天，接口上就不存在；
2. ``history()`` 强制以 ``now`` 为界，越界数据不可达；
3. 回测与实盘共用同一份策略代码：实盘侧换成实时 ``Context`` 实现即可，策略零改动。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from engine.backtest.data import DataFrameBarProvider
from engine.ports import AdjType
from engine.types import Account, Bar, Position

_MARKET_CLOSE = 15


@dataclass(frozen=True, slots=True)
class BacktestContext:
    """单日策略上下文（每个交易日构造一个新实例）。

    注意字段名 ``account_state`` / ``position_map`` 刻意与端口方法
    ``account()`` / ``positions()`` 错开——端口方法是策略唯一可见的访问入口，
    数据字段不能遮蔽它们（frozen+slots 的 dataclass 没有类级方法分发保护）。
    """

    provider: DataFrameBarProvider
    strategy_id: str
    trade_date_value: date
    params: Mapping[str, Any] = field(default_factory=dict)
    bars_today: Mapping[str, Bar] = field(default_factory=dict)
    account_state: Account = field(default_factory=lambda: Account())
    position_map: Mapping[str, Position] = field(default_factory=dict)

    # -------------------------------------------------- 端口属性
    @property
    def now(self) -> datetime:
        """当前时点：当日收盘（策略在收盘后决策）。"""
        return datetime(self.trade_date_value.year, self.trade_date_value.month, self.trade_date_value.day, _MARKET_CLOSE)

    @property
    def trade_date(self) -> date:
        return self.trade_date_value

    # -------------------------------------------------- 数据访问（全部只读、受时点约束）
    def bars(self, codes: Sequence[str] | None = None) -> Mapping[str, Bar]:
        if codes is None:
            return dict(self.bars_today)
        wanted = set(codes)
        return {code: bar for code, bar in self.bars_today.items() if code in wanted}

    def history(
        self,
        code: str,
        *,
        field: str = "close",
        window: int = 1,
        adj: AdjType = "post",  # noqa: ARG002 —— 保持端口签名稳定；存储层统一后复权，见 docstring
    ) -> list[Decimal]:
        """截至当日（含）的历史值，升序。

        注意 ``adj`` 仅作口径提示：存储层统一为后复权，前复权由展示层处理，
        这里不做换算（与 ``engine.datahub`` 的口径约定一致）。
        """
        return self.provider.history(code, field=field, end=self.trade_date_value, window=window)

    def factor(self, name: str, code: str) -> float | None:  # noqa: ARG002 —— 端口签名；P2 接入因子层后启用
        """因子值（P1 未接因子层，返回 None；P2 接入后按 calc_date <= now 取用）。"""
        return None

    def universe(self) -> tuple[str, ...]:
        return tuple(sorted(self.bars_today))

    # -------------------------------------------------- 账户视图
    def positions(self) -> Mapping[str, Position]:
        return dict(self.position_map)

    def position(self, code: str) -> Position | None:
        return self.position_map.get(code)

    def account(self) -> Account:
        return self.account_state


__all__ = ["BacktestContext"]
