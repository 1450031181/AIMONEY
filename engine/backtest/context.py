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
    #: 当日可见的因子值 ``code → {factor: value}``（由引擎按日切片注入，
    #: 策略无法访问其他日期——越界在接口上不存在，而不是靠约定）
    factor_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    #: 回测是否**装载了**因子快照。用于区分两种完全不同的情况：
    #: "因子根本没接入"（配置错误，应当报错）vs "今日无值"（预热期/停牌/缺数据，属于正常）
    factors_configured: bool = False

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

    def factor(self, name: str, code: str) -> float | None:
        """**当日**因子值（截面标准化后）。当日无该标的或该因子缺失时返回 None。

        返回 None 而不是抛错，是刻意的：真实场景里"某标的某天没有因子值"很常见
        （历史不足、次新股、数据缺口），策略必须显式处理它；
        若抛错，策略作者就会写 try/except 把缺口悄悄吞掉。
        """
        return self.factor_values.get(code, {}).get(name)

    def factor_names(self) -> tuple[str, ...]:
        """**当日有值**的因子名（预热期可能为空，这属于正常）。"""
        names: set[str] = set()
        for values in self.factor_values.values():
            names.update(values)
        return tuple(sorted(names))

    def has_factors(self) -> bool:
        """回测是否装载了因子快照（与"今日是否有值"无关）。"""
        return self.factors_configured

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
