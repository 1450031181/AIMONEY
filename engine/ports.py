"""端口定义（T-04）：engine 与外部世界之间的全部契约。

三条铁律：

1. **端口用 ``typing.Protocol`` 定义**，适配器无需继承即可实现（结构化子类型），
   engine 不会被具体实现细节绑架；
2. **``Context`` 刻意不提供任何下单方法** —— 策略拿不到券商句柄、也无法自己发单，
   只能产出 ``Signal``，由引擎统一走「组合 → 风控 → OMS」。
   这是设计文档 D6「风控不可绕过」在类型层面的落地（并有测试守住）；
3. 端口只使用 ``engine.types`` 领域模型 + ``pandas.DataFrame``（批量行情），
   不出现任何具体数据源/券商 SDK 的类型。

关于复权口径：``daily(adj=...)`` 的 ``post`` 为回测计算的内部口径；
真实下单价必须由 ``engine.datahub.adjust`` 还原（见设计文档 5.4）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol

from engine.types import (
    Account,
    Bar,
    Order,
    OrderPlan,
    Position,
    RiskDecision,
    RiskLevel,
    Signal,
    Trade,
)

if TYPE_CHECKING:  # 仅类型检查期依赖 pandas，运行时由数据层自行引入
    import pandas as pd

#: 复权口径：none 不复权 / pre 前复权 / post 后复权
AdjType = Literal["none", "pre", "post"]

OrderCallback = Callable[[Order], None]
TradeCallback = Callable[[Trade], None]


# ============================================================ 交易日历
class TradingCalendar(Protocol):
    """交易日历。所有「按交易日推进」的逻辑都必须通过它，禁止用自然日推算。"""

    @property
    def name(self) -> str: ...

    def is_trade_day(self, day: date) -> bool: ...

    def sessions(self) -> Sequence[date]:
        """全部交易日（升序）。"""
        ...

    def between(self, start: date, end: date, *, inclusive: bool = True) -> Sequence[date]:
        """区间内的交易日。"""
        ...

    def prev_trade_day(self, day: date, *, include_self: bool = False) -> date: ...

    def next_trade_day(self, day: date, *, include_self: bool = False) -> date: ...

    def shift(self, day: date, n: int) -> date:
        """在交易日序列上平移 n 个交易日（n 可负），越界即报错而非静默夹取。"""
        ...


# ============================================================ 数据源
class DataSource(Protocol):
    """行情/基础数据源（akshare / tushare / 券商行情 / 内存替身）。"""

    @property
    def name(self) -> str: ...

    def trading_calendar(self) -> TradingCalendar: ...

    def daily(self, codes: Sequence[str], start: date, end: date, *, adj: AdjType = "post") -> pd.DataFrame:
        """日线行情，列规范见 ``engine.datahub.schema.DAILY_COLUMNS``。"""
        ...

    def adjust_factors(self, codes: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """复权因子，列为 ``code / date / adj_factor``。"""
        ...

    def universe(self, index_code: str, on: date) -> Sequence[str]:
        """指定日期的指数成分股（必须按成分生效日生效，不得用最新成分回填历史）。"""
        ...

    def index_members(self, index_code: str) -> Sequence[str]:
        """指数**当前**成分快照（用于以抓取日为生效日登记成分历史）。"""
        ...

    def stock_info(self) -> pd.DataFrame:
        """股票基础信息：``code / name / list_date / delist_date / is_st / board``。"""
        ...


# ============================================================ 券商适配器
class BrokerAdapter(Protocol):
    """券商交易接口适配器。

    ``place_order`` **必须幂等**：同一 ``Order.idem_key`` 重复调用只能产生一笔真实委托。
    """

    @property
    def name(self) -> str: ...

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def query_account(self) -> Account: ...

    def query_positions(self) -> list[Position]: ...

    def query_orders(self) -> list[Order]:
        """全量查单：用于断线后状态校准（本地 UNKNOWN 订单必须以此为准）。"""
        ...

    def place_order(self, order: Order) -> str:
        """报单，返回券商委托号。"""
        ...

    def cancel_order(self, broker_order_id: str) -> bool: ...

    def on_order_update(self, callback: OrderCallback) -> None: ...

    def on_trade_update(self, callback: TradeCallback) -> None: ...


# ============================================================ 策略
class Context(Protocol):
    """策略的**唯一**对外通道：全部只读，且刻意不含任何下单方法。

    策略要交易只能返回 ``Signal``；要查数据只能读这里的数据，
    且所有查询严格受 ``now`` 约束（引擎负责在越界时抛 ``LookaheadError``）。
    """

    @property
    def strategy_id(self) -> str: ...

    @property
    def trade_date(self) -> date: ...

    @property
    def now(self) -> datetime:
        """当前时点。读取任何行情都必须以它为界（防未来函数）。"""
        ...

    @property
    def params(self) -> Mapping[str, Any]:
        """策略配置参数（来自 config.yaml）。"""
        ...

    def universe(self) -> Sequence[str]:
        """当日股票池（已按 config 过滤 ST/停牌/次新）。"""
        ...

    def bars(self, codes: Sequence[str] | None = None) -> Mapping[str, Bar]:
        """当前时点各标的行情。"""
        ...

    def history(
        self,
        code: str,
        *,
        field: str = "close",
        window: int = 1,
        adj: AdjType = "post",
    ) -> list[Decimal]:
        """截至 ``now``（含）的最近 ``window`` 个值，升序返回。"""
        ...

    def factor(self, name: str, code: str) -> float | None:
        """因子值（按 ``calc_date <= now`` 取用）。"""
        ...

    def positions(self) -> Mapping[str, Position]: ...

    def position(self, code: str) -> Position | None: ...

    def account(self) -> Account: ...


class Strategy(Protocol):
    """策略协议。回测与实盘使用同一份实现，仅注入的 Context / 执行器不同。"""

    @property
    def strategy_id(self) -> str: ...

    def init(self, ctx: Context) -> None:
        """初始化：读取参数、预加载数据。"""
        ...

    def on_bar(self, ctx: Context, bars: Mapping[str, Bar]) -> Sequence[Signal]:
        """行情推进时产出信号（返回空序列表示不动）。"""
        ...

    def on_order_filled(self, ctx: Context, trade: Trade) -> None:
        """成交回报（用于日内止损等需要即时反应的场景）。"""
        ...


# ============================================================ 风控
@dataclass(frozen=True, slots=True)
class RiskContext:
    """事前风控上下文：针对单条订单计划的校验入参。"""

    trade_date: date
    plan: OrderPlan
    account: Account
    positions: Mapping[str, Position]
    bars: Mapping[str, Bar] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.NORMAL
    extra: Mapping[str, Any] = field(default_factory=dict)

    def bar_of(self, code: str) -> Bar | None:
        return self.bars.get(code)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """事中风控上下文：组合级指标（不针对单条计划），由风控引擎定时构造。"""

    trade_date: date
    account: Account
    positions: Mapping[str, Position]
    risk_level: RiskLevel = RiskLevel.NORMAL
    day_pnl_ratio: float = 0.0
    drawdown: float = 0.0
    orders_today: int = 0
    canceled_today: int = 0

    @property
    def position_count(self) -> int:
        return sum(1 for position in self.positions.values() if not position.is_empty)


class PreTradeRiskRule(Protocol):
    """事前风控规则：逐条订单计划校验，可拒单或缩量。

    ``name`` / ``priority`` 声明为**只读属性**而不是可变属性：规则实现多为
    ``frozen`` dataclass，其字段是只读的——若协议声明可变属性，冻结实现就无法满足它。
    """

    @property
    def name(self) -> str: ...

    @property
    def priority(self) -> int: ...

    def check(self, ctx: RiskContext) -> RiskDecision: ...


class IntradayRiskRule(Protocol):
    """事中风控规则：组合级指标触线时降级（HALT / FLATTEN）。"""

    @property
    def name(self) -> str: ...

    @property
    def priority(self) -> int: ...

    def check_portfolio(self, snapshot: PortfolioSnapshot) -> RiskDecision: ...


__all__ = [
    "AdjType",
    "BrokerAdapter",
    "Context",
    "DataSource",
    "IntradayRiskRule",
    "OrderCallback",
    "PortfolioSnapshot",
    "PreTradeRiskRule",
    "RiskContext",
    "Strategy",
    "TradeCallback",
    "TradingCalendar",
]
