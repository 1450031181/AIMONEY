"""P1-B3 回测引擎主循环测试。

这批用例验证的是"整条链路的正确性"，任何一条红都意味着回测结果不可信：
- 信号与成交强制错开（T 日信号 → T+1 开盘成交）；
- T+1：撮合日收盘时 avail 仍为 0，日切后才可卖；
- 资金/持仓守恒；
- 停牌顺延与过期；
- 卖出优先于买入（资金复用）；
- Context 防未来函数。
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest

from engine.backtest.context import BacktestContext
from engine.backtest.data import DataFrameBarProvider
from engine.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from engine.errors import ValidationError
from engine.ports import Strategy
from engine.types import Bar, Direction, Side, Signal, Trade

START = date(2026, 10, 12)
END = date(2026, 10, 16)


def _bars_frame() -> pd.DataFrame:
    """4 个交易日、3 只标的的行情（600000 平稳、000001 上涨、300750 停牌两天）。"""
    rows = []
    prices = {
        "600000": {12: 10.0, 13: 10.0, 14: 10.0, 15: 10.0, 16: 10.0},
        "000001": {12: 20.0, 13: 20.5, 14: 21.0, 15: 21.5, 16: 22.0},
    }
    for code, series in prices.items():
        for day, close in series.items():
            rows.append(
                {
                    "code": code,
                    "date": f"2026-10-{day:02d}",
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                    "adj_factor": 1.0,
                }
            )
    # 300750：12、13 停牌，14 起恢复
    for day in (12, 13, 14, 15, 16):
        suspended = day in (12, 13)
        rows.append(
            {
                "code": "300750",
                "date": f"2026-10-{day:02d}",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 0 if suspended else 500_000,
                "amount": 0 if suspended else 50_000_000,
                "is_suspended": suspended,
                "adj_factor": 1.0,
            }
        )
    return pd.DataFrame(rows)


class FixedWeightStrategy:
    """每日对指定标的发出固定目标权重信号（测试用策略）。"""

    def __init__(self, targets: dict[str, float], *, strategy_id: str = "fixed") -> None:
        self.targets = targets
        self.strategy_id = strategy_id
        self.seen_contexts: list[dict[str, object]] = []

    def init(self, context: object) -> None:
        return None

    def _signal(self, code: str, weight: float, context: BacktestContext) -> Signal:
        return Signal(
            code=code,
            strategy_id=self.strategy_id,
            trade_date=context.trade_date,
            direction=Direction.BUY if weight > 0 else Direction.SELL,
            target_weight=weight,
        )

    def on_bar(self, context: object, bars: dict[str, Bar]) -> list[Signal]:
        assert isinstance(context, BacktestContext)
        return [
            self._signal(code, weight, context)
            for code, weight in self.targets.items()
            if code in bars
        ]

    def on_order_filled(self, context: object, trade: Trade) -> None:
        return None


class OneShotStrategy(FixedWeightStrategy):
    """只在指定交易日发信号（其余日子静默），用于精确断言成交次数与价格。"""

    def __init__(
        self, targets: dict[str, float], *, signal_days: set[date] | None = None, strategy_id: str = "oneshot"
    ) -> None:
        super().__init__(targets, strategy_id=strategy_id)
        self.signal_days = signal_days or {START}

    def on_bar(self, context: object, bars: dict[str, Bar]) -> list[Signal]:
        assert isinstance(context, BacktestContext)
        # 每个交易日都记录观察值（无论是否发信号），供 T+1 等时序断言使用
        self.seen_contexts.append(
            {"day": context.trade_date, "avail": {code: pos.avail_qty for code, pos in context.positions().items()}}
        )
        if context.trade_date not in self.signal_days:
            return []
        return super().on_bar(context, bars)


def _run(strategy: Strategy, **overrides: object) -> BacktestResult:
    # 默认零滑点：让成交价可手算断言；滑点本身的行为在撮合器测试里单独验证
    options: dict[str, object] = {"slippage_bps": 0.0, **overrides}
    config = BacktestConfig(start=START, end=END, initial_cash="1000000", **options)  # type: ignore[arg-type]
    engine = BacktestEngine(config, provider=DataFrameBarProvider(_bars_frame()), strategy=strategy)
    return engine.run()


def _adjusted_frame() -> pd.DataFrame:
    """真实价 10 元、存储（后复权）价 20 元：模拟"有过分红送转"的标的。"""
    return pd.DataFrame(
        [
            {
                "code": "600000",
                "date": f"2026-10-{day:02d}",
                "open": 20.0,
                "high": 20.0,
                "low": 20.0,
                "close": 20.0,
                "volume": 1_000_000,
                "amount": 20_000_000,
                "adj_factor": 2.0,
            }
            for day in (12, 13, 14)
        ]
    )


# ============================================================ 复权口径（回归）
def test_撮合按真实价而非后复权价成交() -> None:
    """回归：曾把**后复权价**直接喂给撮合器。

    在没有分红送转的数据（``adj_factor == 1``）上完全看不出来；一旦因子 ≠ 1，
    用真实人民币按放大后的价格成交 → 仓位被系统性缩小 ``1/F``，
    且与上游的真实价涨跌停比较必然错乱。这类错误不会报任何异常。
    """
    config = BacktestConfig(start=START, end=date(2026, 10, 14), initial_cash="1000000", slippage_bps=0.0)
    engine = BacktestEngine(
        config,
        provider=DataFrameBarProvider(_adjusted_frame()),
        strategy=OneShotStrategy({"600000": 0.95}),
    )
    result = engine.run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.price == Decimal("10.00"), "成交价必须是真实价（后复权价 20 ÷ 因子 2）"
    assert trade.qty == 95_000, "95 万 ÷ 真实价 10 = 95,000 股；若用后复权价会只剩 47,500 股"
    assert result.final_account.market_value == Decimal("950000.00")


def test_bar取真实价而history取后复权() -> None:
    """两条通道的口径必须分开：``bars`` 是执行口径，``history`` 是计算口径。

    若 ``history`` 也返回真实价，除权日会产生假跳变（收益/均线/动量全部失真）；
    若 ``bars`` 返回后复权价，撮合与涨跌停判断全部错。
    """
    provider = DataFrameBarProvider(_adjusted_frame())
    bar = provider.bars_on(START)["600000"]

    assert bar.close == Decimal("10.00"), "执行口径：真实价"
    assert bar.adj_factor == Decimal("2.0"), "因子保留在 bar 上供还原与排查"
    assert provider.history("600000", field="close", end=START, window=1) == [Decimal("20.0")], (
        "计算口径：后复权"
    )


# ============================================================ 基本成交
def test_信号次日开盘成交而非当日() -> None:
    # 权重取 0.95：目标数量低于含费可买上限，一笔即可完成（权重 1.0 会因费用产生部分成交余量）
    result = _run(OneShotStrategy({"600000": 0.95}))

    assert len(result.trades) == 1, "每日重复发信号会不断用零头补仓，须用一次性策略才能精确断言"
    trade = result.trades[0]
    assert trade.trade_time.date() == date(2026, 10, 13), "10-12 收盘的信号最早 10-13 开盘成交"
    assert trade.price == Decimal("10.00")


def test_买入后现金与持仓正确且守恒() -> None:
    result = _run(OneShotStrategy({"600000": 0.95}))

    trade = result.trades[0]
    account = result.final_account

    assert account.market_value == Decimal("950000.00"), "95% 仓位 × 100 万 = 95 万市值"
    assert account.total_asset == account.cash + account.market_value
    expected_cash = Decimal("1000000") + trade.cash_flow  # 买入的 cash_flow 为负
    assert account.cash == expected_cash, "现金变动必须完全由 Trade.cash_flow 驱动"


def test_单日盯市净值等于总资产比初始资金() -> None:
    result = _run(FixedWeightStrategy({"000001": 1.0}))

    first = result.nav.iloc[0]
    assert first == pytest.approx(1.0), "首日无持仓，净值应为 1.0"
    last = result.nav.iloc[-1]
    account = result.final_account
    assert last == pytest.approx(float(account.total_asset / account.initial_cash))


# ============================================================ T+1
def test_T加1_撮合日收盘时可卖数量仍为零() -> None:
    strategy = OneShotStrategy({"600000": 0.95})
    result = _run(strategy)

    # 10-13 撮合买入，当日收盘策略视角下 avail 必须仍是 0
    seen = {entry["day"]: entry["avail"] for entry in strategy.seen_contexts}
    assert seen[date(2026, 10, 13)]["600000"] == 0, "T+1：当日买入不可卖"

    # 10-14（日切之后）可卖数量解冻
    assert seen[date(2026, 10, 14)]["600000"] == result.trades[0].qty


def test_T加1_次日可正常卖出() -> None:
    class BuyThenSell(FixedWeightStrategy):
        def on_bar(self, context: object, bars: dict[str, Bar]) -> list[Signal]:
            assert isinstance(context, BacktestContext)
            if context.trade_date == START:  # 首日买入
                return [self._signal("600000", 0.95, context)]
            if context.trade_date == date(2026, 10, 14):  # 次日清仓
                return [self._signal("600000", -1.0, context)]
            return []

    result = _run(BuyThenSell({}))

    sides = [trade.side for trade in result.trades]
    assert sides == [Side.BUY, Side.SELL], "买入后次日应能卖出"
    assert result.trades[1].price == Decimal("10.00")


# ============================================================ 停牌顺延与过期
def test_停牌期间挂单顺延_复牌后成交() -> None:
    result = _run(OneShotStrategy({"300750": 0.95}))

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.trade_time.date() == date(2026, 10, 14), "10-12 的信号在 10-13 停牌顺延，14 日复牌成交"
    assert result.trading_days == 5


def test_长期停牌挂单过期() -> None:
    class StartOnlyStrategy(FixedWeightStrategy):
        def on_bar(self, context: object, bars: dict[str, Bar]) -> list[Signal]:
            if context.trade_date != START:
                return []
            return super().on_bar(context, bars)

    # max_delay_days=0：10-13 首次顺延即超限，立即放弃
    config = BacktestConfig(start=START, end=END, initial_cash="1000000", max_delay_days=0, slippage_bps=0.0)
    engine = BacktestEngine(config, provider=DataFrameBarProvider(_bars_frame()), strategy=StartOnlyStrategy({"300750": 1.0}))
    result = engine.run()

    assert len(result.trades) == 0
    assert len(result.expired) == 1, "顺延超过 max_delay_days 应放弃挂单"
    assert result.final_account.cash == Decimal("1000000"), "未成交则现金不变"


# ============================================================ 卖出优先
def test_卖出优先于买入使资金可复用() -> None:
    class RotateStrategy(FixedWeightStrategy):
        def on_bar(self, context: object, bars: dict[str, Bar]) -> list[Signal]:
            assert isinstance(context, BacktestContext)
            if context.trade_date == START:
                return [self._signal("600000", 0.95, context)]
            if context.trade_date == date(2026, 10, 14):
                return [
                    self._signal("600000", -1.0, context),
                    self._signal("000001", 0.95, context),
                ]
            return []

    result = _run(RotateStrategy({}))

    day14 = [trade for trade in result.trades if trade.trade_time.date() == date(2026, 10, 15)]
    assert [trade.side for trade in day14] == [Side.SELL, Side.BUY], "同日撮合必须先卖后买"
    buy = day14[1]
    assert buy.qty > 0, "卖出的资金必须可用于当日买入"


# ============================================================ 防未来函数
def test_历史查询拿不到未来数据() -> None:
    provider = DataFrameBarProvider(_bars_frame())
    # end=10-14 时窗口再多也只能拿到 <= 10-14 的数据
    history = provider.history("000001", end=date(2026, 10, 14), window=10)

    assert len(history) == 3, "10-12、10-13、10-14 共 3 根，绝不会泄漏 10-15/16"


def test_策略看到的bars只有当日() -> None:
    class ProbeStrategy(FixedWeightStrategy):
        def on_bar(self, context: object, bars: dict[str, Bar]) -> list[Signal]:
            assert isinstance(context, BacktestContext)
            if context.trade_date == START:
                assert max(bar.dt for bar in bars.values()) == datetime(2026, 10, 12), "策略只应看到当日行情"
            return []

    _run(ProbeStrategy({}))


def test_区间外无交易日时报错() -> None:
    config = BacktestConfig(start=date(2027, 1, 1), end=date(2027, 1, 31))
    engine = BacktestEngine(config, provider=DataFrameBarProvider(_bars_frame()), strategy=FixedWeightStrategy({}))

    with pytest.raises(ValidationError):
        engine.run()


def test_区间非法时报错() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(start=END, end=START)


# ============================================================ 指标与结果
def test_上涨标的策略收益为正且指标齐全() -> None:
    result = _run(FixedWeightStrategy({"000001": 1.0}))

    assert result.metrics.total_return > 0
    assert result.trading_days == 5
    assert result.total_signals == 5
    assert not result.expired
    frame = result.trades_frame()
    assert list(frame.columns) >= ["code", "side", "qty", "price", "total_fee"]
    payload = result.metrics.to_jsonable()
    assert payload["total_return"] > 0
