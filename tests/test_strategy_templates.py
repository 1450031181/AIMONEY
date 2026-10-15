"""P1-B5 策略模板与加载器测试。

模板策略必须跑通**完整回测引擎**才算数——单元正确但跑不通引擎的模板没有价值。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from engine.backtest.data import DataFrameBarProvider
from engine.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from engine.errors import StrategyError
from engine.strategy.loader import load_strategy
from engine.strategy.templates.dual_moving_average import DualMovingAverageStrategy
from engine.strategy.templates.momentum_rotation import MomentumRotationStrategy

START = date(2026, 9, 1)
END = date(2026, 10, 30)


def _trend_frame() -> pd.DataFrame:
    """两波段行情：先跌后涨再跌，制造金叉与死叉。"""
    closes: list[float] = (
        [10.0] * 22  # 底部盘整（足够慢均线预热）
        + [10.0 + i * 0.2 for i in range(1, 11)]  # 上涨段 → 金叉
        + [12.0] * 10  # 高位盘整
        + [12.0 - i * 0.3 for i in range(1, 11)]  # 下跌段 → 死叉
    )
    rows = []
    for index, close in enumerate(closes):
        day = pd.Timestamp(START) + pd.Timedelta(days=index)
        if day.dayofweek >= 5:  # 跳过周末，贴近真实交易日分布
            continue
        rows.append(
            {
                "code": "600000",
                "date": day.date().isoformat(),
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
                "amount": close * 1_000_000,
                "adj_factor": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _rotation_frame() -> pd.DataFrame:
    """两只标的：000001 强动量上行，600000 平盘——动量策略应只选前者。"""
    rows = []
    for index in range(30):
        day = (pd.Timestamp(START) + pd.Timedelta(days=index)).date().isoformat()
        rows.append(
            {
                "code": "000001",
                "date": day,
                "open": 20.0 + index * 0.2,
                "high": 20.3 + index * 0.2,
                "low": 19.8 + index * 0.2,
                "close": 20.0 + index * 0.2,
                "volume": 1_000_000,
                "amount": (20.0 + index * 0.2) * 1_000_000,
                "adj_factor": 1.0,
            }
        )
        rows.append(
            {
                "code": "600000",
                "date": day,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1_000_000,
                "amount": 10_000_000,
                "adj_factor": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _run(provider_frame: pd.DataFrame, strategy: Any, **overrides: object) -> BacktestResult:
    options: dict[str, object] = {"slippage_bps": 0.0, **overrides}
    config = BacktestConfig(start=START, end=END, initial_cash="1000000", **options)  # type: ignore[arg-type]
    engine = BacktestEngine(config, provider=DataFrameBarProvider(provider_frame), strategy=strategy)
    return engine.run()


# ============================================================ 双均线
def test_双均线_上涨段产生金叉买入() -> None:
    result = _run(_trend_frame(), DualMovingAverageStrategy(params={"fast": 3, "slow": 10, "target_weight": 0.9}))

    buys = [trade for trade in result.trades if trade.side.value == "BUY"]
    assert buys, "上涨段应出现金叉买入"
    assert buys[0].price > Decimal("10"), "金叉发生在上涨确认之后，成交价应高于底部价"


def test_双均线_下跌段产生死叉清仓() -> None:
    result = _run(_trend_frame(), DualMovingAverageStrategy(params={"fast": 3, "slow": 10}))

    sells = [trade for trade in result.trades if trade.side.value == "SELL"]
    assert sells, "下跌段应出现死叉清仓"
    assert sells[-1].trade_time.date() > sells[0].trade_time.date() if len(sells) > 1 else True


def test_双均线_参数非法直接报错() -> None:
    result = _run(_trend_frame(), DualMovingAverageStrategy(params={"fast": 10, "slow": 5}))

    assert result.halted is not None, "策略抛错应触发停仓而不是让回测崩溃"
    assert "fast" in (result.halted or "")


# ============================================================ 动量轮动
def test_动量轮动_只选强动量标的() -> None:
    result = _run(_rotation_frame(), MomentumRotationStrategy(params={"lookback": 5, "top_k": 1, "rebalance_interval": 5}))

    bought_codes = {trade.code for trade in result.trades if trade.side.value == "BUY"}
    assert bought_codes == {"000001"}, "只应买入动量最强的 000001，平盘的 600000 不入选"


def test_动量轮调仓节奏控制换手() -> None:
    strategy = MomentumRotationStrategy(params={"lookback": 5, "top_k": 1, "rebalance_interval": 5})
    result = _run(_rotation_frame(), strategy)

    trade_days = sorted({trade.trade_time.date() for trade in result.trades})
    assert len(trade_days) <= 5, "30 个交易日、5 日调仓一次，成交日应远少于交易日"


def test_动量轮动_全负动量时空仓() -> None:
    frame = _rotation_frame()
    # 把两只都改成下跌：动量全为负 → 不应产生任何买入
    frame.loc[frame["code"] == "000001", ["open", "high", "low", "close"]] = 20.0 - (
        frame.loc[frame["code"] == "000001", "close"] - 20.0
    )

    result = _run(frame, MomentumRotationStrategy(params={"lookback": 5, "top_k": 1}))

    assert not [trade for trade in result.trades if trade.side.value == "BUY"], "动量全为负时空仓等待"
    assert result.final_account.cash == Decimal("1000000")


# ============================================================ 策略加载器
def _write_strategy(
    directory: Path,
    *,
    body: str,
    config: dict[str, Any] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "strategy.py").write_text(body, encoding="utf-8")
    if config is not None:
        (directory / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return directory


_MINIMAL_BODY = """
from engine.strategy.base import BaseStrategy
from engine.types import Direction, Signal


class AlwaysBuy(BaseStrategy):
    def on_bar(self, context, bars):
        return [
            Signal(
                code=code,
                strategy_id=self.strategy_id,
                trade_date=context.trade_date,
                direction=Direction.BUY,
                target_weight=0.5,
                reason="always",
            )
            for code in sorted(bars)
        ]


def create_strategy(params):
    return AlwaysBuy(strategy_id=str(params.get("strategy_id", "always_buy")), params=params)
"""


def test_加载器加载用户策略并跑通引擎(tmp_path: Path) -> None:
    directory = _write_strategy(tmp_path / "always_buy", body=_MINIMAL_BODY, config={"weight_hint": 0.5})

    loaded = load_strategy(directory)
    assert loaded.strategy_id == "always_buy"
    assert loaded.params["weight_hint"] == 0.5

    result = _run(_trend_frame(), loaded.strategy, slippage_bps=0.0)
    assert result.trades, "用户策略应能驱动完整回测"


def test_加载器_params覆盖config(tmp_path: Path) -> None:
    directory = _write_strategy(tmp_path / "always_buy", body=_MINIMAL_BODY, config={"keep": "from_config"})

    loaded = load_strategy(directory, params_override={"from_cli": 1})
    assert loaded.params["keep"] == "from_config"
    assert loaded.params["from_cli"] == 1


def test_缺少strategy_py时报错(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(StrategyError):
        load_strategy(tmp_path / "empty")


def test_模块崩溃被包装为StrategyError(tmp_path: Path) -> None:
    body = "raise RuntimeError('boom')\n"
    directory = _write_strategy(tmp_path / "broken", body=body)
    with pytest.raises(StrategyError) as excinfo:
        load_strategy(directory)
    assert "boom" in str(excinfo.value)


def test_缺少工厂函数时报错(tmp_path: Path) -> None:
    directory = _write_strategy(tmp_path / "no_factory", body="X = 1\n")
    with pytest.raises(StrategyError):
        load_strategy(directory)


# ============================================================ 策略沙箱（FR-3.5）
def test_策略运行期异常触发停仓但回测继续() -> None:
    class ExplodeMidway(DualMovingAverageStrategy):
        def on_bar(self, context: Any, bars: Any) -> list:
            if context.trade_date.isoformat() >= "2026-09-20":
                raise RuntimeError("策略内部错误")
            return super().on_bar(context, bars)

    result = _run(_trend_frame(), ExplodeMidway(params={"fast": 3, "slow": 10}))

    assert result.halted is not None
    assert "策略内部错误" in result.halted
    assert len(result.nav) > 0, "停仓后净值仍须记录到最后一天"
    assert result.trading_days > 0


def test_初始化失败直接终止回测() -> None:
    class BadInit(DualMovingAverageStrategy):
        def init(self, context: Any) -> None:
            raise RuntimeError("init 失败")

    config = BacktestConfig(start=START, end=END, initial_cash="1000000")
    engine = BacktestEngine(config, provider=DataFrameBarProvider(_trend_frame()), strategy=BadInit())

    with pytest.raises(RuntimeError):
        engine.run()
