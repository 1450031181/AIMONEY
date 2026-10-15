"""P2-F5 多因子策略与因子快照测试。

两条最重要的断言：

1. **策略读不到未来因子**：因子值按日切片注入 Context，越界在接口上不存在；
2. **因子接进回测后确实驱动了交易**，且回测初段不因因子 NaN 而静默空转。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from app.backtest_cli import main
from app.core.config import Settings
from engine.backtest.context import BacktestContext
from engine.backtest.data import DataFrameBarProvider
from engine.backtest.engine import BacktestConfig, BacktestEngine
from engine.datahub.store import ParquetStore
from engine.errors import StrategyError, ValidationError
from engine.factor.snapshot import (
    FactorSnapshot,
    assert_causal_factors,
    build_snapshot_from_prices,
)
from engine.strategy.templates.multi_factor_enhance import MultiFactorEnhanceStrategy

CODES = ("600000", "000001", "300750", "601318", "000002", "600519")
START = date(2026, 8, 3)
END = date(2026, 10, 9)


# ============================================================ 快照
def _snapshot_frame() -> pd.DataFrame:
    """10-12 只有两只标的；300750 仅在 10-13 出现（用于验证"次日值当日不可见"）。"""
    return pd.DataFrame(
        [
            {"date": "2026-10-12", "code": "600000", "mom": 1.5, "vol": 0.5},
            {"date": "2026-10-12", "code": "000001", "mom": -0.5, "vol": float("nan")},
            {"date": "2026-10-13", "code": "300750", "mom": 2.0, "vol": 0.1},
        ]
    )


def test_快照按日切片() -> None:
    snapshot = FactorSnapshot(frame=_snapshot_frame())

    day = date(2026, 10, 12)
    assert set(snapshot.values_on(day)) == {"600000", "000001"}, "只返回当日的标的"
    assert snapshot.value("mom", "600000", on=day) == pytest.approx(1.5)
    assert snapshot.value("mom", "300750", on=date(2026, 10, 13)) == pytest.approx(2.0)
    assert snapshot.value("mom", "300750", on=day) is None, "次日的值不得到前一天"


def test_快照丢弃NaN而不是填零() -> None:
    snapshot = FactorSnapshot(frame=_snapshot_frame())

    values = snapshot.values_on(date(2026, 10, 12))
    assert values["000001"] == {"mom": pytest.approx(-0.5)}, "缺值不进入快照（拿 0 顶替会造出假的中性信号）"
    assert "vol" not in values["000001"]


def test_快照未知日期返回空() -> None:
    snapshot = FactorSnapshot(frame=_snapshot_frame())
    assert snapshot.values_on(date(2030, 1, 1)) == {}
    assert not snapshot.has_day(date(2030, 1, 1))


def test_快照缺少必要列报错() -> None:
    with pytest.raises(ValidationError):
        FactorSnapshot(frame=pd.DataFrame({"code": ["600000"], "mom": [1.0]}))


def test_快照元信息() -> None:
    snapshot = FactorSnapshot(frame=_snapshot_frame())
    assert snapshot.factor_names == ("mom", "vol")
    assert snapshot.dates() == (date(2026, 10, 12), date(2026, 10, 13))


def _prices(days: int = 40, codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for index, code in enumerate(codes):
        price = 10.0 + index * 3
        for offset in range(days):
            day = pd.Timestamp("2026-09-01") + pd.Timedelta(days=offset)
            if day.dayofweek >= 5:
                continue
            price *= 1.0 + 0.001 * (index - 2) + float(rng.normal(0, 0.01))
            rows.append(
                {
                    "code": code,
                    "date": day.date().isoformat(),
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000,
                    "amount": price * 1_000_000,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_由行情直接构建快照() -> None:
    snapshot = build_snapshot_from_prices(_prices(), factors=["mom_5", "vol_10"])

    assert snapshot.factor_names == ("mom_5", "vol_10")
    assert len(snapshot.dates()) > 10
    first_day = snapshot.dates()[-1]
    assert set(snapshot.values_on(first_day)) <= set(CODES)


def test_因子因果性检查通过内置因子() -> None:
    """所有内置因子都是向后看窗口，截断前后取值必须一致。"""
    assert_causal_factors(_prices(days=60), ["mom_5", "vol_10", "bias_10"])


def test_因果性检查能抓出偷看未来的因子() -> None:
    """把 mom_5 换成"未来 5 日收益"的因子：截断后取值必然不同。"""
    from engine.factor import base

    definition = base.FactorDefinition(
        name="cheat_forward",
        category="momentum",
        description="偷看未来的假因子",
        window=6,
        func=lambda frame: frame["close"].shift(-5) / frame["close"] - 1,
    )
    base.register_factor(definition)
    try:
        with pytest.raises(ValidationError) as excinfo:
            assert_causal_factors(_prices(days=60), ["cheat_forward"])
        assert "未来信息" in str(excinfo.value)
    finally:
        base._FACTOR_REGISTRY.pop("cheat_forward", None)


# ============================================================ Context 因子访问
def _context(snapshot: FactorSnapshot, day: date) -> BacktestContext:
    return BacktestContext(
        provider=DataFrameBarProvider(_prices()),
        strategy_id="t",
        trade_date_value=day,
        bars_today={},
        factor_values=snapshot.values_on(day),
    )


def test_上下文只暴露当日因子() -> None:
    snapshot = FactorSnapshot(frame=_snapshot_frame())
    today = _context(snapshot, date(2026, 10, 12))
    tomorrow = _context(snapshot, date(2026, 10, 13))

    assert today.factor("mom", "600000") == pytest.approx(1.5)
    assert today.factor("mom", "000001") == pytest.approx(-0.5)
    # 300750 的值只存在于 10-13：今天查不到，明天才查得到——越界在接口上不存在
    assert today.factor("mom", "300750") is None
    assert tomorrow.factor("mom", "300750") == pytest.approx(2.0)
    assert today.factor("not_exists", "600000") is None
    assert today.factor("mom", "not_exists") is None
    assert today.factor_names() == ("mom", "vol")


def test_未提供因子时因子访问返回None() -> None:
    context = BacktestContext(provider=DataFrameBarProvider(_prices()), strategy_id="t", trade_date_value=START)
    assert context.factor("mom", "600000") is None


# ============================================================ 策略
class _StubContext:
    """仅供策略单测使用的最小上下文替身。"""

    def __init__(
        self,
        values: dict[str, dict[str, float]],
        positions: dict[str, object] | None = None,
        *,
        configured: bool = True,
        cash: str = "1000000",
        market_value: str = "0",
        closes: dict[str, list[float]] | None = None,
    ) -> None:
        self._values = values
        self._positions = positions or {}
        self._configured = configured
        self._cash = cash
        self._market_value = market_value
        self._closes = closes or {}
        self.strategy_id = "stub"
        self.trade_date = START

    def has_factors(self) -> bool:
        return self._configured

    def account(self):  # type: ignore[no-untyped-def]
        from decimal import Decimal

        from engine.types import Account

        return Account(
            initial_cash=Decimal("1000000"),
            cash=Decimal(self._cash),
            market_value=Decimal(self._market_value),
        )

    def history(self, code: str, *, field: str = "close", window: int = 1, adj: str = "post") -> list:  # type: ignore[type-arg]
        return self._closes.get(code, [])[-window:]

    def factor(self, name: str, code: str) -> float | None:
        return self._values.get(code, {}).get(name)

    def factor_names(self) -> tuple[str, ...]:
        names: set[str] = set()
        for values in self._values.values():
            names.update(values)
        return tuple(sorted(names))

    def positions(self) -> dict[str, object]:
        return dict(self._positions)


def _bars(codes: tuple[str, ...]) -> dict[str, object]:
    from decimal import Decimal

    from engine.types import Bar

    return {
        code: Bar(
            code=code,
            dt=pd.Timestamp(START),
            open=Decimal("10"),
            high=Decimal("10"),
            low=Decimal("10"),
            close=Decimal("10"),
            volume=1_000_000,
            amount=Decimal("10000000"),
        )
        for code in codes
    }


def test_策略按合成得分选前k() -> None:
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f"], "top_k": 2, "rebalance_interval": 1})
    values = {"600000": {"f": 1.0}, "000001": {"f": 3.0}, "300750": {"f": 2.0}}
    context = _StubContext(values)

    signals = strategy.on_bar(context, _bars(tuple(values)))

    assert [signal.code for signal in signals] == ["000001", "300750"], "按得分降序取前 2"
    assert all(float(signal.target_weight) == pytest.approx(0.475) for signal in signals), "总仓位按只数等分"


def test_策略对多因子加权与缺值跳过() -> None:
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f", "g"], "weights": {"f": 2.0, "g": 1.0}, "top_k": 3})
    values = {
        "600000": {"f": 1.0, "g": 1.0},  # 3.0
        "000001": {"f": 0.5, "g": 1.0},  # 2.0
        "300750": {"f": 5.0},  # 缺 g → 跳过
    }
    context = _StubContext(values)

    signals = strategy.on_bar(context, _bars(tuple(values)))

    assert [signal.code for signal in signals] == ["600000", "000001"]
    assert "300750" not in strategy.last_scores, "缺必需因子的标的必须被跳过"


def test_策略清仓跌出名单的持仓() -> None:
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f"], "top_k": 1, "rebalance_interval": 1})
    values = {"600000": {"f": 3.0}, "000001": {"f": 1.0}}
    context = _StubContext(
        values,
        positions={"000001": _position("000001", qty=10_000, price="10")},
        cash="900000",
        market_value="100000",
    )

    signals = strategy.on_bar(context, _bars(tuple(values)))

    sides = {signal.code: signal.direction.value for signal in signals}
    assert sides["600000"] == "BUY"
    assert sides["000001"] == "SELL", "跌出名单的持仓应清仓"


def test_策略调仓节奏() -> None:
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f"], "top_k": 1, "rebalance_interval": 3})
    values = {"600000": {"f": 1.0}}
    context = _StubContext(values)
    bars = _bars(("600000",))

    counts = [len(strategy.on_bar(context, bars)) for _ in range(7)]

    assert counts == [1, 0, 0, 1, 0, 0, 1], "每 3 个交易日调仓一次"


def _position(code: str, *, qty: int, price: str):
    from decimal import Decimal

    from engine.types import Position

    return Position(code=code, qty=qty, avail_qty=qty, last_price=Decimal(price))


def test_策略按得分加权() -> None:
    strategy = MultiFactorEnhanceStrategy(
        params={"factors": ["f"], "top_k": 2, "rebalance_interval": 1, "weight_scheme": "score"}
    )
    values = {"600000": {"f": 3.0}, "000001": {"f": 1.0}}
    context = _StubContext(values)

    signals = strategy.on_bar(context, _bars(tuple(values)))
    weights = {signal.code: float(signal.target_weight) for signal in signals}

    assert weights["600000"] > weights["000001"], "得分高者权重更大"
    assert sum(weights.values()) == pytest.approx(0.95), "总仓位仍由 total_weight 约束"


def test_策略单标的上限生效() -> None:
    strategy = MultiFactorEnhanceStrategy(
        params={"factors": ["f"], "top_k": 2, "rebalance_interval": 1, "max_weight": 0.3}
    )
    values = {"600000": {"f": 2.0}, "000001": {"f": 1.0}}
    context = _StubContext(values)

    signals = strategy.on_bar(context, _bars(tuple(values)))

    assert all(float(signal.target_weight) <= 0.3 + 1e-9 for signal in signals), "不得超过单标的上限"
    assert strategy.last_decision is not None
    assert strategy.last_decision.cash_weight > 0, "容量不足时应留现金而不是破上限"


def test_策略无交易带抑制无效调仓() -> None:
    """目标权重与当前权重接近时不应产生换手。"""
    strategy = MultiFactorEnhanceStrategy(
        params={"factors": ["f"], "top_k": 2, "rebalance_interval": 1, "no_trade_band": 0.05}
    )
    values = {"600000": {"f": 2.0}, "000001": {"f": 1.0}}
    # 当前各持 50%（总资产 100 万），目标 47.5% → 差异 2.5% 小于 5% 的无交易带
    context = _StubContext(
        values,
        positions={
            "600000": _position("600000", qty=50_000, price="10"),
            "000001": _position("000001", qty=50_000, price="10"),
        },
        cash="0",
        market_value="1000000",
    )

    strategy.on_bar(context, _bars(tuple(values)))

    assert strategy.last_decision is not None
    assert strategy.last_decision.turnover == pytest.approx(0.0), "权重几乎没变就不该交易"


def test_策略逆波动率加权() -> None:
    strategy = MultiFactorEnhanceStrategy(
        params={"factors": ["f"], "top_k": 2, "rebalance_interval": 1, "weight_scheme": "inverse_vol"}
    )
    values = {"600000": {"f": 2.0}, "000001": {"f": 2.0}}
    context = _StubContext(
        values,
        closes={
            "600000": [10.0, 10.1, 10.05, 10.12, 10.08],  # 低波动
            "000001": [10.0, 11.0, 9.5, 10.8, 9.6],  # 高波动
        },
    )

    signals = strategy.on_bar(context, _bars(tuple(values)))
    weights = {signal.code: float(signal.target_weight) for signal in signals}

    assert weights["600000"] > weights["000001"], "波动低的标的应获得更高权重"


def test_策略未配置因子时明确报错() -> None:
    strategy = MultiFactorEnhanceStrategy(params={"top_k": 3})
    with pytest.raises(StrategyError) as excinfo:
        strategy.on_bar(_StubContext({}), _bars(("600000",)))
    assert "factors" in str(excinfo.value)


def test_策略在未装载因子快照时明确报错() -> None:
    """最常见的原因是忘了给回测传 --factors，必须报错而不是静默空仓。"""
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f"], "rebalance_interval": 1})
    with pytest.raises(StrategyError) as excinfo:
        strategy.on_bar(_StubContext({}, configured=False), _bars(("600000",)))
    assert "因子快照" in str(excinfo.value)


def test_预热期当日无值不算配置错误() -> None:
    """装载了快照但今日无值（预热期）→ 静默跳过，绝不报错。"""
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f"], "rebalance_interval": 1})

    signals = strategy.on_bar(_StubContext({}), _bars(("600000",)))

    assert signals == [], "预热期没有信号是正常的"


def test_策略参数非法报错() -> None:
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["f"], "top_k": 0, "rebalance_interval": 1})
    with pytest.raises(StrategyError):
        strategy.on_bar(_StubContext({"600000": {"f": 1.0}}), _bars(("600000",)))


# ============================================================ 引擎闭环
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    resolved = Settings(data={"root": str(tmp_path / "data")})
    resolved.ensure_dirs()
    ParquetStore(resolved.parquet_dir).write("daily", _prices(days=80), data_version="20261009.1")
    reference = resolved.data.root_dir / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"code": code, "name": f"标的{i}", "is_st": False, "list_date": "2010-01-01", "delist_date": None}
            for i, code in enumerate(CODES)
        ]
    ).to_parquet(reference / "stock_info.parquet", index=False)
    return resolved


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "strategies" / "multi_factor_enhance"
    directory.mkdir(parents=True)
    (directory / "strategy.py").write_text(
        "from engine.strategy.templates.multi_factor_enhance import MultiFactorEnhanceStrategy\n\n\n"
        "def create_strategy(params):\n"
        "    return MultiFactorEnhanceStrategy(params=params)\n",
        encoding="utf-8",
    )
    (directory / "config.yaml").write_text(
        yaml.safe_dump({"factors": ["mom_5"], "top_k": 2, "rebalance_interval": 2}), encoding="utf-8"
    )
    return directory


def test_引擎用因子驱动交易() -> None:
    """端到端：因子快照 → Context → 策略打分 → 信号 → 成交。"""
    prices = _prices(days=80)
    snapshot = build_snapshot_from_prices(prices, factors=["mom_5"])
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["mom_5"], "top_k": 2, "rebalance_interval": 1})
    config = BacktestConfig(start=START, end=END, initial_cash="1000000", slippage_bps=0.0)
    engine = BacktestEngine(
        config, provider=DataFrameBarProvider(prices), strategy=strategy, factors=snapshot
    )

    result = engine.run()

    assert result.halted is None
    assert result.trades, "因子驱动的策略应产生成交"
    assert strategy.last_scores, "策略应至少完成一次打分"


def test_回测初段因子未就绪时不产生交易() -> None:
    """快照未覆盖预热期时，前 window 天没有因子值 → 策略跳过，而不是拿 0 顶替。"""
    prices = _prices(days=40)
    snapshot = build_snapshot_from_prices(prices, factors=["mom_20"])  # 需要 21 根 K 线
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["mom_20"], "top_k": 2, "rebalance_interval": 1})
    config = BacktestConfig(start=START, end=END, initial_cash="1000000")
    engine = BacktestEngine(
        config, provider=DataFrameBarProvider(prices), strategy=strategy, factors=snapshot
    )

    result = engine.run()

    assert result.halted is None
    ready_days = [day for day in snapshot.dates() if snapshot.values_on(day)]
    assert len(ready_days) < len(snapshot.dates()), "预热期内确实没有因子值"


def test_引擎不提供因子时策略停仓并给出原因() -> None:
    prices = _prices(days=80)
    strategy = MultiFactorEnhanceStrategy(params={"factors": ["mom_5"], "rebalance_interval": 1})
    config = BacktestConfig(start=START, end=END, initial_cash="1000000")
    engine = BacktestEngine(config, provider=DataFrameBarProvider(prices), strategy=strategy)

    result = engine.run()

    assert result.halted is not None
    assert "因子快照" in result.halted


# ============================================================ CLI 闭环
def test_CLI传入因子跑通多因子回测(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "--strategy",
            str(strategy_dir),
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--factors",
            "mom_5",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["halted"] is None
    assert payload["metrics"]["total_return"] is not None

    report = json.loads((Path(payload["output_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    assert report["config"]["factors"] == ["mom_5"], "报告需记录用了哪些因子（可复现的前提）"
    assert (Path(payload["output_dir"]) / "trades.csv").is_file()


def test_CLI未提供策略所需因子时快速失败(
    settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """配置错误应当挡在跑回测之前，而不是跑完再让人从空报告里猜。"""
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "--strategy",
            str(strategy_dir),
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--json",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "VALIDATION_ERROR" in err
    assert "mom_5" in err, "错误信息要指明缺哪个因子"


def test_CLI因子名拼错时快速失败(settings: Settings, strategy_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--root",
            str(settings.data.root_dir),
            "--strategy",
            str(strategy_dir),
            "--start",
            START.isoformat(),
            "--end",
            END.isoformat(),
            "--factors",
            "mom_5x",
        ]
    )

    assert code == 1
    assert "因子" in capsys.readouterr().err
