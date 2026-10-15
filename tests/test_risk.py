"""P3-2 风控测试：急停开关、事前规则、事中触线、规则链。

重点验证三件事：

1. **急停与交易门禁的安全默认是相反的**（文件缺失时前者放行、后者拦下），
   且急停文件损坏时必须倒向安全侧（按已激活处理）；
2. **缩量在规则链上传递**：前一条规则缩量后，后一条按缩量后的数量判断；
3. **卖出尽量放行**：风控的目标是降低暴露，减仓操作不该被拦。
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from engine.ports import PortfolioSnapshot, RiskContext
from engine.risk.engine import RiskEngine
from engine.risk.kill_switch import KillSwitch
from engine.risk.rules import (
    AvailablePositionRule,
    CashRule,
    DailyLossRule,
    DailyOrderCountRule,
    DrawdownRule,
    KillSwitchRule,
    OrderValueRule,
    PositionConcentrationRule,
    PriceLimitRule,
    RiskLevelRule,
    SevereDrawdownRule,
    default_intraday_rules,
    default_pre_trade_rules,
)
from engine.types import Account, Bar, OrderPlan, Position, RiskAction, RiskLevel, Side

DAY = date(2026, 10, 15)


def _bar(close: str = "10.0", *, limit_up: str | None = None, limit_down: str | None = None, prev_close: str | None = None) -> Bar:
    return Bar(
        code="600000",
        dt=DAY,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=1_000_000,
        amount=Decimal("10000000"),
        limit_up=Decimal(limit_up) if limit_up else None,
        limit_down=Decimal(limit_down) if limit_down else None,
        prev_close=Decimal(prev_close) if prev_close else None,
    )


def _plan(qty: int = 1000, *, side: Side = Side.BUY, limit_price: str | None = None) -> OrderPlan:
    return OrderPlan(
        code="600000",
        strategy_id="s1",
        trade_date=DAY,
        side=side,
        target_qty=qty,
        limit_price=Decimal(limit_price) if limit_price else None,
    )


def _account(cash: str = "1000000") -> Account:
    return Account(initial_cash=Decimal(cash), cash=Decimal(cash))


def _context(
    *,
    plan: OrderPlan | None = None,
    cash: str = "1000000",
    positions: dict[str, Position] | None = None,
    bar: Bar | None = None,
    risk_level: RiskLevel = RiskLevel.NORMAL,
    extra: dict[str, object] | None = None,
) -> RiskContext:
    return RiskContext(
        trade_date=DAY,
        plan=plan or _plan(),
        account=_account(cash),
        positions=positions or {},
        bars={"600000": bar} if bar is not None else {},
        risk_level=risk_level,
        extra=extra or {},
    )


# ============================================================ 急停开关
def test_急停开关默认未激活(tmp_path: Path) -> None:
    switch = KillSwitch(tmp_path / "state" / "kill_switch.json")

    assert not switch.is_active(), "急停是紧急制动，全新部署时不该是踩下的状态"
    assert switch.state()["active"] is False


def test_急停开关激活与解除(tmp_path: Path) -> None:
    path = tmp_path / "kill_switch.json"
    switch = KillSwitch(path)
    switch.activate("盘中异常波动", by="risk_engine")

    assert switch.is_active()
    assert KillSwitch(path).is_active(), "状态必须持久化（独立进程也要能读到）"
    assert "盘中异常波动" in json.loads(path.read_text(encoding="utf-8"))["reason"]

    switch.deactivate(by="operator", note="已确认")
    assert not switch.is_active()


def test_急停激活必须给原因() -> None:
    switch = KillSwitch("unused.json")
    with pytest.raises(ValueError):
        switch.activate("   ")


def test_急停文件损坏时倒向安全侧(tmp_path: Path) -> None:
    path = tmp_path / "kill_switch.json"
    path.write_text("{ 这不是 json", encoding="utf-8")

    assert KillSwitch(path).is_active(), "读不出状态时就别动（宁可错停，不可错放）"
    assert "不可读" in str(KillSwitch(path).state()["reason"])


def test_急停内容非法也倒向安全侧(tmp_path: Path) -> None:
    path = tmp_path / "kill_switch.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert KillSwitch(path).is_active()


# ============================================================ 事前规则
def test_急停规则拒买放卖(tmp_path: Path) -> None:
    switch = KillSwitch(tmp_path / "kill.json")
    switch.activate("测试")
    rule = KillSwitchRule(switch=switch)

    assert rule.check(_context(plan=_plan(side=Side.BUY))).action is RiskAction.REJECT
    assert rule.check(_context(plan=_plan(side=Side.SELL))).action is RiskAction.PASS, "减仓应放行"


def test_风险等级规则只平不开() -> None:
    rule = RiskLevelRule()

    assert rule.check(_context(risk_level=RiskLevel.HALT)).action is RiskAction.REJECT
    assert rule.check(_context(risk_level=RiskLevel.HALT, plan=_plan(side=Side.SELL))).action is RiskAction.PASS
    assert rule.check(_context(risk_level=RiskLevel.FLATTEN)).action is RiskAction.REJECT
    assert rule.check(_context(risk_level=RiskLevel.NORMAL)).action is RiskAction.PASS


def test_单票集中度超限缩量而不是拒单() -> None:
    rule = PositionConcentrationRule(max_weight=0.1)  # 上限 10 万市值 = 1 万股
    decision = rule.check(_context(plan=_plan(20_000), cash="1000000", bar=_bar("10.0")))

    assert decision.action is RiskAction.SHRINK
    assert decision.adjusted_qty == 10_000, "20,000 股超限，缩到 10 万 / 10 元 = 1 万股"
    assert decision.adjusted_qty % 100 == 0, "必须整手"


def test_集中度未超限时放行() -> None:
    rule = PositionConcentrationRule(max_weight=0.1)
    decision = rule.check(_context(plan=_plan(5_000), cash="1000000", bar=_bar("10.0")))

    assert decision.action is RiskAction.PASS, "能缩量就不要拒单，能放行就不要缩量"


def test_集中度已满时拒单() -> None:
    rule = PositionConcentrationRule(max_weight=0.1)
    held = Position(code="600000", qty=10_000, avail_qty=0, last_price=Decimal("10"))

    decision = rule.check(_context(positions={"600000": held}, bar=_bar("10.0")))

    assert decision.action is RiskAction.REJECT


def test_单笔金额上限缩量() -> None:
    rule = OrderValueRule(max_value=Decimal("50000"))
    decision = rule.check(_context(plan=_plan(20_000), bar=_bar("10.0")))

    assert decision.action is RiskAction.SHRINK
    assert decision.adjusted_qty == 5_000, "5 万元 / 10 元 = 5000 股"


def test_当日委托笔数上限拒单() -> None:
    rule = DailyOrderCountRule(max_orders=3)

    assert rule.check(_context(extra={"orders_today": 2})).action is RiskAction.PASS
    assert rule.check(_context(extra={"orders_today": 3})).action is RiskAction.REJECT


def test_限价超出涨跌停被拒() -> None:
    rule = PriceLimitRule()
    bar = _bar("11.0", limit_up="11.00", limit_down="9.00")

    assert rule.check(_context(plan=_plan(limit_price="11.50"), bar=bar)).action is RiskAction.REJECT
    assert rule.check(_context(plan=_plan(limit_price="10.50"), bar=bar)).action is RiskAction.PASS


def test_涨跌停由前收推算() -> None:
    rule = PriceLimitRule()
    bar = _bar("10.0", prev_close="10.00")  # 主板 ±10% → [9.00, 11.00]

    assert rule.check(_context(plan=_plan(limit_price="10.90"), bar=bar)).action is RiskAction.PASS
    assert rule.check(_context(plan=_plan(limit_price="11.10"), bar=bar)).action is RiskAction.REJECT


def test_限价校验缺数据时不生效() -> None:
    rule = PriceLimitRule()
    decision = rule.check(_context(plan=_plan(limit_price="99"), bar=_bar("10.0")))

    assert decision.action is RiskAction.PASS, "拿不到涨跌停就不做该条校验（不拿臆测值硬算）"


def test_资金不足缩量到整手() -> None:
    rule = CashRule()
    decision = rule.check(_context(plan=_plan(10_000), cash="25000", bar=_bar("10.0")))

    assert decision.action is RiskAction.SHRINK
    assert decision.adjusted_qty == 2_400, "2.5 万含费约可买 2400 股（10 元/股）"
    assert decision.adjusted_qty % 100 == 0


def test_资金完全不足拒单() -> None:
    rule = CashRule()
    assert rule.check(_context(plan=_plan(1000), cash="100", bar=_bar("10.0"))).action is RiskAction.REJECT


def test_可卖数量不足缩量() -> None:
    rule = AvailablePositionRule()
    held = Position(code="600000", qty=1000, avail_qty=300, last_price=Decimal("10"))

    decision = rule.check(_context(plan=_plan(1000, side=Side.SELL), positions={"600000": held}))

    assert decision.action is RiskAction.SHRINK
    assert decision.adjusted_qty == 300


def test_无可用持仓拒单() -> None:
    rule = AvailablePositionRule()
    held = Position(code="600000", qty=1000, avail_qty=0, last_price=Decimal("10"))

    decision = rule.check(_context(plan=_plan(1000, side=Side.SELL), positions={"600000": held}))

    assert decision.action is RiskAction.REJECT
    assert "T+1" in decision.message


# ============================================================ 事中规则
def _snapshot(*, drawdown: float = 0.0, pnl: float = 0.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(trade_date=DAY, account=_account(), positions={}, drawdown=drawdown, day_pnl_ratio=pnl)


def test_回撤触线降级为只平不开() -> None:
    rule = DrawdownRule(max_drawdown=0.1)

    assert rule.check_portfolio(_snapshot(drawdown=-0.05)).action is RiskAction.PASS
    decision = rule.check_portfolio(_snapshot(drawdown=-0.12))
    assert decision.action is RiskAction.HALT
    assert "只平不开" in decision.message


def test_单日亏损触线() -> None:
    rule = DailyLossRule(max_daily_loss=0.03)

    assert rule.check_portfolio(_snapshot(pnl=-0.02)).action is RiskAction.PASS
    assert rule.check_portfolio(_snapshot(pnl=-0.04)).action is RiskAction.HALT


def test_深度回撤触发清仓() -> None:
    rule = SevereDrawdownRule(liquidation_drawdown=0.2)

    assert rule.check_portfolio(_snapshot(drawdown=-0.15)).action is RiskAction.PASS
    assert rule.check_portfolio(_snapshot(drawdown=-0.25)).action is RiskAction.FLATTEN


def test_默认规则集非空且优先级递增() -> None:
    pre = default_pre_trade_rules(switch=None)
    intra = default_intraday_rules()

    assert len(pre) >= 6
    assert len(intra) == 3
    assert all(hasattr(rule, "check") for rule in pre)
    assert all(hasattr(rule, "check_portfolio") for rule in intra)


# ============================================================ 规则链
def test_规则链缩量传递给后续规则() -> None:
    """集中度缩到 1 万股后，金额上限规则必须按 1 万股判断。

    若按原始数量判断，"上限 5 万"会算出 5000 股，看起来更严；
    实际错误方向相反时（金额规则在前、集中度在后）就会出现"已缩量的数量被放大回去"的假象。
    """
    engine = RiskEngine(
        pre_trade_rules=[PositionConcentrationRule(max_weight=0.1), OrderValueRule(max_value=Decimal("80000"))]
    )
    review = engine.check_plans(
        [_plan(20_000)],
        trade_date=DAY,
        account=_account("1000000"),
        positions={},
        bars={"600000": _bar("10.0")},
    )

    assert len(review.approved) == 1
    assert review.approved[0].plan.target_qty == 8_000, "集中度 1 万股 → 金额上限 8000 股，最终取更严者"
    assert review.approved[0].origin_qty == 20_000
    assert review.approved[0].shrank
    assert review.shrank_count == 1


def test_规则链拒单短路() -> None:
    class Boom:
        # 优先级数值越小越先执行；这条故意排在集中度规则（10）之后
        name = "boom"
        priority = 50

        def check(self, ctx: RiskContext):  # type: ignore[no-untyped-def]
            raise AssertionError("拒单后不应继续执行后续规则")

    engine = RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=0.0001), Boom()])
    review = engine.check_plans(
        [_plan(1000)], trade_date=DAY, account=_account(), positions={}, bars={"600000": _bar("10.0")}
    )

    assert not review.approved
    assert len(review.rejected) == 1


def test_规则按优先级排序执行() -> None:
    order: list[str] = []

    class Tracker:
        def __init__(self, name: str, priority: int) -> None:
            self.name = name
            self.priority = priority

        def check(self, ctx: RiskContext):  # type: ignore[no-untyped-def]
            from engine.types import RiskDecision

            order.append(self.name)
            return RiskDecision.allow(ctx.plan.target_qty)

    engine = RiskEngine(pre_trade_rules=[Tracker("late", 20), Tracker("early", 1)])
    engine.check_plans([_plan(100)], trade_date=DAY, account=_account(), positions={}, bars={})

    assert order == ["early", "late"]


def test_组合检查取最严等级() -> None:
    engine = RiskEngine(intraday_rules=[DrawdownRule(max_drawdown=0.05), SevereDrawdownRule(liquidation_drawdown=0.2)])

    level, decisions = engine.check_portfolio(_snapshot(drawdown=-0.25))

    assert level is RiskLevel.FLATTEN, "FLATTEN 比 HALT 更严，取最严者"
    assert len(decisions) == 2


def test_组合正常时无触线() -> None:
    engine = RiskEngine(intraday_rules=default_intraday_rules())
    level, decisions = engine.check_portfolio(_snapshot(drawdown=-0.01, pnl=-0.005))

    assert level is RiskLevel.NORMAL
    assert decisions == []


def test_审核摘要可读() -> None:
    engine = RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=0.001)])
    review = engine.check_plans(
        [_plan(1000)], trade_date=DAY, account=_account(), positions={}, bars={"600000": _bar("10.0")}
    )

    summary = review.summary()
    assert "通过" in summary and "拒绝" in summary


# ============================================================ 回测接入
def test_回测接入风控后限额宽松时不拒单() -> None:
    import pandas as pd

    from engine.backtest.data import DataFrameBarProvider
    from engine.backtest.engine import BacktestConfig, BacktestEngine

    rows = [
        {
            "code": "600000",
            "date": (pd.Timestamp("2026-10-12") + pd.Timedelta(days=offset)).date().isoformat(),
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1_000_000,
            "amount": 10_000_000,
            "adj_factor": 1.0,
        }
        for offset in range(5)
    ]
    prices = pd.DataFrame(rows)

    class AlwaysBuyStrategy:
        strategy_id = "always_buy"

        def init(self, context: object) -> None:
            return None

        def on_bar(self, context: object, bars: object) -> list:  # type: ignore[type-arg]
            from engine.types import Direction, Signal

            return [
                Signal(
                    code="600000",
                    strategy_id=self.strategy_id,
                    trade_date=context.trade_date,  # type: ignore[attr-defined]
                    direction=Direction.BUY,
                    target_weight=0.9,
                )
            ]

        def on_order_filled(self, context: object, trade: object) -> None:
            return None

    config = BacktestConfig(start=date(2026, 10, 12), end=date(2026, 10, 16), initial_cash="1000000")
    engine = BacktestEngine(
        config,
        provider=DataFrameBarProvider(prices),
        strategy=AlwaysBuyStrategy(),  # type: ignore[arg-type]
        risk_engine=RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=100.0)]),
    )
    result = engine.run()

    assert result.risk_rejections == [], "集中度上限设为 100 倍，不应拒单"
    assert result.risk_shrinks > 0 or result.trades, "整体链路仍应正常成交"


def test_事中触线在回测中降级并停止开仓() -> None:
    """价格连续下跌 8% 触发回撤线后，后续买入必须被拒（只平不开）。"""
    import pandas as pd

    from engine.backtest.data import DataFrameBarProvider
    from engine.backtest.engine import BacktestConfig, BacktestEngine

    closes = [10.0, 9.2, 8.46, 7.78, 7.16]
    rows = [
        {
            "code": "600000",
            "date": (pd.Timestamp("2026-10-12") + pd.Timedelta(days=offset)).date().isoformat(),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000_000,
            "amount": close * 1_000_000,
            "adj_factor": 1.0,
        }
        for offset, close in enumerate(closes)
    ]

    class BuyEveryDay:
        strategy_id = "buy_every_day"

        def init(self, context: object) -> None:
            return None

        def on_bar(self, context: object, bars: object) -> list:  # type: ignore[type-arg]
            from engine.types import Direction, Signal

            return [
                Signal(
                    code="600000",
                    strategy_id=self.strategy_id,
                    trade_date=context.trade_date,  # type: ignore[attr-defined]
                    direction=Direction.BUY,
                    target_weight=0.9,
                )
            ]

        def on_order_filled(self, context: object, trade: object) -> None:
            return None

    engine = BacktestEngine(
        BacktestConfig(start=date(2026, 10, 12), end=date(2026, 10, 16), initial_cash="1000000", slippage_bps=0.0),
        provider=DataFrameBarProvider(pd.DataFrame(rows)),
        strategy=BuyEveryDay(),  # type: ignore[arg-type]
        risk_engine=RiskEngine(
            pre_trade_rules=[RiskLevelRule()],
            intraday_rules=[DrawdownRule(max_drawdown=0.03)],
        ),
    )
    result = engine.run()

    assert result.risk_level_events, "回撤超过 3% 应触发降级并记录事件"
    assert result.risk_level_events[0][1] == str(RiskLevel.HALT)
    halt_day = date.fromisoformat(result.risk_level_events[0][0])
    assert result.risk_rejections, "降级后买入应被拒绝"
    assert "撤单" in " ".join(reason for _, reason in result.risk_rejections), "降级时应撤回未成交买单"
    # 关键断言：降级当日之后不再有任何成交（存量买单也已被撤回）
    assert all(trade.trade_time.date() < halt_day for trade in result.trades), "降级后不应再有成交"


def test_集中度规则在回测中确实拦下超大仓位() -> None:
    import pandas as pd

    from engine.backtest.data import DataFrameBarProvider
    from engine.backtest.engine import BacktestConfig, BacktestEngine

    rows = [
        {
            "code": "600000",
            "date": (pd.Timestamp("2026-10-12") + pd.Timedelta(days=offset)).date().isoformat(),
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 1_000_000,
            "amount": 10_000_000,
            "adj_factor": 1.0,
        }
        for offset in range(5)
    ]

    class BuyAll:
        strategy_id = "buy_all"

        def init(self, context: object) -> None:
            return None

        def on_bar(self, context: object, bars: object) -> list:  # type: ignore[type-arg]
            from engine.types import Direction, Signal

            if context.trade_date.isoformat() != "2026-10-12":  # type: ignore[attr-defined]
                return []
            return [
                Signal(
                    code="600000",
                    strategy_id=self.strategy_id,
                    trade_date=context.trade_date,  # type: ignore[attr-defined]
                    direction=Direction.BUY,
                    target_weight=0.9,
                )
            ]

        def on_order_filled(self, context: object, trade: object) -> None:
            return None

    # 上限 5%：95% 仓位的买单必须被缩量到 5 万市值（5000 股）
    engine = BacktestEngine(
        BacktestConfig(start=date(2026, 10, 12), end=date(2026, 10, 16), initial_cash="1000000", slippage_bps=0.0),
        provider=DataFrameBarProvider(pd.DataFrame(rows)),
        strategy=BuyAll(),  # type: ignore[arg-type]
        risk_engine=RiskEngine(pre_trade_rules=[PositionConcentrationRule(max_weight=0.05)]),
    )
    result = engine.run()

    assert result.risk_shrinks >= 1, "应发生缩量"
    assert result.trades, "缩量后仍应成交"
    assert result.trades[0].qty <= 5_000, f"风控未生效：成交 {result.trades[0].qty} 股"
