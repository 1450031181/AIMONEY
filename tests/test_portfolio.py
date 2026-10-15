"""P3-1 组合层测试：分配方案与约束全部用手算值验证。

约束类逻辑最容易出的错是"悄悄改变了总仓位"——上限截断后不重新分配、
剔除碎仓后不重新分配，都会让组合多出莫名奇妙的现金敞口，而且不会报任何错。
因此本文件重点断言**权重之和**与**上限是否被突破**。
"""

from __future__ import annotations

import pytest

from engine.errors import ValidationError
from engine.portfolio.allocator import SUPPORTED_SCHEMES, allocate, normalize, realized_volatility, top_k
from engine.portfolio.builder import PortfolioBuilder
from engine.portfolio.constraints import (
    PortfolioConstraints,
    apply_no_trade_band,
    cap_weights,
    fit_to_budget,
    limit_turnover,
    scale_to_budget,
    turnover,
)


# ============================================================ 分配方案
def test_等权分配() -> None:
    weights = allocate({"a": 5.0, "b": 1.0, "c": -3.0})
    assert all(value == pytest.approx(1 / 3) for value in weights.values()), "等权与得分无关"


def test_得分为负的标的被排除() -> None:
    weights = allocate({"a": 2.0, "b": 1.0, "c": -1.0}, scheme="score")
    assert "c" not in weights, "long-only 下负分标的没有持有依据"
    assert weights["a"] == pytest.approx(2 / 3)
    assert weights["b"] == pytest.approx(1 / 3)


def test_全为负分时返回空组合() -> None:
    assert allocate({"a": -1.0, "b": -2.0}, scheme="score") == {}, "宁可空仓，也不挑'最不差'的买入"


def test_排名分配() -> None:
    weights = allocate({"a": 5.0, "b": 3.0, "c": 1.0}, scheme="rank")
    assert weights["a"] == pytest.approx(3 / 6)
    assert weights["b"] == pytest.approx(2 / 6)
    assert weights["c"] == pytest.approx(1 / 6)


def test_逆波动率分配() -> None:
    weights = allocate({"a": 1.0, "b": 1.0}, scheme="inverse_vol", volatilities={"a": 0.1, "b": 0.2})
    assert weights["a"] == pytest.approx(2 / 3), "波动越低权重越高"
    assert weights["b"] == pytest.approx(1 / 3)


def test_逆波动率缺波动数据时被排除() -> None:
    weights = allocate(
        {"a": 1.0, "b": 1.0, "c": 1.0}, scheme="inverse_vol", volatilities={"a": 0.1, "b": 0.0, "c": 0.2}
    )
    assert set(weights) == {"a", "c"}, "波动为零/缺失者无法度量风险，不参与（不用臆测值凑数）"


def test_逆波动率缺少波动率参数报错() -> None:
    with pytest.raises(ValidationError):
        allocate({"a": 1.0}, scheme="inverse_vol")


def test_未知分配方案报错() -> None:
    with pytest.raises(ValidationError):
        allocate({"a": 1.0}, scheme="magic")


def test_归一化夹掉负值() -> None:
    weights = normalize({"a": 3.0, "b": -1.0})
    assert weights == {"a": pytest.approx(1.0), "b": pytest.approx(0.0)}


def test_选前k名且并列时按代码稳定排序() -> None:
    selected = top_k({"b": 1.0, "a": 1.0, "c": 0.5}, 2)
    assert list(selected) == ["a", "b"], "得分并列时按代码排序，保证结果可复现"


def test_波动率手算() -> None:
    assert realized_volatility([0.01, -0.01, 0.01, -0.01]) == pytest.approx(0.011547, rel=1e-4)
    assert realized_volatility([0.01]) == 0.0


# ============================================================ 上限与预算
def test_上限截断后重新分配而不是丢掉() -> None:
    capped = cap_weights({"a": 0.8, "b": 0.1, "c": 0.1}, 0.5)

    assert capped["a"] == pytest.approx(0.5)
    assert capped["b"] == pytest.approx(0.25)
    assert capped["c"] == pytest.approx(0.25)
    assert sum(capped.values()) == pytest.approx(1.0), "多余权重必须重新分配，否则等于偷偷加现金仓位"


def test_容量不足时留现金而不破上限() -> None:
    capped = cap_weights({"a": 0.5, "b": 0.3, "c": 0.2}, 0.2)

    assert all(value <= 0.2 + 1e-12 for value in capped.values())
    assert sum(capped.values()) == pytest.approx(0.6), "3 只 × 上限 0.2 = 0.6，余下只能是现金"


def test_不设上限时权重不变() -> None:
    weights = {"a": 0.9, "b": 0.1}
    assert cap_weights(weights, None) == pytest.approx(weights)


def test_现金缓冲与下限() -> None:
    weights, dropped = fit_to_budget({"a": 0.98, "b": 0.02}, PortfolioConstraints(min_weight=0.05, cash_buffer=0.05))

    assert "b" in dropped
    assert weights["a"] == pytest.approx(0.95), "剔除碎仓后重新分配到预算上限"
    assert sum(weights.values()) == pytest.approx(0.95)


def test_上下限同时生效且总量不超预算() -> None:
    weights, _ = fit_to_budget(
        {"a": 0.6, "b": 0.3, "c": 0.1},
        PortfolioConstraints(min_weight=0.05, max_weight=0.5, cash_buffer=0.1),
    )

    assert all(0.05 - 1e-9 <= value <= 0.5 + 1e-9 for value in weights.values())
    assert sum(weights.values()) <= 0.9 + 1e-9, "总仓位不得超过预算（1 − 现金缓冲）"


def test_归一化后仍低于下限的标的被全部剔除() -> None:
    """输入是**相对权重**：先归一化到预算，再判定下限。

    30 只等权 → 每只 3.3%，低于 5% 的下限，于是全部被剔除 → 空仓。
    这条语义很重要：min_weight 判的是"实际会持有的权重"，不是输入数值本身。
    """
    weights, dropped = fit_to_budget(
        {f"c{index:02d}": 1.0 for index in range(30)}, PortfolioConstraints(min_weight=0.05)
    )

    assert weights == {}
    assert len(dropped) == 30


def test_预算缩放保留相对关系() -> None:
    scaled = scale_to_budget({"a": 0.75, "b": 0.25}, 0.5)
    assert scaled == {"a": pytest.approx(0.375), "b": pytest.approx(0.125)}


def test_约束参数校验() -> None:
    with pytest.raises(ValidationError):
        PortfolioConstraints(max_weight=0.0)
    with pytest.raises(ValidationError):
        PortfolioConstraints(min_weight=0.6, max_weight=0.5)
    with pytest.raises(ValidationError):
        PortfolioConstraints(cash_buffer=1.0)
    with pytest.raises(ValidationError):
        PortfolioConstraints(max_turnover=0.0)
    with pytest.raises(ValidationError):
        PortfolioConstraints(no_trade_band=-0.1)


# ============================================================ 换手
def test_换手率口径() -> None:
    assert turnover({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert turnover({}, {"a": 1.0}) == pytest.approx(0.5), "全部清仓 = 换手 100%"
    assert turnover({"a": 1.0}, {}) == pytest.approx(0.5)


def test_无交易带抑制微小变化() -> None:
    adjusted = apply_no_trade_band({"a": 0.52, "b": 0.48}, {"a": 0.5, "b": 0.5}, 0.05)

    assert adjusted == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}, "变化小于阈值就不动"


def test_无交易带不抑制大变化() -> None:
    adjusted = apply_no_trade_band({"a": 0.8, "b": 0.2}, {"a": 0.5, "b": 0.5}, 0.05)
    assert adjusted["a"] == pytest.approx(0.8)


def test_换手上限把目标向当前收缩() -> None:
    """换手口径是 ``0.5×Σ|Δw|``，所以"清仓"的换手只有 50%。

    要把换手从 0.5 压到 0.3，需要走 0.3/0.5 = 60% 的路 → 权重落在 0.4 而不是 0.7。
    """
    adjusted, actual = limit_turnover({}, {"a": 1.0}, 0.3)

    assert actual == pytest.approx(0.3)
    assert adjusted["a"] == pytest.approx(0.4)
    assert turnover(adjusted, {"a": 1.0}) == pytest.approx(0.3), "调整后的实际换手必须正好等于上限"


def test_未超限时不调整() -> None:
    adjusted, actual = limit_turnover({"a": 0.5}, {"a": 0.5}, 0.3)
    assert adjusted == {"a": pytest.approx(0.5)}
    assert actual == pytest.approx(0.0)


# ============================================================ 组合构建
def test_构建器等权默认() -> None:
    decision = PortfolioBuilder(top_k=3).build({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5})

    assert set(decision.weights) == {"a", "b", "c"}
    assert decision.invested_weight == pytest.approx(1.0)
    assert decision.cash_weight == pytest.approx(0.0)
    assert "d" in decision.dropped and "未入选" in decision.dropped["d"]


def test_构建器现金缓冲() -> None:
    decision = PortfolioBuilder(top_k=2, constraints=PortfolioConstraints(cash_buffer=0.1)).build(
        {"a": 1.0, "b": 1.0}
    )

    assert decision.invested_weight == pytest.approx(0.9)
    assert decision.cash_weight == pytest.approx(0.1)


def test_构建器得分为负方案给空仓() -> None:
    decision = PortfolioBuilder(scheme="score", top_k=2).build({"a": -1.0, "b": -2.0})

    assert decision.weights == {}
    assert decision.cash_weight == pytest.approx(1.0)
    assert all("负" in reason for reason in decision.dropped.values())


def test_构建器记录当前权重与换手() -> None:
    decision = PortfolioBuilder(top_k=2).build({"a": 1.0, "b": 1.0}, current_weights={"a": 0.5})

    assert decision.current_weights == {"a": 0.5}
    assert decision.turnover == pytest.approx(0.5 * (abs(0.5 - 0.5) + abs(0.5 - 0.0)))


def test_构建器无得分时清空持仓() -> None:
    decision = PortfolioBuilder(top_k=2).build({}, current_weights={"a": 1.0})

    assert decision.weights == {}
    assert decision.turnover == pytest.approx(0.5)
    assert "a" in decision.dropped


def test_构建器参数校验() -> None:
    with pytest.raises(ValidationError):
        PortfolioBuilder(scheme="magic")
    with pytest.raises(ValidationError):
        PortfolioBuilder(top_k=0)


def test_构建器摘要可读() -> None:
    decision = PortfolioBuilder(top_k=2).build({"a": 1.0, "b": 1.0})
    summary = decision.summary()

    assert "等权" in summary or "equal" in summary
    assert "仓位" in summary and "换手" in summary


def test_分配方案清单完整() -> None:
    assert set(SUPPORTED_SCHEMES) == {"equal", "score", "rank", "inverse_vol"}
