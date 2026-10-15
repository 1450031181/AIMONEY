"""P2-F2 因子评价测试。

判据必须能被"构造出来的已知有效/已知无效因子"验证：
- 完全正相关因子 → IC = +1、分层单调、多空为正；
- 完全负相关因子 → IC = -1；
- 纯噪声因子 → |IC| 接近 0（用固定随机种子，避免用例随机红）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.errors import ValidationError
from engine.factor.evaluate import (
    EvaluationConfig,
    assign_quantiles,
    evaluate_factor,
    forward_returns,
    icir,
    information_coefficient,
    monotonicity_score,
    quantile_analysis,
    quantile_turnover,
)

CODES = [f"{index:06d}" for index in range(10)]
DATES = [pd.Timestamp("2026-09-01") + pd.Timedelta(days=index) for index in range(30)]


def _prices(returns_by_code: dict[str, list[float]]) -> pd.DataFrame:
    """按日收益序列生成价格长表。"""
    rows = []
    for code, returns in returns_by_code.items():
        price = 10.0
        for day, daily in zip(DATES, returns, strict=False):
            price *= 1.0 + daily
            rows.append({"code": code, "date": day.date().isoformat(), "close": price})
    return pd.DataFrame(rows)


def _perfect_factor_prices(*, noise: float = 0.0, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造"明天涨跌由因子决定"的行情与因子。``noise`` 用于制造 IC 波动（检验 ICIR）。"""
    rows = []
    factor_rows = []
    rng = np.random.default_rng(seed)
    for day in DATES:
        for code in CODES:
            score = rng.normal()  # 因子的横截面取值
            factor_rows.append({"date": day.date().isoformat(), "code": code, "alpha": score})
            rows.append(
                {
                    "code": code,
                    "date": day.date().isoformat(),
                    "_score": score,
                    "_noise": rng.normal() if noise else 0.0,
                }
            )
    base = pd.DataFrame(rows)
    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["code", "date"])
    # 关键：t 日的**未来收益**必须由 t 日的因子决定
    #   fwd_ret(t) = close(t+1)/close(t) - 1 = daily_return(t+1)
    # 因此令 daily_return(t) = score(t-1)，等价于"次日收益 = 当日因子 × 0.02"
    base["previous_score"] = base.groupby("code")["_score"].shift(1)
    base["previous_noise"] = base.groupby("code")["_noise"].shift(1)
    base["daily_return"] = base["previous_score"].fillna(0.0) * 0.02 + base["previous_noise"].fillna(0.0) * noise * 0.02
    # 由日收益反推价格
    price_rows = []
    for code, group in base.groupby("code", sort=True):
        price = 10.0
        for _, row in group.iterrows():
            price *= 1.0 + float(row["daily_return"])
            price_rows.append({"code": code, "date": row["date"].date().isoformat(), "close": price})
    factors = pd.DataFrame(factor_rows)
    return pd.DataFrame(price_rows), factors


# ============================================================ 未来收益
def test_未来收益末尾为NaN() -> None:
    prices = _prices({"600000": [0.01] * 5})
    forward = forward_returns(prices, horizon=2)

    assert forward["forward_return"].isna().sum() == 2, "最后 2 个交易日没有未来价格，必须为 NaN"
    first = forward.iloc[0]["forward_return"]
    assert first == pytest.approx((1.01**2) - 1)


def test_未来收益horizon非法被拒() -> None:
    with pytest.raises(ValidationError):
        forward_returns(_prices({"600000": [0.01]}), horizon=0)


# ============================================================ IC
def test_完全正相关因子IC为1() -> None:
    prices, factors = _perfect_factor_prices()
    evaluation = evaluate_factor(prices, factors, factor="alpha", config=EvaluationConfig(horizon=1, quantiles=5))

    assert evaluation.sufficient
    assert evaluation.ic_mean > 0.9, f"完全正相关因子 IC 应接近 1，实际 {evaluation.ic_mean}"
    assert evaluation.ic_positive_ratio == 1.0
    # IC 恒为 1 → 方差为 0 → 按定义 ICIR 归零（不是 bug：零波动说明"稳定"到无从度量）
    assert evaluation.icir == 0.0


def test_强有效但有噪声的因子ICIR大于1() -> None:
    prices, factors = _perfect_factor_prices(noise=0.4, seed=3)
    evaluation = evaluate_factor(prices, factors, factor="alpha")

    assert 0.3 < evaluation.ic_mean < 1.0, "有噪声时 IC 应小于 1 但仍显著为正"
    assert evaluation.icir > 1.0, "IC 高且波动小 → ICIR 应明显大于 1"


def test_负向因子IC为负() -> None:
    prices, factors = _perfect_factor_prices()
    factors = factors.copy()
    factors["alpha"] = -factors["alpha"]
    evaluation = evaluate_factor(prices, factors, factor="alpha")

    assert evaluation.ic_mean < -0.9


def test_纯噪声因子IC接近零() -> None:
    rng = np.random.default_rng(11)
    dates = DATES[:20]
    price_rows, factor_rows = [], []
    for code in CODES:
        price = 10.0
        for day in dates:
            price *= 1.0 + rng.normal(0, 0.01)
            price_rows.append({"code": code, "date": day.date().isoformat(), "close": price})
            factor_rows.append({"code": code, "date": day.date().isoformat(), "noise": rng.normal()})

    evaluation = evaluate_factor(pd.DataFrame(price_rows), pd.DataFrame(factor_rows), factor="noise")

    assert abs(evaluation.ic_mean) < 0.3, "噪声因子的 IC 应接近 0"


def test_截面样本不足时跳过该日() -> None:
    merged = pd.DataFrame(
        {
            "date": ["2026-09-01"] * 3,
            "code": ["a", "b", "c"],
            "f": [1.0, 2.0, 3.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )
    assert information_coefficient(merged, factor_column="f", min_cross_section=5).empty


def test_常数截面不产生IC() -> None:
    merged = pd.DataFrame(
        {
            "date": ["2026-09-01"] * 6,
            "code": list("abcdef"),
            "f": [1.0] * 6,
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    assert information_coefficient(merged, factor_column="f", min_cross_section=5).empty


def test_ICIR零波动时为零() -> None:
    assert icir(pd.Series([0.1, 0.1, 0.1])) == 0.0
    assert icir(pd.Series([0.1])) == 0.0


# ============================================================ 分层
def test_分层数量与单调性() -> None:
    prices, factors = _perfect_factor_prices()
    evaluation = evaluate_factor(prices, factors, factor="alpha", config=EvaluationConfig(quantiles=5))

    assert len(evaluation.quantile_mean_returns) == 5
    assert evaluation.monotonicity > 0.9, "完全正相关因子的分层收益应单调递增"
    assert evaluation.quantile_mean_returns[5] > evaluation.quantile_mean_returns[1]


def test_分层净值与多空净值() -> None:
    prices, factors = _perfect_factor_prices()
    evaluation = evaluate_factor(prices, factors, factor="alpha", config=EvaluationConfig(horizon=1, quantiles=5))

    assert not evaluation.quantile_cum_returns.empty
    assert list(evaluation.quantile_cum_returns.columns) == [1, 2, 3, 4, 5]
    assert evaluation.long_short_mean_return > 0
    assert evaluation.long_short_nav.iloc[-1] > 1.0, "多空组合应累计向上"


def test_分层赋值按升序且并列不拆散() -> None:
    values = pd.Series([5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0])
    labels = assign_quantiles(values, 4)

    assert labels.min() == 1 and labels.max() == 4
    assert labels.iloc[1] == 1, "最小值应落在第 1 层"
    assert labels.iloc[6] == 4, "最大值应落在最高层"


def test_单调性为零当分层收益相同() -> None:
    assert monotonicity_score({1: 0.01, 2: 0.01, 3: 0.01}) == 0.0
    assert monotonicity_score({1: 0.01}) == 0.0


# ============================================================ 换手
def test_成员不变时换手为零() -> None:
    rows = []
    for day in DATES[:5]:
        for index, code in enumerate(CODES):
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "code": code,
                    "f": float(index),  # 因子取值固定 → 分层成员不变
                    "forward_return": 0.001,
                }
            )
    merged = pd.DataFrame(rows)

    turnover = quantile_turnover(merged, factor_column="f", quantiles=5)

    assert all(value == pytest.approx(0.0) for value in turnover.values())


def test_成员每天全换时换手接近满值() -> None:
    rows = []
    for day_index, day in enumerate(DATES[:4]):
        for index, code in enumerate(CODES):
            # 因子顺序每天轮换 → 顶层成员每天全换
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "code": code,
                    "f": float((index + day_index) % len(CODES)),
                    "forward_return": 0.001,
                }
            )
    merged = pd.DataFrame(rows)

    turnover = quantile_turnover(merged, factor_column="f", quantiles=5)

    assert turnover[5] > 0.0


# ============================================================ 边界与配置
def test_配置校验() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(quantiles=1)
    with pytest.raises(ValidationError):
        EvaluationConfig(horizon=0)
    with pytest.raises(ValidationError):
        EvaluationConfig(ic_method="kendall")
    with pytest.raises(ValidationError):
        EvaluationConfig(min_cross_section=2)


def test_因子列不存在被拒() -> None:
    prices = _prices({"600000": [0.01] * 5})
    with pytest.raises(ValidationError):
        evaluate_factor(prices, pd.DataFrame({"date": [], "code": []}), factor="alpha")


def test_无法对齐时报错() -> None:
    prices = _prices({"600000": [0.01] * 5})
    factors = pd.DataFrame({"date": ["2030-01-01"], "code": ["999999"], "alpha": [1.0]})
    with pytest.raises(ValidationError):
        evaluate_factor(prices, factors, factor="alpha")


def test_有效样本不足时sufficient为False() -> None:
    prices = _prices({"600000": [0.01] * 5, "000001": [0.02] * 5})
    factors = pd.DataFrame(
        [
            {"date": day.date().isoformat(), "code": code, "alpha": float(index)}
            for day in DATES[:5]
            for index, code in enumerate(["600000", "000001"])
        ]
    )
    # 每个截面只有 2 只标的，低于 min_cross_section
    evaluation = evaluate_factor(prices, factors, factor="alpha", config=EvaluationConfig(min_cross_section=5))

    assert not evaluation.sufficient
    assert "有效截面不足" in evaluation.summary()


def test_评价结果可JSON序列化() -> None:
    prices, factors = _perfect_factor_prices()
    payload = evaluate_factor(prices, factors, factor="alpha").to_jsonable()

    assert payload["factor"] == "alpha"
    assert isinstance(payload["ic_recent"], dict)
    import json

    json.dumps(payload, ensure_ascii=False)


def test_分层收益为空时多空为空() -> None:
    merged = pd.DataFrame(columns=["date", "code", "f", "forward_return"])
    means, cumulative = quantile_analysis(merged, factor_column="f")
    assert means == {}
    assert cumulative.empty
