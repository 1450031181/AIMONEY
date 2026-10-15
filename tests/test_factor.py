"""P2-F1 因子库/计算引擎/预处理测试。

指标类断言坚持"手工可算"：动量、波动率、偏离度都能用构造序列验证到小数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.errors import ValidationError
from engine.factor import (
    PreprocessConfig,
    compute_factors,
    get_factor,
    list_factors,
    preprocess_cross_section,
    winsorize_mad,
    zscore,
)


@pytest.fixture(autouse=True)
def _restore_factor_registry():
    """注册表是进程级全局状态：测试注入的假因子必须回滚，否则会污染后续用例。"""
    from engine.factor import base

    snapshot = dict(base._FACTOR_REGISTRY)
    yield
    base._FACTOR_REGISTRY.clear()
    base._FACTOR_REGISTRY.update(snapshot)


def _frame(closes: list[float], *, code: str = "600000", volume: float = 1_000_000) -> pd.DataFrame:
    """单标的行情帧（date 索引）。"""
    index = pd.DatetimeIndex([pd.Timestamp("2026-09-01") + pd.Timedelta(days=i) for i in range(len(closes))])
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [volume] * len(closes),
            "amount": [value * volume for value in closes],
        },
        index=index,
    )


def _long_frame(*codes: list[float]) -> pd.DataFrame:
    parts = []
    for index, closes in enumerate(codes):
        frame = _frame(closes, code=f"{index:06d}")
        parts.append(frame.reset_index(names="date").assign(code=f"{index:06d}"))
    return pd.concat(parts, ignore_index=True)


# ============================================================ 注册表
def test_内置因子数量与类目() -> None:
    factors = list_factors()
    assert len(factors) >= 20, "内置因子库应具备足够广度（动量/反转/波动/趋势/流动性/结构）"
    assert {item.category for item in factors} <= {"momentum", "reversal", "volatility", "trend", "liquidity", "structure"}


def test_重复注册被拒() -> None:
    from engine.factor.base import FactorDefinition, register_factor

    with pytest.raises(ValidationError):
        register_factor(
            FactorDefinition(
                name="mom_20",
                category="momentum",
                description="重复",
                window=21,
                func=lambda frame: frame["close"],
            )
        )


def test_未注册因子查询报错() -> None:
    with pytest.raises(ValidationError):
        get_factor("not_exists")


def test_窗口必须为正() -> None:
    from engine.factor.base import FactorDefinition, register_factor

    with pytest.raises(ValidationError):
        register_factor(
            FactorDefinition(name="bad_window", category="trend", description="x", window=0, func=lambda f: f["close"])
        )


# ============================================================ 因子数学
def test_动量手工验证() -> None:
    frame = _frame([10.0] * 20 + [12.0])
    mom_20 = get_factor("mom_20").func(frame)

    assert pd.isna(mom_20.iloc[-2]), "历史不足必须为 NaN，不允许硬算"
    assert mom_20.iloc[-1] == pytest.approx(0.2)


def test_波动率等于滚动收益标准差() -> None:
    closes = [10.0, 10.2, 10.1, 10.4, 10.3, 10.5, 10.4, 10.6, 10.5, 10.7, 10.6, 10.8]
    frame = _frame(closes)
    vol_10 = get_factor("vol_10").func(frame)

    returns = np.diff(closes) / np.array(closes[:-1])
    expected = np.std(returns[-10:], ddof=1),  # rolling(10) 用最近 10 个日收益
    assert vol_10.iloc[-1] == pytest.approx(expected[0])


def test_均线偏离手工验证() -> None:
    closes = [10.0] * 19 + [11.0, 12.0]
    frame = _frame(closes)
    bias_20 = get_factor("bias_20").func(frame)

    ma = (10.0 * 18 + 11.0 + 12.0) / 20  # rolling(20) 是最后 20 个收盘（位置 1..20）
    assert bias_20.iloc[-1] == pytest.approx(12.0 / ma - 1)


def test_高低点重合时位置因子为NaN而非inf() -> None:
    """构造 high == low 的真·横盘：分母为零必须返回 NaN，绝不允许 inf 进入因子值。"""
    closes = [10.0] * 25
    index = pd.DatetimeIndex([pd.Timestamp("2026-09-01") + pd.Timedelta(days=i) for i in range(len(closes))])
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000] * len(closes),
            "amount": [10_000_000] * len(closes),
        },
        index=index,
    )

    position = get_factor("close_position_20").func(frame)
    assert pd.isna(position.iloc[-1])


def test_放量因子手工验证() -> None:
    frame = _frame([10.0] * 20 + [10.5], volume=2_000_000)
    # 前 20 日量 1e6，最后一天 2e6
    frame.loc[frame.index[:-1], "volume"] = 1_000_000
    frame.loc[frame.index[:-1], "amount"] = 10_000_000
    ratio = get_factor("volume_ratio_20").func(frame)

    # 均值窗口含当日：(19×1e6 + 2e6)/20 = 1.05e6
    assert ratio.iloc[-1] == pytest.approx(2_000_000 / 1_050_000)


# ============================================================ 计算引擎
def test_多标的计算输出结构正确() -> None:
    long_frame = _long_frame([10.0, 10.5, 11.0], [20.0, 20.0, 20.0])
    result = compute_factors(long_frame, ["mom_5", "vol_10"])

    assert list(result.columns) == ["date", "code", "mom_5", "vol_10"]
    assert len(result) == 6
    assert set(result["code"]) == {"000000", "000001"}


def test_标的之间互相隔离() -> None:
    """000000 有 60 天历史、000001 只有 3 天：后者的 NaN 不得影响前者。"""
    long_history = [10.0 + i * 0.01 for i in range(60)]
    short_history = [20.0, 20.0, 20.0]
    result = compute_factors(_long_frame(long_history, short_history), ["mom_20"])

    first = result[result["code"] == "000000"].tail(1).iloc[0]
    second = result[result["code"] == "000001"].tail(1).iloc[0]

    assert first["mom_20"] == pytest.approx(long_history[-1] / long_history[-21] - 1)
    assert pd.isna(second["mom_20"]), "历史不足为 NaN"


def test_历史不足的位置为NaN() -> None:
    result = compute_factors(_long_frame([10.0, 10.1, 10.2]), ["mom_20"])
    assert result["mom_20"].isna().all()


def test_因子名重复被拒() -> None:
    with pytest.raises(ValidationError):
        compute_factors(_long_frame([10.0, 11.0]), ["mom_5", "mom_5"])


def test_未注册因子被拒() -> None:
    with pytest.raises(ValidationError):
        compute_factors(_long_frame([10.0, 11.0]), ["not_a_factor"])


def test_输入缺少必要列被拒() -> None:
    frame = _long_frame([10.0, 11.0]).drop(columns=["close"])
    with pytest.raises(ValidationError):
        compute_factors(frame, ["mom_5"])


# ============================================================ 截面预处理
def _cross_section() -> pd.DataFrame:
    """两个截面日、每个日 5 只标的（其中含一个极端值）。"""
    rows = []
    for day in ("2026-10-12", "2026-10-13"):
        for code, value in (("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0), ("e", 100.0)):
            rows.append({"date": day, "code": code, "f1": value})
    return pd.DataFrame(rows)


def test_去极值截断极端值() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    clipped = winsorize_mad(values)

    assert clipped.max() < 100.0
    assert clipped.iloc[:-1].tolist() == [1.0, 2.0, 3.0, 4.0], "正常值不应被改动"


def test_zscore均值零方差一() -> None:
    standardized = zscore(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert standardized.mean() == pytest.approx(0.0, abs=1e-12)
    assert standardized.std(ddof=0) == pytest.approx(1.0)


def test_zscore零方差时返回零() -> None:
    standardized = zscore(pd.Series([2.0, 2.0, 2.0]))
    assert (standardized == 0).all(), "常数截面标准化后应全 0，而不是除零得到 inf"


def test_预处理按日期截面独立() -> None:
    result = preprocess_cross_section(_cross_section(), ["f1"], PreprocessConfig(steps=["rank"]))

    day1 = result[result["date"] == "2026-10-12"].set_index("code")["f1"]
    day2 = result[result["date"] == "2026-10-13"].set_index("code")["f1"]
    # 两个截面的排序相同 → 分位相同；若跨日期混算，数值会不同
    assert day1["e"] == day2["e"] == 1.0
    assert day1["a"] == day2["a"] == 0.2


def test_默认流水线先rank后zscore() -> None:
    result = preprocess_cross_section(_cross_section(), ["f1"])

    day1 = result[result["date"] == "2026-10-12"].set_index("code")["f1"]
    assert day1["e"] > day1["d"] > day1["a"], "极端值截断/排名后排序保持"
    assert abs(day1.mean()) < 1e-9, "zscore 后截面均值为 0"
    # 未处理的极端值 100 不应残留在结果里
    assert day1["e"] < 2.0


def test_未知预处理步骤被拒() -> None:
    with pytest.raises(ValidationError):
        PreprocessConfig(steps=("normalize",))


def test_待处理列不存在被拒() -> None:
    with pytest.raises(ValidationError):
        preprocess_cross_section(_cross_section(), ["not_a_factor"])
