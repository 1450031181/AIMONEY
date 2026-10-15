"""内置因子库（P2-F1）。

全部基于**日线量价数据**（OHLCV + amount）——不依赖流通股本、行业等尚未接入的数据，
因此每个因子都能在现有数据底座上直接计算。接入财务/另类数据后会按类目继续扩充。

命名约定：``<语义>_<窗口>``，窗口直接体现在名字里（参数化变体显式注册为独立因子），
这样评价结果、缓存键、看板展示都是自解释的。

所有因子对"历史不足"的处理统一为 **NaN**（rolling 的 ``min_periods=窗口``），
绝不硬算——短窗口算出的动量/波动率会系统性失真，这是最难察觉的脏数据。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from engine.factor.base import factor


# ============================================================ 公共子表达式
def _daily_return(frame: pd.DataFrame) -> pd.Series:
    return frame["close"].pct_change()


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """分母为 0 时返回 NaN（横盘时高低点重合、量为 0 等场景），绝不允许 inf 混进因子值。"""
    result = numerator / denominator
    return result.where(denominator != 0)


# ============================================================ 动量类
def _momentum(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame) -> pd.Series:
        return frame["close"] / frame["close"].shift(n) - 1

    return func


for _n in (5, 10, 20, 60):
    factor(name=f"mom_{_n}", category="momentum", description=f"{_n} 日动量", window=_n + 1)(_momentum(_n))

# ============================================================ 反转类
def _reversal(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame) -> pd.Series:
        return -(frame["close"] / frame["close"].shift(n) - 1)

    return func


for _n in (3, 5, 10):
    factor(name=f"rev_{_n}", category="reversal", description=f"{_n} 日短期反转（动量取负）", window=_n + 1)(
        _reversal(_n)
    )

# ============================================================ 波动率类
def _volatility(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame) -> pd.Series:
        return _daily_return(frame).rolling(n, min_periods=n).std(ddof=1)

    return func


for _n in (10, 20, 60):
    factor(name=f"vol_{_n}", category="volatility", description=f"{_n} 日收益波动率", window=_n + 1)(_volatility(_n))

# ============================================================ 趋势类
def _ma_bias(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame) -> pd.Series:
        ma = frame["close"].rolling(n, min_periods=n).mean()
        return frame["close"] / ma - 1

    return func


for _n in (10, 20, 60):
    factor(name=f"bias_{_n}", category="trend", description=f"收盘价相对 {_n} 日均线偏离", window=_n)(_ma_bias(_n))


def _high_ratio(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame) -> pd.Series:
        highest = frame["high"].rolling(n, min_periods=n).max()
        return _safe_div(frame["close"], highest)

    return func


for _n in (20, 60):
    factor(name=f"high_ratio_{_n}", category="trend", description=f"收盘价距 {_n} 日最高价比例（接近 1 = 创新高）", window=_n)(
        _high_ratio(_n)
    )

# ============================================================ 流动性类
@factor(name="volume_ratio_20", category="liquidity", description="当日成交量 / 20 日均量（放量程度）", window=20)
def _volume_ratio(frame: pd.DataFrame) -> pd.Series:
    ma = frame["volume"].rolling(20, min_periods=20).mean()
    return _safe_div(frame["volume"], ma)


@factor(name="amount_ratio_20", category="liquidity", description="当日成交额 / 20 日均额", window=20)
def _amount_ratio(frame: pd.DataFrame) -> pd.Series:
    ma = frame["amount"].rolling(20, min_periods=20).mean()
    return _safe_div(frame["amount"], ma)


# ============================================================ 价格结构类
@factor(name="range_20", category="structure", description="20 日平均振幅 (high-low)/close", window=20)
def _range_20(frame: pd.DataFrame) -> pd.Series:
    amplitude = _safe_div(frame["high"] - frame["low"], frame["close"])
    return amplitude.rolling(20, min_periods=20).mean()


def _close_position(n: int) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame) -> pd.Series:
        lowest = frame["low"].rolling(n, min_periods=n).min()
        highest = frame["high"].rolling(n, min_periods=n).max()
        return _safe_div(frame["close"] - lowest, highest - lowest)

    return func


for _n in (20, 60):
    factor(name=f"close_position_{_n}", category="structure", description=f"收盘价在 {_n} 日高低区间内的位置（0~1）", window=_n)(
        _close_position(_n)
    )


@factor(name="gap_20", category="structure", description="20 日平均隔夜跳空 open/prev_close - 1", window=21)
def _gap_20(frame: pd.DataFrame) -> pd.Series:
    overnight = frame["open"] / frame["close"].shift(1) - 1
    return overnight.rolling(20, min_periods=20).mean()


@factor(name="intraday_return_10", category="structure", description="10 日平均日内收益 close/open - 1", window=10)
def _intraday_10(frame: pd.DataFrame) -> pd.Series:
    intraday = _safe_div(frame["close"], frame["open"]) - 1
    return intraday.rolling(10, min_periods=10).mean()
