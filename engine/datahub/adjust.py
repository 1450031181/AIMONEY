"""复权处理（T-09）——本项目最容易被静默算错的地方。

**三种价格口径的职责必须严格分开**（设计文档 5.4）：

| 用途 | 口径 | 公式 |
| --- | --- | --- |
| 回测内部计算（收益、因子、净值） | 后复权 | 直接用存储的 ``close`` 等列 |
| 下单 / 撮合 / 涨跌停校验 | 真实价 | ``真实价 = 后复权价 / 复权因子``，按 0.01 取整 |
| 报告展示 | 前复权 | ``前复权价 = 后复权价 / 最新复权因子`` |

其中 **复权因子 ``F(t) = 后复权价 / 真实价``**（即 ``hfq = 真实价 × F``）。

为什么必须分开：跨除权日直接比较真实价，会把"10 派 3"的除息当成 3% 的暴跌计入收益；
反过来用后复权价下单，则会下出远离盘口的废单。

为什么因子必须落盘、不允许缺省为 1：因子错 → 还原出的下单价错 → **真实亏损**。
因此本模块对缺失/非正因子一律报错，绝不"当作 1 处理"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from engine.datahub.rules import infer_board, price_limit_ratio
from engine.datahub.schema import PRICE_COLUMNS
from engine.errors import DataQualityError, ValidationError
from engine.num import PRICE_TICK, to_decimal

FACTOR_COLUMN = "adj_factor"
#: 连续性校验容许的额外缓冲：停牌复牌、除权处理口径差异等
CONTINUITY_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class ContinuityViolation:
    """相邻交易日价格跳变超出涨跌停限制（说明复权因子缺失或错误）。"""

    code: str
    date: str
    previous_close: float
    close: float
    jump_ratio: float
    allowed_ratio: float


# ============================================================ 单值换算
def restore_real_price(
    adjusted_price: Decimal | float | str,
    adj_factor: Decimal | float | str,
    *,
    tick: Decimal = PRICE_TICK,
) -> Decimal:
    """后复权价 → 真实价（下单价），按最小报价单位四舍五入。"""
    price = to_decimal(adjusted_price)
    factor = to_decimal(adj_factor)
    if factor <= 0:
        raise ValidationError("复权因子必须为正，拒绝用非法因子还原价格", detail={"factor": str(factor)})
    return (price / factor).quantize(tick, rounding=ROUND_HALF_UP)


def to_adjusted_price(real_price: Decimal | float | str, adj_factor: Decimal | float | str) -> Decimal:
    """真实价 → 后复权价。"""
    price = to_decimal(real_price)
    factor = to_decimal(adj_factor)
    if factor <= 0:
        raise ValidationError("复权因子必须为正", detail={"factor": str(factor)})
    return price * factor


# ============================================================ 批量换算
def ensure_factor_column(frame: pd.DataFrame, *, dataset: str = "daily") -> None:
    """校验复权因子列存在且全部为正。

    缺列或出现 0/负值都直接报错：这两种情况都会让下单价变成垃圾值。
    """
    if FACTOR_COLUMN not in frame.columns:
        raise DataQualityError(
            f"{dataset} 缺少复权因子列，无法还原真实价",
            detail={"dataset": dataset, "missing": FACTOR_COLUMN},
        )
    if frame.empty:
        return
    factors = pd.to_numeric(frame[FACTOR_COLUMN], errors="coerce")
    bad = factors.isna() | (factors <= 0)
    if bad.any():
        sample = frame.loc[bad].head(5).to_dict(orient="records")
        raise DataQualityError(
            f"{dataset} 复权因子存在非法值（缺失或 ≤ 0）",
            detail={"dataset": dataset, "bad_rows": int(bad.sum()), "sample": sample},
        )


def restore_real_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = PRICE_COLUMNS,
    tick: Decimal = PRICE_TICK,
    validate: bool = True,
) -> pd.DataFrame:
    """整表后复权价 → 真实价（下单/撮合用），保留其他列不变。

    逐元素用 ``Decimal`` 四舍五入：pandas 的 ``round`` 是"四舍六入五成双"，
    与交易所"四舍五入到 0.01"的口径不一致，会偶发差 1 分。本函数用于下单/撮合，
    输入规模是单标的少量行，性能不是问题，正确性优先。
    """
    if validate:
        ensure_factor_column(frame)
    result = frame.copy()
    if result.empty:
        return result
    factor = pd.to_numeric(result[FACTOR_COLUMN], errors="coerce")
    for column in columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        result[column] = [
            None if pd.isna(value) or pd.isna(factor_value) else float(restore_real_price(value, factor_value, tick=tick))
            for value, factor_value in zip(values, factor, strict=True)
        ]
    return result


def to_pre_adjusted(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = PRICE_COLUMNS,
    validate: bool = True,
) -> pd.DataFrame:
    """后复权价 → 前复权价（仅用于展示）。

    前复权以**最新一期因子**为基准，使最新价等于真实价；因此它依赖入参区间的最新因子，
    区间不同结果不同——这正是前复权不能进回测计算的原因。
    """
    if validate:
        ensure_factor_column(frame)
    result = frame.copy()
    if result.empty:
        return result
    latest_factor = pd.to_numeric(result[FACTOR_COLUMN], errors="coerce").iloc[-1]
    if not latest_factor or latest_factor <= 0:
        raise ValidationError("无法确定最新复权因子，拒绝计算前复权", detail={"factor": str(latest_factor)})
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce") / latest_factor
    return result


# ============================================================ 连续性校验
def find_continuity_violations(
    frame: pd.DataFrame,
    *,
    price_column: str = "close",
    code_column: str = "code",
    date_column: str = "date",
    is_st_column: str | None = "is_st",
    tolerance: float = CONTINUITY_TOLERANCE,
) -> list[ContinuityViolation]:
    """找出相邻交易日价格跳变超出涨跌停限制的记录。

    这是**复权因子缺失/错误的探针**：后复权序列本应连续，一旦某天没乘上因子，
    就会出现远大于涨跌停幅度的跳变（例如 10 派 3 会造出约 30% 的假跌幅）。
    """
    if frame.empty:
        return []
    # 向量化实现：全市场 10 年数据约 1200 万行，逐行 iterrows 会慢到不可用
    ordered = frame.sort_values([code_column, date_column]).reset_index(drop=True)
    prices = pd.to_numeric(ordered[price_column], errors="coerce")
    previous = prices.groupby(ordered[code_column]).shift(1)
    jump = (prices - previous) / previous

    codes = ordered[code_column].astype(str)
    if is_st_column and is_st_column in ordered.columns:
        st_flags = ordered[is_st_column].fillna(False).astype(bool)
    else:
        st_flags = pd.Series(False, index=ordered.index)
    allowed = pd.Series(
        [
            float(price_limit_ratio(infer_board(code), is_st=bool(is_st))) + tolerance
            for code, is_st in zip(codes, st_flags, strict=True)
        ],
        index=ordered.index,
    )

    suspicious = previous.notna() & jump.notna() & (previous > 0) & (jump.abs() > allowed)
    return [
        ContinuityViolation(
            code=codes.iloc[position],
            date=str(ordered[date_column].iloc[position]),
            previous_close=float(previous.iloc[position]),
            close=float(prices.iloc[position]),
            jump_ratio=round(float(jump.iloc[position]), 6),
            allowed_ratio=round(float(allowed.iloc[position]), 6),
        )
        for position in ordered.index[suspicious]
    ]


def assert_continuity(frame: pd.DataFrame, *, dataset: str = "daily", **kwargs: object) -> None:
    """连续性校验失败即抛 ``DataQualityError``（供同步/盘前任务调用）。"""
    violations = find_continuity_violations(frame, **kwargs)  # type: ignore[arg-type]
    if violations:
        raise DataQualityError(
            f"{dataset} 存在超出涨跌停限制的价格跳变，疑似复权因子缺失或错误",
            detail={
                "dataset": dataset,
                "violations": len(violations),
                # 用 asdict 而非 __dict__：frozen+slots 的 dataclass 没有 __dict__
                "sample": [asdict(item) for item in violations[:5]],
            },
        )


__all__ = [
    "CONTINUITY_TOLERANCE",
    "FACTOR_COLUMN",
    "ContinuityViolation",
    "assert_continuity",
    "ensure_factor_column",
    "find_continuity_violations",
    "restore_real_frame",
    "restore_real_price",
    "to_adjusted_price",
    "to_pre_adjusted",
]
