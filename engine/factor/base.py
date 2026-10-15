"""因子定义与注册表（P2-F1）。

为什么用注册表而不是散落的函数：因子是**有元数据的资产**（名字、类目、最少历史、
计算逻辑），评价、看板、缓存、防未来函数检查都要读这些元数据。散落的话，
"这个因子需要多少天历史"这类问题就只能靠口口相传。

计算函数契约：输入**单标的** DataFrame（date 升序索引，OHLCV+amount 列），
输出与索引对齐的 ``pd.Series``。逐标的独立计算是刻意的——
因子只允许看自身历史，跨标的泄漏在结构上不可能发生。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pandas as pd

from engine.errors import ValidationError

# 因子计算函数的输入类型：单标的、date 索引
FactorFunc = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    name: str
    category: str
    description: str
    #: 计算该因子需要的最少历史 bar 数（少于该值的结果为 NaN，而不是硬算）
    window: int
    func: FactorFunc


_FACTOR_REGISTRY: dict[str, FactorDefinition] = {}


def register_factor(definition: FactorDefinition) -> None:
    if definition.name in _FACTOR_REGISTRY:
        raise ValidationError("因子重复注册", detail={"name": definition.name})
    if definition.window <= 0:
        raise ValidationError("因子窗口必须为正", detail={"name": definition.name, "window": definition.window})
    if definition.category not in _ALLOWED_CATEGORIES:
        raise ValidationError(
            "因子类目不在允许范围内",
            detail={"category": definition.category, "allowed": sorted(_ALLOWED_CATEGORIES)},
        )
    _FACTOR_REGISTRY[definition.name] = definition


#: 类目收敛是为了看板与相关性分析有稳定的分组维度，随手造新类目会让分类失去意义
_ALLOWED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"momentum", "reversal", "volatility", "trend", "liquidity", "structure"}
)


def factor(name: str, *, category: str, description: str, window: int) -> Callable[[FactorFunc], FactorFunc]:
    """注册装饰器。import ``engine.factor.library`` 即完成全部内置因子注册。"""

    def decorator(func: FactorFunc) -> FactorFunc:
        register_factor(
            FactorDefinition(name=name, category=category, description=description, window=window, func=func)
        )
        return func

    return decorator


def get_factor(name: str) -> FactorDefinition:
    definition = _FACTOR_REGISTRY.get(name)
    if definition is None:
        raise ValidationError(
            "因子未注册",
            detail={"name": name, "available_count": len(_FACTOR_REGISTRY)},
        )
    return definition


def list_factors() -> tuple[FactorDefinition, ...]:
    return tuple(sorted(_FACTOR_REGISTRY.values(), key=lambda item: (item.category, item.name)))


def factor_names() -> tuple[str, ...]:
    return tuple(definition.name for definition in list_factors())


__all__ = [
    "FactorDefinition",
    "factor",
    "factor_names",
    "get_factor",
    "list_factors",
    "register_factor",
]
